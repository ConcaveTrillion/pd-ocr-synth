"""Semantic validation for loaded recipes.

Pydantic enforces the structural schema at load time (required keys,
literal types, schema_version drift). This module enforces the rules
that need filesystem or domain knowledge:

- font paths exist (and optional fonts are tolerated when missing)
- local corpus paths exist
- output destination is writable
- ``paper_texture`` texture directory exists
- ``publish.hf_dataset.description_file`` exists if specified
- degradation ``kind`` values are recognized
- layout ``mode`` is consistent with the keys actually set

Network-touching checks (web/wikisource reachability, HF dataset
existence) are out of scope for M02 — those land with the corpus
providers in M03.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pd_ocr_synth.recipe import Recipe
from pd_ocr_synth.recipe.models import SUPPORTED_SCHEMA_VERSIONS, Layout, LocalCorpus

# Built-in degradation kinds, per docs/specs/07-degradation.md. ``preset``
# is a structural marker (the loader expands it before this validator
# runs) — accepted here for forward-compat with M04+.
KNOWN_DEGRADATION_KINDS: frozenset[str] = frozenset(
    {
        # geometric
        "skew",
        "perspective",
        "scale",
        # optical
        "blur",
        "noise",
        "brightness",
        "contrast",
        "gamma",
        # print / paper
        "ink_bleed",
        "ink_thin",
        "paper_texture",
        "foxing",
        "bleed_through",
        "scratches",
        "fold_line",
        # compression
        "jpeg",
        "webp",
        # color space
        "grayscale",
        "binarize",
        # presets
        "preset",
    }
)


# Layout keys that are only meaningful for certain modes. Keys not
# listed are accepted in every mode.
_LAYOUT_KEYS_BY_MODE: dict[str, frozenset[str]] = {
    "word_crops": frozenset({"padding_px", "baseline_jitter_px"}),
    "lines": frozenset({"padding_px", "max_width_px", "line_spacing"}),
    # ``paragraph_spacing`` is the vertical gap between *paragraphs* on a
    # page, so it's only meaningful for ``pages`` mode — a
    # ``paragraphs`` sample renders a single paragraph by construction.
    "paragraphs": frozenset(
        {
            "padding_px",
            "max_width_px",
            "line_spacing",
            "paragraph_alignment",
        }
    ),
    "pages": frozenset(
        {
            "padding_px",
            "max_width_px",
            "line_spacing",
            "paragraph_spacing",
            "paragraph_indent_px",
            "paragraph_alignment",
            "page_size_px",
        }
    ),
}

# Per docs/specs/08-output-format.md §Modes:
#
#   Detection-mode rendering requires layout.mode in {paragraphs, pages}.
#   Recognition-mode rendering requires layout.mode in {word_crops, lines}.
#
# Anything outside these pairings is rejected at validate time so a
# malformed recipe can't reach the render writer.
_LAYOUT_MODES_BY_OUTPUT_MODE: dict[str, frozenset[str]] = {
    "recognition": frozenset({"word_crops", "lines"}),
    "detection": frozenset({"paragraphs", "pages"}),
}

# Severity levels. ``error`` blocks ``validate`` from exiting 0;
# ``warning`` is informational only.
Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    severity: Severity
    code: str
    message: str
    location: str | None = None

    def format(self) -> str:
        prefix = f"{self.severity.upper()} [{self.code}]"
        if self.location:
            return f"{prefix} {self.location}: {self.message}"
        return f"{prefix} {self.message}"


@dataclass(frozen=True, slots=True)
class ValidationReport:
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def errors(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[ValidationIssue, ...]:
        return tuple(i for i in self.issues if i.severity == "warning")

    @property
    def is_ok(self) -> bool:
        return not self.errors


def validate_recipe(recipe: Recipe, *, offline: bool = False) -> ValidationReport:
    """Run semantic validation on a loaded recipe.

    ``offline=True`` is reserved for future network-touching checks
    (M03+); M02 has none, so the flag is currently a no-op.
    """

    issues: list[ValidationIssue] = []
    issues.extend(_check_schema_version(recipe))
    issues.extend(_check_output(recipe))
    issues.extend(_check_fonts(recipe))
    issues.extend(_check_corpus(recipe))
    issues.extend(_check_text_transforms(recipe))
    issues.extend(_check_layout(recipe))
    issues.extend(_check_output_layout_pairing(recipe))
    issues.extend(_check_degradation(recipe))
    issues.extend(_check_publish(recipe))
    _ = offline  # placeholder for M03 wiring
    return ValidationReport(issues=tuple(issues))


def _check_schema_version(recipe: Recipe) -> list[ValidationIssue]:
    if recipe.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        # Defensive: pydantic would already have rejected this on load.
        return [
            ValidationIssue(
                severity="error",
                code="schema_version_unsupported",
                message=(
                    f"schema_version {recipe.schema_version} is not in "
                    f"{sorted(SUPPORTED_SCHEMA_VERSIONS)}"
                ),
                location="schema_version",
            )
        ]
    return []


def _check_output(recipe: Recipe) -> list[ValidationIssue]:
    dest = recipe.output.destination
    dest_str = str(dest)
    if "${" in dest_str or dest_str.startswith("~"):
        return [
            ValidationIssue(
                severity="error",
                code="output_destination_unresolved",
                message=(
                    f"destination still contains an unresolved env var or ~: {dest}. "
                    "Set the referenced variable in the environment, or use an absolute path."
                ),
                location="output.destination",
            )
        ]
    if not _writable_destination(dest):
        return [
            ValidationIssue(
                severity="error",
                code="output_destination_unwritable",
                message=(f"destination {dest} is not writable (no writable ancestor exists)."),
                location="output.destination",
            )
        ]
    return []


def _check_fonts(recipe: Recipe) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    for i, font in enumerate(recipe.fonts):
        if not font.path.exists():
            if font.optional:
                out.append(
                    ValidationIssue(
                        severity="warning",
                        code="optional_font_missing",
                        message=(
                            f"optional font not present at {font.path}; "
                            "will be skipped at render time"
                        ),
                        location=f"fonts[{i}].path",
                    )
                )
            else:
                out.append(
                    ValidationIssue(
                        severity="error",
                        code="font_missing",
                        message=f"font file does not exist: {font.path}",
                        location=f"fonts[{i}].path",
                    )
                )
            continue
        # File exists — try to inspect it.
        from pd_ocr_synth.fonts import FontOpenError, open_font

        try:
            info = open_font(font.path)
        except FontOpenError as exc:
            out.append(
                ValidationIssue(
                    severity="error",
                    code="font_unreadable",
                    message=f"could not open font {font.path}: {exc}",
                    location=f"fonts[{i}].path",
                )
            )
            continue
        if info.num_glyphs == 0 or not info.codepoints:
            out.append(
                ValidationIssue(
                    severity="error",
                    code="font_empty",
                    message=f"font {font.path} reports zero glyphs or empty cmap",
                    location=f"fonts[{i}].path",
                )
            )
    return out


def _registered_corpus_provider_types() -> frozenset[str]:
    """Return the set of corpus ``type`` discriminators the runtime can dispatch.

    Triggers the lazy default-registry build (which registers M03's
    builtins). Wrapped in a helper so tests can patch / reuse this when
    asserting against the runtime set, mirroring the
    ``_registered_degradation_kinds`` shape used for degradation kinds.
    """

    from pd_ocr_synth.corpus.registry import default_registry

    return frozenset(default_registry().types())


def _check_corpus(recipe: Recipe) -> list[ValidationIssue]:
    """Per-entry corpus validation.

    - ``local_corpus_missing`` (error): ``LocalCorpus`` paths that don't
      resolve to a file on disk.
    - ``corpus_provider_not_implemented`` (error): the recipe model
      accepts the ``type`` discriminator (so YAML loads cleanly) but
      the M03 runtime registry doesn't ship a provider for it. Render
      would raise ``ProviderError(f"unknown corpus provider …")`` in
      :func:`pd_ocr_synth.corpus.runner.run_providers` deep into the
      pipeline; surfacing it at validate time mirrors the iter-65
      ``degradation_kind_not_implemented`` precedent so users discover
      the gap before paying corpus-fetch + setup costs. Currently this
      catches ``hf_dataset`` (modelled in ``recipe.models`` but not yet
      registered with ``default_registry``); see
      ``docs/roadmap/03-corpus.md`` "Built-in providers" for status.
    """

    out: list[ValidationIssue] = []
    registered = _registered_corpus_provider_types()
    for i, entry in enumerate(recipe.corpus):
        if isinstance(entry, LocalCorpus) and not entry.path.exists():
            out.append(
                ValidationIssue(
                    severity="error",
                    code="local_corpus_missing",
                    message=f"local corpus path does not exist: {entry.path}",
                    location=f"corpus[{i}].path",
                )
            )
        # Recipe-model-accepted-but-not-registered: surface up front
        # rather than at first fetch. ``entry.type`` is the discriminator
        # literal pydantic enforces on the corpus union.
        if entry.type not in registered:
            out.append(
                ValidationIssue(
                    severity="error",
                    code="corpus_provider_not_implemented",
                    message=(
                        f"corpus provider '{entry.type}' is in the recipe schema but not "
                        f"yet implemented by the M03 runtime; render would raise. "
                        f"Implemented providers: {', '.join(sorted(registered))}. "
                        "See docs/roadmap/03-corpus.md (Built-in providers)."
                    ),
                    location=f"corpus[{i}].type",
                )
            )
    return out


def _registered_text_transform_names() -> frozenset[str]:
    """Return the set of text-transform names the runtime registry can dispatch.

    Triggers the lazy default-registry build (which registers M04's
    builtins). Wrapped in a helper so tests can patch / reuse this when
    asserting against the runtime set, mirroring the
    ``_registered_corpus_provider_types`` and
    ``_registered_degradation_kinds`` shapes used elsewhere in this
    module.
    """

    from pd_ocr_synth.text_transforms import default_registry

    return frozenset(default_registry().names())


def _check_text_transforms(recipe: Recipe) -> list[ValidationIssue]:
    """Surface recipe-named transforms the runtime registry doesn't know.

    The recipe model accepts any string for ``TextTransform.name`` (so
    YAML loads cleanly), but ``apply_pipeline`` will raise
    :class:`pd_ocr_synth.text_transforms.UnknownTransformError` deep in
    the render pipeline if the registry doesn't ship the named callable.
    Surfacing it at validate time mirrors the iter-65
    ``degradation_kind_not_implemented`` and iter-73
    ``corpus_provider_not_implemented`` precedents: tell the user up
    front before they kick off corpus fetch + render.

    Currently this catches the spec-05 transforms left deferred in the
    M04 roadmap: ``u_v_swap``, ``i_j_swap``, ``ct_st_ligature_marker``
    (antique-conventions), and the ``python:`` inline loader. See
    ``docs/roadmap/04-text-transforms.md`` "Antique-conventions
    built-ins" / "``python:`` inline loader" for status.

    Note this is a strict catalog check — third-party transforms
    registered through the ``pd_ocr_synth.text_transforms`` entry-point
    group participate in the default registry, so this check naturally
    accepts them too.
    """

    out: list[ValidationIssue] = []
    registered = _registered_text_transform_names()
    for i, transform in enumerate(recipe.text_transforms):
        if transform.name in registered:
            continue
        out.append(
            ValidationIssue(
                severity="error",
                code="text_transform_not_implemented",
                message=(
                    f"text transform '{transform.name}' is in the recipe schema but not "
                    f"yet implemented by the M04 runtime; render would raise. "
                    f"Implemented transforms: {', '.join(sorted(registered))}. "
                    "See docs/roadmap/04-text-transforms.md "
                    "(Antique-conventions built-ins / python: inline loader)."
                ),
                location=f"text_transforms[{i}].name",
            )
        )
    return out


def _check_layout(recipe: Recipe) -> list[ValidationIssue]:
    allowed = _LAYOUT_KEYS_BY_MODE.get(recipe.layout.mode, frozenset())
    # Enumerate set_keys directly from the Layout model so adding a new
    # field on Layout automatically participates in the unused-key check
    # without a parallel hand-written list to keep in sync. ``mode``
    # itself is not a "configuration key" — it's the discriminator the
    # permitted-keys table is keyed *by*.
    out: list[ValidationIssue] = []
    for key in Layout.model_fields:
        if key == "mode":
            continue
        value = getattr(recipe.layout, key)
        if value is None:
            continue
        if key not in allowed:
            out.append(
                ValidationIssue(
                    severity="warning",
                    code="layout_key_unused",
                    message=(
                        f"key '{key}' is set but layout.mode='{recipe.layout.mode}' "
                        f"does not use it; it will be ignored."
                    ),
                    location=f"layout.{key}",
                )
            )
    return out


def _check_output_layout_pairing(recipe: Recipe) -> list[ValidationIssue]:
    """Enforce the spec-08 pairing between ``output.mode`` and ``layout.mode``.

    Recognition mode is for tight per-word/per-line crops; detection
    mode is for full-page synthesis with bbox annotations. Mixing them
    yields output the trainer can't consume, so we block it here.
    """

    output_mode = recipe.output.mode
    layout_mode = recipe.layout.mode
    allowed = _LAYOUT_MODES_BY_OUTPUT_MODE.get(output_mode)
    if allowed is None:
        # Unknown output.mode would have been rejected by pydantic; the
        # safety net keeps mypy and future literal-additions honest.
        return []
    if layout_mode in allowed:
        return []
    return [
        ValidationIssue(
            severity="error",
            code="output_layout_mode_mismatch",
            message=(
                f"output.mode='{output_mode}' requires layout.mode in "
                f"{{{', '.join(sorted(allowed))}}}, got '{layout_mode}'. "
                "See docs/specs/08-output-format.md (Modes table)."
            ),
            location="layout.mode",
        )
    ]


def _registered_degradation_kinds() -> frozenset[str]:
    """Return the set of degradation kinds the runtime registry can dispatch.

    Triggers the lazy builtin registration so the registry reflects
    everything M06 ships with. Wrapped in a helper so tests can patch
    or reuse this when they need to assert against the runtime set.
    """

    from pd_ocr_synth.degradation.pipeline import REGISTRY, _ensure_builtins_registered

    _ensure_builtins_registered()
    return frozenset(REGISTRY)


def _check_degradation(recipe: Recipe) -> list[ValidationIssue]:
    out: list[ValidationIssue] = []
    registered_kinds = _registered_degradation_kinds()
    for i, stage in enumerate(recipe.degradation):
        if stage.kind not in KNOWN_DEGRADATION_KINDS:
            out.append(
                ValidationIssue(
                    severity="error",
                    code="degradation_kind_unknown",
                    message=(
                        f"unknown degradation kind '{stage.kind}'. Known kinds: "
                        f"{', '.join(sorted(KNOWN_DEGRADATION_KINDS))}"
                    ),
                    location=f"degradation[{i}].kind",
                )
            )
            continue
        # ``preset`` is a structural marker — the loader has already
        # expanded any preset entries by the time this validator runs,
        # so it never appears in ``recipe.degradation`` at runtime.
        # Nothing else to check; carry on.
        if stage.kind == "preset":
            continue
        if stage.kind not in registered_kinds:
            # Spec-known but not registered with the M06 runtime — the
            # render pass would raise ``DegradationError`` on first use.
            # Flag it as an error at validate time so the user discovers
            # the gap before kicking off a long render. See
            # docs/roadmap/06-degradation.md "Future work" for the list
            # of planned kinds.
            out.append(
                ValidationIssue(
                    severity="error",
                    code="degradation_kind_not_implemented",
                    message=(
                        f"degradation kind '{stage.kind}' is in the spec but not yet "
                        f"implemented by the M06 runtime; render would raise. "
                        f"Implemented kinds: {', '.join(sorted(registered_kinds))}. "
                        "See docs/roadmap/06-degradation.md (Future work)."
                    ),
                    location=f"degradation[{i}].kind",
                )
            )
            continue
        if stage.kind == "paper_texture":
            directory = (stage.model_extra or {}).get("directory")
            if directory is None:
                out.append(
                    ValidationIssue(
                        severity="error",
                        code="paper_texture_missing_directory",
                        message="paper_texture stage requires a 'directory' key",
                        location=f"degradation[{i}]",
                    )
                )
            else:
                p = Path(str(directory))
                if not p.exists():
                    out.append(
                        ValidationIssue(
                            severity="error",
                            code="paper_texture_directory_missing",
                            message=f"paper_texture directory does not exist: {p}",
                            location=f"degradation[{i}].directory",
                        )
                    )
                elif not p.is_dir():
                    out.append(
                        ValidationIssue(
                            severity="error",
                            code="paper_texture_directory_not_dir",
                            message=f"paper_texture directory is not a directory: {p}",
                            location=f"degradation[{i}].directory",
                        )
                    )
    return out


def _check_publish(recipe: Recipe) -> list[ValidationIssue]:
    if recipe.publish is None or recipe.publish.hf_dataset is None:
        return []
    hf = recipe.publish.hf_dataset
    out: list[ValidationIssue] = []
    if hf.description_file is not None and not hf.description_file.exists():
        out.append(
            ValidationIssue(
                severity="warning",
                code="publish_description_file_missing",
                message=(
                    f"publish.hf_dataset.description_file does not exist: {hf.description_file}. "
                    "It will be skipped at publish time."
                ),
                location="publish.hf_dataset.description_file",
            )
        )
    if "/" not in hf.repo or hf.repo.startswith("CHANGE-ME/"):
        out.append(
            ValidationIssue(
                severity="warning",
                code="publish_repo_placeholder",
                message=(
                    f"publish.hf_dataset.repo looks like a placeholder ({hf.repo}). "
                    "Edit it before running publish, or override with --repo."
                ),
                location="publish.hf_dataset.repo",
            )
        )
    return out


def _writable_destination(dest: Path) -> bool:
    """Walk ``dest`` upward until an existing ancestor is found.

    Writable iff the deepest existing ancestor is writable. We do not
    create the directory here — render does that with ``--force``
    semantics.
    """

    p = dest if dest.is_absolute() else Path.cwd() / dest
    current = p
    while True:
        if current.exists():
            return os.access(current, os.W_OK)
        parent = current.parent
        if parent == current:
            return False
        current = parent
