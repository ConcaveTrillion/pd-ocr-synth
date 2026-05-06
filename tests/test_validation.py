"""Tests for ``pd_ocr_synth.validation``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.recipe import load_recipe
from pd_ocr_synth.validation import (
    KNOWN_DEGRADATION_KINDS,
    ValidationReport,
    validate_recipe,
)


# Reused minimal recipe — kept fully in-memory so each test can mutate
# only what it needs and fix up paths against tmp_path.
def _minimal_yaml(*, font: str, dest: str, corpus: str) -> str:
    return f"""
schema_version: 1
name: minimal
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {dest}
  count: 100
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {font}
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color:
    r: 10
    g: 10
    b: 10
  background_color:
    r: 240
    g: 240
    b: 240
layout:
  mode: word_crops
  padding_px: 8
"""


@pytest.fixture
def good_recipe(tmp_path: Path, writable_font_bytes: bytes):
    """Build an in-memory recipe whose paths all exist + dest is writable."""
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hello world\n", encoding="utf-8")
    dest = tmp_path / "out"
    yaml_text = _minimal_yaml(font=str(font), dest=str(dest), corpus=str(corpus))
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml_text, encoding="utf-8")
    return load_recipe(recipe_path)


def test_minimal_recipe_validates_clean(good_recipe) -> None:
    report = validate_recipe(good_recipe)
    assert isinstance(report, ValidationReport)
    assert report.is_ok, [i.format() for i in report.issues]
    assert report.errors == ()


def test_missing_font_is_error(tmp_path: Path) -> None:
    yaml_text = _minimal_yaml(
        font=str(tmp_path / "ghost.otf"),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "font_missing" in codes


def test_missing_optional_font_is_warning(tmp_path: Path) -> None:
    seed = _make_file(tmp_path / "seed.txt")
    yaml_text = _minimal_yaml(
        font=str(tmp_path / "ghost.otf"),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    # Mark the font optional.
    yaml_text = yaml_text.replace(
        f"  - path: {tmp_path}/ghost.otf",
        f"  - path: {tmp_path}/ghost.otf\n    optional: true",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    assert report.errors == ()
    codes = [i.code for i in report.warnings]
    assert "optional_font_missing" in codes


def test_missing_local_corpus_is_error(tmp_path: Path) -> None:
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(tmp_path / "no-seed.txt"),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "local_corpus_missing" in codes


def test_unimplemented_corpus_provider_is_error(tmp_path: Path) -> None:
    """Spec-known but not-yet-registered corpus types must error at validate time.

    ``hf_dataset`` is in ``recipe.models.CorpusEntry`` (so the YAML
    loads cleanly through pydantic) but the M03 runtime registry does
    not register it yet — see ``docs/roadmap/03-corpus.md``
    "Built-in providers". Calling render on a recipe that uses it would
    raise ``ProviderError(f"unknown corpus provider 'hf_dataset' …")``
    deep in :func:`pd_ocr_synth.corpus.runner.run_providers`, well
    after corpus-fetch + setup costs for any preceding entries.

    Mirrors the iter-65 ``degradation_kind_not_implemented`` precedent:
    surface the gap up front, with a distinct error code separate from
    the truly-unknown-type path that pydantic itself rejects.
    """

    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    # Inject a second corpus entry of the unimplemented type *before*
    # ``fonts:``. Keep the local one so the test exercises the
    # registered/unregistered interleaving the validator must handle
    # correctly. ``_minimal_yaml`` keeps a stable layout, so the
    # ``fonts:\n`` anchor is safe.
    extra_entry = (
        "  - type: hf_dataset\n    name: example/irish-corpus\n    split: train\n    field: text\n"
    )
    yaml_text = yaml_text.replace("fonts:\n", extra_entry + "fonts:\n")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "corpus_provider_not_implemented" in codes, [i.format() for i in report.issues]
    msg = next(i.message for i in report.errors if i.code == "corpus_provider_not_implemented")
    # Error message points at the roadmap so the user knows where to
    # look for status / contribute, matching the
    # ``degradation_kind_not_implemented`` shape.
    assert "03-corpus.md" in msg
    assert "hf_dataset" in msg


def test_implemented_corpus_providers_pass_clean(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """A recipe using only registered providers must validate clean.

    Companion to ``test_unimplemented_corpus_provider_is_error``: pin
    the negative-space invariant that the new
    ``corpus_provider_not_implemented`` check does *not* fire for the
    M03 builtins. If a future refactor accidentally drops ``local`` or
    ``web`` from the default registry, this test surfaces it
    immediately rather than as a downstream regression in render.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "corpus_provider_not_implemented" not in codes, [i.format() for i in report.issues]


def test_unresolved_env_var_in_destination_is_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest="${DEFINITELY_UNSET_VAR}/out",
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "output_destination_unresolved" in codes


def test_unwritable_destination_is_error(tmp_path: Path) -> None:
    # /proc/1 has no writable ancestor for our user.
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest="/proc/1/no-write/here",
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "output_destination_unwritable" in codes


def test_unknown_degradation_kind_is_error(tmp_path: Path) -> None:
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += "degradation:\n  - kind: not_a_real_stage\n    probability: 0.5\n"
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "degradation_kind_unknown" in codes


@pytest.mark.parametrize(
    "kind",
    ["perspective", "scale", "bleed_through", "scratches", "fold_line", "binarize"],
)
def test_unimplemented_degradation_kind_is_error(tmp_path: Path, kind: str) -> None:
    """Spec-known but not-yet-registered kinds must error at validate time.

    These kinds appear in ``docs/specs/07-degradation.md`` but the M06
    runtime registry does not implement them yet (see
    ``docs/roadmap/06-degradation.md`` "Future work"). Calling render
    on a recipe that references one would raise
    ``DegradationError(f"unknown degradation kind {kind!r}")`` deep in
    the pipeline, well after corpus fetch + setup costs. Surface the
    gap at validate time so the user discovers it up front, with a
    distinct error code separate from the truly-unknown kind path.
    """

    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += f"degradation:\n  - kind: {kind}\n    probability: 0.5\n"
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    # Must use the dedicated code, not the generic unknown-kind code.
    assert "degradation_kind_not_implemented" in codes, [i.format() for i in report.issues]
    assert "degradation_kind_unknown" not in codes
    # Error message points at the roadmap so the user knows where to
    # look for status / contribute.
    msg = next(i.message for i in report.errors if i.code == "degradation_kind_not_implemented")
    assert "06-degradation.md" in msg


def test_paper_texture_missing_directory_key_is_error(tmp_path: Path) -> None:
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += "degradation:\n  - kind: paper_texture\n    probability: 0.5\n"
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "paper_texture_missing_directory" in codes


def test_paper_texture_directory_missing_is_error(tmp_path: Path) -> None:
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += (
        "degradation:\n"
        "  - kind: paper_texture\n"
        "    probability: 0.5\n"
        f"    directory: {tmp_path / 'no-textures'}\n"
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "paper_texture_directory_missing" in codes


def test_layout_mode_warns_on_unused_keys(tmp_path: Path) -> None:
    # word_crops + line_spacing → key is set but mode does not use it.
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        "layout:\n  mode: word_crops\n  padding_px: 8\n  line_spacing: 1.2\n",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.warnings]
    assert "layout_key_unused" in codes


@pytest.mark.parametrize("layout_mode", ["word_crops", "lines", "paragraphs"])
def test_paragraph_spacing_warns_on_non_pages_modes(
    tmp_path: Path,
    writable_font_bytes: bytes,
    layout_mode: str,
) -> None:
    """``paragraph_spacing`` is only meaningful for ``pages`` mode.

    Setting it on ``word_crops`` / ``lines`` / ``paragraphs`` (a single-
    paragraph sample) emits a ``layout_key_unused`` warning so the user
    knows the value will be ignored.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    # Build a mode-appropriate layout block plus the paragraph_spacing key.
    if layout_mode == "word_crops":
        new_layout = "layout:\n  mode: word_crops\n  padding_px: 8\n  paragraph_spacing: 1.4\n"
    else:
        new_layout = (
            f"layout:\n"
            f"  mode: {layout_mode}\n"
            f"  padding_px: 8\n"
            f"  max_width_px: 800\n"
            f"  paragraph_spacing: 1.4\n"
        )
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        new_layout,
    )
    # paragraphs/pages need detection output mode for the pairing check.
    if layout_mode in {"paragraphs", "pages"}:
        yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    warning_codes_at_paragraph_spacing = [
        i.code for i in report.warnings if i.location == "layout.paragraph_spacing"
    ]
    assert "layout_key_unused" in warning_codes_at_paragraph_spacing, [
        i.format() for i in report.issues
    ]


def test_paragraph_spacing_accepted_on_pages_mode(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """``paragraph_spacing`` is permitted on ``pages`` mode without warning."""
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        (
            "layout:\n"
            "  mode: pages\n"
            "  padding_px: 8\n"
            "  max_width_px: 800\n"
            "  paragraph_spacing: { min: 1.2, max: 1.8 }\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    # No layout_key_unused warning for paragraph_spacing on pages mode.
    paragraph_spacing_warnings = [
        i for i in report.warnings if i.location == "layout.paragraph_spacing"
    ]
    assert paragraph_spacing_warnings == [], [i.format() for i in paragraph_spacing_warnings]
    assert report.is_ok, [i.format() for i in report.issues]


@pytest.mark.parametrize("layout_mode", ["word_crops", "lines", "paragraphs"])
def test_paragraph_indent_px_warns_on_non_pages_modes(
    tmp_path: Path,
    writable_font_bytes: bytes,
    layout_mode: str,
) -> None:
    """``paragraph_indent_px`` is only meaningful for ``pages`` mode.

    Setting it on ``word_crops`` / ``lines`` / ``paragraphs`` (a single-
    paragraph sample, where indent would just add to the leading
    padding) emits a ``layout_key_unused`` warning so the user knows
    the value will be ignored.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    if layout_mode == "word_crops":
        new_layout = "layout:\n  mode: word_crops\n  padding_px: 8\n  paragraph_indent_px: 40\n"
    else:
        new_layout = (
            f"layout:\n"
            f"  mode: {layout_mode}\n"
            f"  padding_px: 8\n"
            f"  max_width_px: 800\n"
            f"  paragraph_indent_px: 40\n"
        )
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        new_layout,
    )
    if layout_mode in {"paragraphs", "pages"}:
        yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    warning_codes_at_indent = [
        i.code for i in report.warnings if i.location == "layout.paragraph_indent_px"
    ]
    assert "layout_key_unused" in warning_codes_at_indent, [i.format() for i in report.issues]


def test_paragraph_indent_px_accepted_on_pages_mode(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """``paragraph_indent_px`` is permitted on ``pages`` mode without warning."""
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        (
            "layout:\n"
            "  mode: pages\n"
            "  padding_px: 8\n"
            "  max_width_px: 800\n"
            "  paragraph_indent_px: 50\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    indent_warnings = [i for i in report.warnings if i.location == "layout.paragraph_indent_px"]
    assert indent_warnings == [], [i.format() for i in indent_warnings]
    assert report.is_ok, [i.format() for i in report.issues]


def test_known_degradation_set_includes_canonical_kinds() -> None:
    # Spot-check the catalog matches docs/specs/07-degradation.md.
    for k in ("skew", "blur", "paper_texture", "jpeg", "noise", "ink_bleed"):
        assert k in KNOWN_DEGRADATION_KINDS


# ---------------------------------------------------------------------------
# Spec ↔ code drift guard for degradation kinds.
#
# ``KNOWN_DEGRADATION_KINDS`` is the catalog the validator dispatches
# against — anything not in this set surfaces ``degradation_kind_unknown``
# (a *typo* error). Anything in this set but not yet registered with the
# M06 runtime surfaces ``degradation_kind_not_implemented`` (a clean
# "future work" gate, see iter 65 fix).
#
# That dispatch only works while the catalog is **kept in sync with**
# ``docs/specs/07-degradation.md``. If a spec PR adds a new kind but
# forgets ``KNOWN_DEGRADATION_KINDS``, recipes using the spec'd kind
# fall through to ``degradation_kind_unknown`` and the user gets a
# misleading "did you typo this?" error instead of a clear "not yet
# implemented" message. Conversely, if a kind is removed from the spec
# but the catalog still lists it, ``validate`` happily accepts a kind
# the docs no longer describe.
#
# This pair of meta-tests parses the spec doc and asserts the two sides
# match. They have to be kept in lockstep — by design — so any drift
# caught here is a real bug in either the doc or the catalog.
# ---------------------------------------------------------------------------


def _spec_known_degradation_kinds() -> frozenset[str]:
    """Extract the canonical degradation-kind set from the spec doc.

    Parses h3 headers in ``docs/specs/07-degradation.md`` of the form
    ``### \\`name\\``` or ``### \\`a\\` / \\`b\\``` (the ``brightness``
    / ``contrast`` pair). Anything outside the kind catalog is gated
    by section header so unrelated h3s in future revisions of the doc
    don't accidentally enter the set.
    """

    import re

    spec_path = Path(__file__).resolve().parent.parent / "docs" / "specs" / "07-degradation.md"
    text = spec_path.read_text(encoding="utf-8")
    # Catalog sections, in spec order. "Composition presets" and
    # "Custom degradation stages" are not kind catalogs and are
    # intentionally excluded.
    catalog_sections = (
        "## Geometric",
        "## Optical",
        "## Print / paper",
        "## Compression",
        "## Color space",
    )
    kinds: set[str] = set()
    in_catalog = False
    h3_re = re.compile(r"^###\s+(.+)$")
    backtick_re = re.compile(r"`([a-z_][a-z0-9_]*)`")
    for line in text.splitlines():
        if line.startswith("## "):
            in_catalog = line.strip() in catalog_sections
            continue
        if not in_catalog:
            continue
        m = h3_re.match(line)
        if not m:
            continue
        for name in backtick_re.findall(m.group(1)):
            kinds.add(name)
    return frozenset(kinds)


def test_known_degradation_kinds_matches_spec_doc() -> None:
    """``KNOWN_DEGRADATION_KINDS`` must mirror the spec catalog 1:1.

    ``preset`` is the structural marker the loader expands away before
    the validator runs; it is intentionally accepted by the catalog
    even though the spec lists it under "Composition presets" rather
    than as a kind h3. Strip it out for the comparison.
    """

    spec_kinds = _spec_known_degradation_kinds()
    catalog_kinds = KNOWN_DEGRADATION_KINDS - {"preset"}

    missing_from_catalog = spec_kinds - catalog_kinds
    extra_in_catalog = catalog_kinds - spec_kinds
    assert not missing_from_catalog, (
        f"docs/specs/07-degradation.md lists kinds not in "
        f"KNOWN_DEGRADATION_KINDS: {sorted(missing_from_catalog)}. "
        "Update src/pd_ocr_synth/validation.py:KNOWN_DEGRADATION_KINDS."
    )
    assert not extra_in_catalog, (
        f"KNOWN_DEGRADATION_KINDS lists kinds not in "
        f"docs/specs/07-degradation.md: {sorted(extra_in_catalog)}. "
        "Either add the kind to the spec doc or drop it from the catalog."
    )


def test_registered_degradation_kinds_subset_of_spec() -> None:
    """The M06 runtime registry must only register spec-listed kinds.

    A registered kind absent from the spec catalog would render fine
    but never be reachable from a validated recipe (the validator would
    reject it as ``degradation_kind_unknown``). That's a silent gap
    between what the runtime can do and what users can ask for —
    catch it here.

    The reverse direction (catalog kinds *not* registered) is
    intentional and covered by ``degradation_kind_not_implemented`` at
    validate time, so it's not asserted here.
    """

    from pd_ocr_synth.degradation.pipeline import (
        REGISTRY,
        _ensure_builtins_registered,
    )

    _ensure_builtins_registered()
    registered = frozenset(REGISTRY)
    spec_kinds = _spec_known_degradation_kinds()
    extra = registered - spec_kinds
    assert not extra, (
        f"degradation registry registers kinds not listed in "
        f"docs/specs/07-degradation.md: {sorted(extra)}. "
        "Either add the kind to the spec doc or remove the "
        "register_*_stage call."
    )


# ---------------------------------------------------------------------------
# paragraph_alignment (M09 paragraph alignment)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_mode", ["paragraphs", "pages"])
@pytest.mark.parametrize("alignment", ["left", "center", "right", "justify"])
def test_paragraph_alignment_accepted_on_paragraph_modes(
    tmp_path: Path,
    writable_font_bytes: bytes,
    layout_mode: str,
    alignment: str,
) -> None:
    """``paragraph_alignment`` is permitted on ``paragraphs`` + ``pages`` without warning.

    All four implemented values (``left`` / ``center`` / ``right`` /
    ``justify``) must round-trip through the validator cleanly on
    both paragraph-style modes.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        (
            f"layout:\n"
            f"  mode: {layout_mode}\n"
            f"  padding_px: 8\n"
            f"  max_width_px: 800\n"
            f"  paragraph_alignment: {alignment}\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    align_warnings = [i for i in report.warnings if i.location == "layout.paragraph_alignment"]
    assert align_warnings == [], [i.format() for i in align_warnings]
    assert report.is_ok, [i.format() for i in report.issues]


@pytest.mark.parametrize("layout_mode", ["word_crops", "lines"])
def test_paragraph_alignment_warns_on_recognition_modes(
    tmp_path: Path,
    writable_font_bytes: bytes,
    layout_mode: str,
) -> None:
    """``paragraph_alignment`` is not meaningful for recognition-mode layouts.

    Setting it on ``word_crops`` / ``lines`` emits ``layout_key_unused``
    so the user knows the value will be ignored.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    if layout_mode == "word_crops":
        new_layout = "layout:\n  mode: word_crops\n  padding_px: 8\n  paragraph_alignment: center\n"
    else:
        new_layout = (
            f"layout:\n"
            f"  mode: {layout_mode}\n"
            f"  padding_px: 8\n"
            f"  max_width_px: 800\n"
            f"  paragraph_alignment: center\n"
        )
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        new_layout,
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    align_warning_codes = [
        i.code for i in report.warnings if i.location == "layout.paragraph_alignment"
    ]
    assert "layout_key_unused" in align_warning_codes, [i.format() for i in report.issues]


def test_paragraph_alignment_unknown_value_rejected_at_load(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """Pydantic's Literal rejects unknown alignment values at load time.

    ``"justify"`` joined the supported set in iter 41 — pick a string
    outside the four-value vocabulary to verify the gate is still
    enforced.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        (
            "layout:\n"
            "  mode: paragraphs\n"
            "  padding_px: 8\n"
            "  max_width_px: 800\n"
            "  paragraph_alignment: kerning\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    with pytest.raises(Exception, match="paragraph_alignment"):
        load_recipe(_write(tmp_path, yaml_text))


# ---------------------------------------------------------------------------
# page_size_px (M09 explicit fixed-canvas)
# ---------------------------------------------------------------------------


def test_page_size_px_accepted_on_pages_mode(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """``page_size_px`` is permitted on ``pages`` mode without warning."""
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        (
            "layout:\n"
            "  mode: pages\n"
            "  padding_px: 8\n"
            "  max_width_px: 800\n"
            "  page_size_px: [1200, 1800]\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    page_size_warnings = [i for i in report.warnings if i.location == "layout.page_size_px"]
    assert page_size_warnings == [], [i.format() for i in page_size_warnings]
    assert report.is_ok, [i.format() for i in report.issues]


@pytest.mark.parametrize("layout_mode", ["word_crops", "lines", "paragraphs"])
def test_page_size_px_warns_on_non_pages_modes(
    tmp_path: Path,
    writable_font_bytes: bytes,
    layout_mode: str,
) -> None:
    """``page_size_px`` is only meaningful for ``pages`` mode.

    A ``paragraphs`` sample is a tight single-paragraph crop with no
    notion of a "page"; ``word_crops`` / ``lines`` are likewise tight
    crops. Setting it on those modes emits a ``layout_key_unused``
    warning so the user knows the value will be ignored.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    if layout_mode == "word_crops":
        new_layout = "layout:\n  mode: word_crops\n  padding_px: 8\n  page_size_px: [1200, 1800]\n"
    else:
        new_layout = (
            f"layout:\n"
            f"  mode: {layout_mode}\n"
            f"  padding_px: 8\n"
            f"  max_width_px: 800\n"
            f"  page_size_px: [1200, 1800]\n"
        )
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        new_layout,
    )
    if layout_mode in {"paragraphs", "pages"}:
        yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    warning_codes_at_size = [i.code for i in report.warnings if i.location == "layout.page_size_px"]
    assert "layout_key_unused" in warning_codes_at_size, [i.format() for i in report.issues]


@pytest.mark.parametrize("bad", [(0, 100), (100, 0), (-5, 100), (100, -5)])
def test_page_size_px_rejects_non_positive_at_load(
    tmp_path: Path, writable_font_bytes: bytes, bad: tuple[int, int]
) -> None:
    """Pydantic's ``page_size_px`` validator rejects zero / negative dimensions."""
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        (
            "layout:\n"
            "  mode: pages\n"
            "  padding_px: 8\n"
            "  max_width_px: 800\n"
            f"  page_size_px: [{bad[0]}, {bad[1]}]\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    with pytest.raises(Exception, match="page_size_px"):
        load_recipe(_write(tmp_path, yaml_text))


# ---------------------------------------------------------------------------
# output.mode / layout.mode pairing (spec 08, §Modes)
# ---------------------------------------------------------------------------


def _swap_output_layout_modes(yaml_text: str, *, output_mode: str, layout_mode: str) -> str:
    """Swap the recognition/word_crops defaults emitted by ``_minimal_yaml``."""
    swapped = yaml_text.replace("mode: recognition", f"mode: {output_mode}")
    return swapped.replace("mode: word_crops", f"mode: {layout_mode}")


def _layout_block_for(layout_mode: str) -> str:
    """Render a minimal but mode-appropriate layout block.

    ``_minimal_yaml`` already includes ``padding_px: 8`` for word_crops;
    other modes add ``max_width_px`` so the keys-by-mode warning logic
    doesn't fire and obscure the pairing assertion under test.
    """
    if layout_mode == "word_crops":
        return "layout:\n  mode: word_crops\n  padding_px: 8\n"
    return f"layout:\n  mode: {layout_mode}\n  padding_px: 8\n  max_width_px: 800\n"


@pytest.mark.parametrize(
    ("output_mode", "layout_mode"),
    [
        ("recognition", "paragraphs"),
        ("recognition", "pages"),
        ("detection", "word_crops"),
        ("detection", "lines"),
    ],
)
def test_output_layout_mode_mismatch_is_error(
    tmp_path: Path,
    writable_font_bytes: bytes,
    output_mode: str,
    layout_mode: str,
) -> None:
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    # Replace the layout block wholesale so we can pick mode-appropriate keys.
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        _layout_block_for(layout_mode),
    )
    yaml_text = yaml_text.replace("mode: recognition", f"mode: {output_mode}")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "output_layout_mode_mismatch" in codes, [i.format() for i in report.issues]


@pytest.mark.parametrize(
    ("output_mode", "layout_mode"),
    [
        ("recognition", "word_crops"),
        ("recognition", "lines"),
        ("detection", "paragraphs"),
        ("detection", "pages"),
    ],
)
def test_output_layout_mode_pairing_valid_combinations_pass(
    tmp_path: Path,
    writable_font_bytes: bytes,
    output_mode: str,
    layout_mode: str,
) -> None:
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        _layout_block_for(layout_mode),
    )
    yaml_text = yaml_text.replace("mode: recognition", f"mode: {output_mode}")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "output_layout_mode_mismatch" not in codes, [i.format() for i in report.issues]


def test_output_layout_mode_mismatch_message_cites_spec(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _swap_output_layout_modes(
        _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed)),
        output_mode="detection",
        layout_mode="word_crops",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    msg = next(i.message for i in report.errors if i.code == "output_layout_mode_mismatch")
    assert "08-output-format.md" in msg
    # The message should hint at which layout modes are valid for detection.
    assert "paragraphs" in msg and "pages" in msg


# ---------------------------------------------------------------------------
# Layout-model ↔ permitted-keys drift guard.
#
# ``_LAYOUT_KEYS_BY_MODE`` lists, per layout mode, which keys are
# *meaningful* in that mode. Anything set but not listed surfaces a
# ``layout_key_unused`` warning. That dispatch only works while the
# table is **kept in sync with** the ``Layout`` model in
# ``recipe/models.py``:
#
# * If a new Layout field is added but never enters
#   ``_LAYOUT_KEYS_BY_MODE``, every recipe that sets it gets a
#   misleading "unused" warning regardless of mode (false positive),
#   *or* the field is silently dead (no mode reads it).
# * If ``_LAYOUT_KEYS_BY_MODE`` lists a key that isn't actually a
#   Layout field, the validator advertises a config knob the model
#   doesn't accept — ``extra="forbid"`` rejects it at load time, so the
#   permitted-keys entry is a false promise.
#
# These two meta-tests catch both directions. They have to be kept in
# lockstep — by design — so any drift caught here is a real bug.
# ---------------------------------------------------------------------------


def _layout_field_names() -> frozenset[str]:
    """Pydantic model fields on ``Layout``, including ``mode``."""

    from pd_ocr_synth.recipe.models import Layout

    return frozenset(Layout.model_fields.keys())


def _permitted_layout_keys() -> frozenset[str]:
    """Union of keys permitted by any layout mode."""

    from pd_ocr_synth.validation import _LAYOUT_KEYS_BY_MODE

    out: set[str] = set()
    for keys in _LAYOUT_KEYS_BY_MODE.values():
        out.update(keys)
    return frozenset(out)


def test_permitted_layout_keys_are_all_layout_fields() -> None:
    """Every key in ``_LAYOUT_KEYS_BY_MODE`` must be an actual ``Layout`` field.

    A permitted key that isn't a model field is a false promise — pydantic
    rejects the field at load time (``extra='forbid'``), so the validator
    table advertises a knob users can never set.
    """

    permitted = _permitted_layout_keys()
    fields = _layout_field_names()
    extra = permitted - fields
    assert not extra, (
        f"_LAYOUT_KEYS_BY_MODE lists keys not on the Layout model: "
        f"{sorted(extra)}. Either add the field to "
        "src/pd_ocr_synth/recipe/models.py:Layout or drop the key from "
        "_LAYOUT_KEYS_BY_MODE in src/pd_ocr_synth/validation.py."
    )


def test_every_layout_field_appears_in_some_mode() -> None:
    """Every ``Layout`` field (except ``mode``) must be permitted in at least one mode.

    A Layout field absent from every permitted set means either:
    * the validator emits ``layout_key_unused`` whenever the user sets
      it regardless of mode (false positive), or
    * the renderer silently ignores it (dead field).

    ``mode`` is excluded — it's the discriminator the table is keyed
    *by*, not a per-mode configuration key.
    """

    fields = _layout_field_names() - {"mode"}
    permitted = _permitted_layout_keys()
    missing = fields - permitted
    assert not missing, (
        f"Layout fields not permitted in any mode by _LAYOUT_KEYS_BY_MODE: "
        f"{sorted(missing)}. Add each field to the appropriate mode(s) in "
        "src/pd_ocr_synth/validation.py:_LAYOUT_KEYS_BY_MODE, or drop it "
        "from src/pd_ocr_synth/recipe/models.py:Layout if it's truly unused."
    )


def test_permitted_layout_keys_table_modes_match_layout_mode_literal() -> None:
    """``_LAYOUT_KEYS_BY_MODE`` must cover every value of ``Layout.mode``.

    A mode missing from the table falls through to ``frozenset()`` in
    ``_check_layout`` — every key the user sets would then warn as
    unused, regardless of legitimacy. A mode in the table that isn't a
    legal ``Layout.mode`` literal is dead config (pydantic would reject
    it at load time, so the validator never sees it).
    """

    import typing

    from pd_ocr_synth.recipe.models import Layout
    from pd_ocr_synth.validation import _LAYOUT_KEYS_BY_MODE

    mode_field = Layout.model_fields["mode"]
    literal_modes = frozenset(typing.get_args(mode_field.annotation))
    table_modes = frozenset(_LAYOUT_KEYS_BY_MODE.keys())
    missing = literal_modes - table_modes
    extra = table_modes - literal_modes
    assert not missing, (
        f"_LAYOUT_KEYS_BY_MODE is missing layout modes: {sorted(missing)}. "
        "Every value of Layout.mode must have a permitted-keys entry."
    )
    assert not extra, (
        f"_LAYOUT_KEYS_BY_MODE lists modes not in Layout.mode: "
        f"{sorted(extra)}. Drop them from "
        "src/pd_ocr_synth/validation.py:_LAYOUT_KEYS_BY_MODE."
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_file(p: Path, content: str = "") -> Path:
    p.write_text(content, encoding="utf-8")
    return p


def _write(dirpath: Path, yaml_text: str, name: str = "recipe.yaml") -> Path:
    rp = dirpath / name
    rp.write_text(yaml_text, encoding="utf-8")
    return rp
