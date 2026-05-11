"""End-to-end render orchestration for recognition + detection (M07/M09).

``run_recipe`` ties M03-M06 together into the trainer-consumable
profile layout described in :mod:`pd_ocr_synth.output`:

1. Run corpus providers + recipe-level text transforms.
2. Tokenize for the recipe's layout mode.
3. Pick ``count`` deterministic samples.
4. Render + degrade each sample, streaming results into the writer
   that matches ``output.mode`` (recognition or detection).
5. Honor ``--force`` / ``--resume`` semantics, including snapshot
   integrity checks.

Layouts wired in:

- ``recognition`` + ``word_crops`` (M07) — one word per sample.
- ``recognition`` + ``lines`` (M09) — one line per sample, with
  per-word bboxes carried through the manifest.
- ``detection`` + ``paragraphs`` (M09) — multi-line paragraph per
  sample, with per-line + per-word bboxes carried into ``labels.json``.
- ``detection`` + ``pages`` (M09) — multi-paragraph page per sample,
  with per-paragraph + per-line + per-word bboxes; per-line GT lands
  in ``labels.json`` (the writer flattens lines across paragraphs in
  reading order).

Paragraph input shape: tokenization yields one paragraph per sample
(blank-line-separated). ``render_paragraph`` requires a pre-fitted
``list[str]`` of lines, which we produce in one of two ways:

- ``layout.max_width_px`` is set → the wrap-fitter
  (:func:`pd_ocr_synth.render.wrap.fit_lines`) shapes each candidate
  line against the paragraph's pre-sampled font + pixel size and
  greedy-packs words into lines that fit. Hard newlines in the
  corpus token are preserved as line breaks. To keep the wrap budget
  aligned with what the renderer paints, we pre-sample a
  :class:`pd_ocr_synth.render.paragraph.ParagraphStyle` and thread
  it through both ``fit_lines`` and ``render_paragraph``.
- ``layout.max_width_px`` is not set → split on embedded newlines
  only (after a ``str.splitlines`` + empty-line filter). A
  paragraph with no embedded newlines becomes a single-line
  paragraph, still a legal detection-mode sample because
  ``DetectionWriter`` only requires per-line + per-word ground truth,
  not multiple lines.

Page input shape: tokenization for ``pages`` mode yields one
**page-sized** chunk per sample, where paragraphs inside a page are
separated by a blank line (a run of 2+ newlines) and pages are
separated by a **triple**-blank-line boundary (a run of 3+ newlines).
The pages splitter inside ``_render_sample`` re-splits the page
token on the paragraph boundary and then runs the same per-paragraph
wrap path as ``paragraphs`` mode (each paragraph wraps independently
against ``layout.max_width_px``). The page-level ``PageStyle`` is
pre-sampled once so wrap-fitting measures against the same font +
pixel size the renderer paints with — same seam as paragraphs mode,
lifted to the page level.

Determinism is the same as :mod:`pd_ocr_synth.render.preview`: the
per-token pick is keyed on ``seed ^ 0xC0FFEE``; per-sample render +
degradation RNGs are reseeded by sample index. Worker count is not
load-bearing on output bytes.
"""

from __future__ import annotations

import multiprocessing
import random
import re
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pd_ocr_synth.audit import (
    AUDIT_FILENAME,
    AuditEntry,
    append_audit_entry,
    compute_recipe_sha,
    default_global_audit_path,
    now_timestamp,
    should_emit_audit,
    should_emit_global_audit,
)
from pd_ocr_synth.corpus import CacheStore, ProviderContext, default_cache_root
from pd_ocr_synth.corpus.runner import collect_corpus_text
from pd_ocr_synth.degradation import DegradationError, apply_degradation
from pd_ocr_synth.output import DetectionWriter, RecognitionWriter
from pd_ocr_synth.render.context import RenderContext
from pd_ocr_synth.render.line import render_line
from pd_ocr_synth.render.page import (
    PageStyle,
    render_page,
    sample_page_style,
)
from pd_ocr_synth.render.paragraph import (
    ParagraphStyle,
    render_paragraph,
    sample_paragraph_style,
)
from pd_ocr_synth.render.word_crop import (
    MissingGlyphError,
    RenderError,
    render_word_crop,
)
from pd_ocr_synth.render.wrap import fit_lines
from pd_ocr_synth.tokenization import tokenize

# Layout modes wired through ``run_recipe``'s render dispatch. Each
# entry must also be paired with a compatible ``output.mode`` (per
# ``pd_ocr_synth.validation``) — ``word_crops`` / ``lines`` go with
# the recognition writer; ``paragraphs`` / ``pages`` go with the
# detection writer.
_SUPPORTED_LAYOUTS: frozenset[str] = frozenset({"word_crops", "lines", "paragraphs", "pages"})

if TYPE_CHECKING:
    from pd_ocr_synth.recipe import Recipe


# ---------------------------------------------------------------------------
# Config / result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunPlan:
    """What ``run_recipe`` would do, surfaced by ``--dry-run``.

    The ``describe`` and ``--dry-run`` paths print this without
    touching disk so an author can confirm the resolved config before
    a long render.
    """

    recipe_name: str
    output_dir: str
    count: int
    seed: int
    workers: int
    layout_mode: str
    fonts_present: int
    fonts_missing_optional: int
    transforms: list[str]
    degradation_stages: list[str]
    corpus_entries: int
    corpus_total_chars: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RunResult:
    """High-level summary returned by ``run_recipe``."""

    output_dir: str
    rendered: int
    skipped: int
    skip_reasons: dict[str, int]
    wall_time_seconds: float


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def plan_recipe(
    recipe: Recipe,
    *,
    output_dir: Path,
    count: int | None = None,
    seed: int | None = None,
    workers: int = 1,
    cache_dir: Path | None = None,
    no_cache: bool = False,
) -> RunPlan:
    """Compute the dry-run plan without writing.

    Touches the network only insofar as the corpus runner does (to
    determine total chars). Callers that want a fully offline plan
    should pre-fetch first via ``pd-ocr-synth fetch``.

    ``no_cache`` (default ``False``) forwards the ``--no-cache`` CLI
    flag through to ``collect_corpus_text``: when ``True`` the corpus
    runner bypasses the on-disk cache and re-fetches every cache-aware
    provider from upstream.
    """

    if recipe.layout.mode not in _SUPPORTED_LAYOUTS:
        raise RenderError(
            f"render does not yet support layout.mode={recipe.layout.mode!r}; "
            f"current dispatch accepts {sorted(_SUPPORTED_LAYOUTS)}."
        )

    effective_count = count if count is not None else recipe.output.count
    effective_seed = recipe.seed if seed is None else int(seed)

    fonts_present = sum(1 for f in recipe.fonts if f.path.exists())
    fonts_missing_optional = sum(1 for f in recipe.fonts if f.optional and not f.path.exists())

    # Compute corpus size for the plan. This pays the (cached) fetch
    # cost; it's the same work the real run does, so the dry-run
    # surfaces network failures early.
    cache_root = cache_dir or default_cache_root()
    if recipe.source_path is None:
        raise RenderError("recipe has no source_path; load it via load_recipe(path)")
    ctx = ProviderContext(recipe_dir=recipe.source_path.parent, cache=CacheStore(root=cache_root))
    text = collect_corpus_text(recipe, ctx=ctx, no_cache=no_cache)

    return RunPlan(
        recipe_name=recipe.name,
        output_dir=str(output_dir),
        count=effective_count,
        seed=effective_seed,
        workers=workers,
        layout_mode=recipe.layout.mode,
        fonts_present=fonts_present,
        fonts_missing_optional=fonts_missing_optional,
        transforms=[t.name for t in recipe.text_transforms],
        degradation_stages=[s.kind for s in recipe.degradation],
        corpus_entries=len(recipe.corpus),
        corpus_total_chars=len(text),
    )


def run_recipe(
    recipe: Recipe,
    *,
    output_dir: Path,
    count: int | None = None,
    seed: int | None = None,
    workers: int = 1,
    cache_dir: Path | None = None,
    force: bool = False,
    resume: bool = False,
    progress: bool = True,
    audit: bool = True,
    no_cache: bool = False,
) -> RunResult:
    """Render the full dataset for ``recipe`` into ``output_dir``.

    Parameters mirror the CLI flags. ``count`` overrides
    ``recipe.output.count`` for smoke tests; ``seed`` overrides the
    recipe seed; ``workers`` is the multiprocessing pool size (1 =
    in-process serial path); ``cache_dir`` overrides the corpus cache
    root. ``force`` clears the destination first; ``resume`` continues
    from any existing samples after validating the snapshot.

    ``audit`` (default ``True``) emits one JSONL line per run into
    ``<output_dir>/_audit.jsonl`` for traceability. Pass ``False`` to
    suppress (the CLI ``--no-audit`` flag wires through here). The
    env var ``PD_OCR_SYNTH_NO_AUDIT=1`` overrides the kwarg globally;
    this is checked inside ``should_emit_audit``.

    ``no_cache`` (default ``False``) forwards the ``--no-cache`` CLI
    flag through to ``collect_corpus_text``: when ``True`` the corpus
    runner bypasses the on-disk cache and re-fetches every cache-aware
    provider from upstream.

    Determinism contract: same recipe + same effective seed + same
    sample index → identical output bytes regardless of ``workers``.
    The audit log is *not* part of the determinism contract — it
    carries a wall-clock timestamp and lives in a separate file from
    the dataset payload.
    """

    if recipe.layout.mode not in _SUPPORTED_LAYOUTS:
        raise RenderError(
            f"render does not yet support layout.mode={recipe.layout.mode!r}; "
            f"current dispatch accepts {sorted(_SUPPORTED_LAYOUTS)}."
        )

    effective_count = count if count is not None else recipe.output.count
    effective_seed = recipe.seed if seed is None else int(seed)
    if effective_count <= 0:
        raise RenderError(f"render count must be positive (got {effective_count})")
    worker_count = max(1, int(workers))

    cache_root = cache_dir or default_cache_root()
    if recipe.source_path is None:
        raise RenderError("recipe has no source_path; load it via load_recipe(path)")
    ctx = ProviderContext(recipe_dir=recipe.source_path.parent, cache=CacheStore(root=cache_root))

    text = collect_corpus_text(recipe, ctx=ctx, no_cache=no_cache)
    tokens = tokenize(text, mode=recipe.layout.mode)
    if not tokens:
        raise RenderError("corpus produced no tokens after tokenization")

    pick_rng = random.Random(effective_seed ^ 0xC0FFEE)
    chosen: list[str] = [pick_rng.choice(tokens) for _ in range(effective_count)]

    started = time.monotonic()
    progress_reporter = _ProgressReporter(total=effective_count, enabled=progress)

    writer_cls = _writer_class_for(recipe)
    with writer_cls.open(
        recipe,
        output_dir,
        seed=effective_seed,
        force=force,
        resume=resume,
        planned_count=effective_count,
    ) as writer:
        # Resume short-circuit: skip indices already rendered. Skipped
        # indices fall through to retry — the input that caused the
        # skip might have been fixed.
        pending: list[tuple[int, str]] = [
            (i, token) for i, token in enumerate(chosen) if not writer.already_rendered(i)
        ]
        # Track unique tokens *attempted* (rendered + skipped) for the
        # stats summary.
        unique_tokens: set[str] = set()
        for _, tok in pending:
            unique_tokens.add(tok)
        # If resume, fold in tokens we already rendered too.
        for idx in range(effective_count):
            if writer.already_rendered(idx):
                unique_tokens.add(chosen[idx])

        if worker_count == 1:
            _drive_serial(
                pending=pending,
                recipe=recipe,
                writer=writer,
                seed=effective_seed,
                progress=progress_reporter,
            )
        else:
            _drive_parallel(
                pending=pending,
                recipe=recipe,
                writer=writer,
                recipe_path=recipe.source_path,
                seed=effective_seed,
                workers=worker_count,
                progress=progress_reporter,
            )

        writer.stats.tokens_unique = len(unique_tokens)
        writer.stats.wall_time_seconds = round(time.monotonic() - started, 3)
        # Capture the writer's view of the run *before* the writer
        # context manager tears down. Stats are mutable on the writer
        # but the values we need for the audit entry are settled by
        # this point (final bumps happened above).
        rendered_count = writer.stats.samples_written
        skipped_count = writer.stats.samples_skipped
        skip_reasons = dict(writer.stats.skip_reasons)
        wall_time = writer.stats.wall_time_seconds

    progress_reporter.done()

    if should_emit_audit(audit=audit):
        entry = AuditEntry(
            timestamp=now_timestamp(),
            recipe_name=recipe.name,
            recipe_sha=compute_recipe_sha(recipe.source_path),
            output_dir=str(output_dir.resolve()),
            count=effective_count,
            seed=effective_seed,
            workers=worker_count,
            rendered=rendered_count,
            skipped=skipped_count,
            runtime_seconds=wall_time,
        )
        # Best-effort: a write-failure on the audit sidecar must not
        # mask a successful render. Log the failure but return the
        # real result. (We don't have a logger surface here; stderr
        # is the project's existing convention for runner-level
        # diagnostics — see the progress reporter above.)
        try:
            append_audit_entry(output_dir / AUDIT_FILENAME, entry)
        except OSError as exc:  # pragma: no cover - exceptional path
            print(
                f"warning: audit log write failed ({exc}); render output unaffected",
                file=sys.stderr,
            )

        # Global aggregate mirror: append the same entry to the
        # cross-recipe timeline at ``<cache_root>/audit.jsonl``. Best-
        # effort like the per-output-dir write — a slow / read-only
        # cache root must not fail a successful render. Suppressed
        # when ``--no-audit`` / ``PD_OCR_SYNTH_NO_AUDIT`` is set, OR
        # when only the global mirror is suppressed via
        # ``PD_OCR_SYNTH_NO_GLOBAL_AUDIT`` (e.g. a test that doesn't
        # want to touch the user's cache dir; the test suite sets the
        # env var via the ``isolated_global_audit`` fixture in
        # ``conftest.py``).
        if should_emit_global_audit(audit=audit):
            try:
                append_audit_entry(default_global_audit_path(), entry)
            except OSError as exc:
                # Best-effort mirror: a failure here (read-only cache
                # root, disk quota, NFS hiccup) must not mask the
                # successful render or the per-output-dir audit. The
                # OSError branch is covered by
                # ``tests/test_audit_global.py::test_global_audit_mirror_oserror_*``
                # via fault injection; do not gate this with
                # ``# pragma: no cover``.
                print(
                    f"warning: global audit log write failed ({exc}); "
                    f"per-output audit and render output unaffected",
                    file=sys.stderr,
                )

    return RunResult(
        output_dir=str(output_dir),
        rendered=rendered_count,
        skipped=skipped_count,
        skip_reasons=skip_reasons,
        wall_time_seconds=wall_time,
    )


# ---------------------------------------------------------------------------
# Per-sample render core (shared between serial + parallel paths)
# ---------------------------------------------------------------------------


def _writer_class_for(recipe: Recipe) -> type:
    """Pick the writer class that matches ``recipe.output.mode``.

    Validation already enforces the output.mode/layout.mode pairing
    (see :mod:`pd_ocr_synth.validation`); this helper just routes a
    valid pair to the right concrete writer. Detection mode lights up
    in M09; recognition has been the M07 default.
    """

    mode = recipe.output.mode
    if mode == "recognition":
        return RecognitionWriter
    if mode == "detection":
        return DetectionWriter
    raise RenderError(f"unknown output.mode={mode!r}; expected recognition or detection")


def _split_paragraph_into_lines(
    token: str,
    *,
    recipe: Recipe,
    ctx: RenderContext,
    style: ParagraphStyle,
    first_line_indent_px: int = 0,
) -> list[str]:
    """Split a paragraph corpus token into the ``list[str]`` that
    ``render_paragraph`` expects.

    Per spec 06 ``layout.max_width_px`` (when set) drives the wrap
    budget. We fan out to :func:`fit_lines`, which shapes each
    candidate line through HarfBuzz against the **same** font + pixel
    size the renderer will paint with — that's why the call site pre-
    samples a :class:`ParagraphStyle` and threads it through here:
    sampling twice would consume RNG state and the wrap budget would
    drift away from the painted line.

    ``first_line_indent_px`` is forwarded to :func:`fit_lines` so the
    first line's wrap budget shrinks by the indent the renderer will
    apply — without this the painted first line + indent would
    overflow ``max_width_px``. Pages-mode dispatch passes the recipe's
    ``layout.paragraph_indent_px`` here on a per-paragraph basis;
    paragraphs-mode dispatch defaults to ``0`` (no indent there per
    the recipe validator's ``layout_key_unused`` warning).

    Hard line breaks in ``token`` (already-newline-separated lines)
    are preserved by ``fit_lines``: each chunk wraps independently and
    the results are concatenated. A token with no embedded newlines
    becomes one or more wrapped lines.

    When ``layout.max_width_px`` is **not** set the recipe author has
    opted out of wrap fitting — fall back to the previous ``\n``-only
    split. A single-line paragraph token then becomes a one-element
    list, still legal for :func:`render_paragraph`.
    """

    max_width_px = recipe.layout.max_width_px
    if max_width_px is not None and max_width_px > 0:
        handles = ctx.font_handles(style.font_path)
        # ``fit_lines`` measures via HarfBuzz. The freetype face's
        # pixel-size has already been set by ``render_paragraph`` /
        # the next render call, but ``fit_lines`` only reads
        # ``handles.hb_face`` so the freetype state doesn't matter
        # here.
        lines = fit_lines(
            token,
            max_width_px=max_width_px,
            handles=handles,
            pixel_size=style.pixel_size,
            features=style.font_features,
            first_line_indent_px=first_line_indent_px,
        )
        if lines:
            return lines
        # ``fit_lines`` returns ``[]`` for empty / whitespace-only
        # input. Fall through to the legacy splitlines path so an
        # all-whitespace token still surfaces as a render error from
        # ``render_paragraph`` (rather than from us, with a less
        # informative message).
    return [line.strip() for line in token.splitlines() if line.strip()] or [token.strip()]


# Paragraph boundary inside a ``pages``-mode page token: any run of
# 2+ consecutive newlines (with optional intra-run whitespace).
# Mirrors the tokenizer's ``_PARAGRAPH_SPLIT_RE`` so a corpus author's
# blank-line-separated paragraphs round-trip from corpus → tokenizer
# → renderer with the same boundary rule. The tokenizer's pages mode
# uses a **stronger** boundary (3+ newlines) for *page* breaks, so a
# single tokenizer-produced page token can carry multiple
# blank-line-separated paragraphs without being split apart.
_PAGE_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+", flags=re.UNICODE)


def _split_page_into_paragraphs(token: str) -> list[str]:
    r"""Split a pages-mode token into its constituent paragraph bodies.

    A page token is shaped like::

        paragraph 1 line 1
        paragraph 1 line 2

        paragraph 2 line 1

    where any run of 2+ newlines separates paragraphs (regex
    ``\n\s*\n+``). Whitespace-only chunks are dropped; a token with
    no embedded blank lines becomes a single-paragraph page (still
    legal — ``render_page`` accepts any number of paragraphs >= 1).
    """

    chunks = _PAGE_PARAGRAPH_SPLIT_RE.split(token)
    out = [chunk.strip() for chunk in chunks if chunk.strip()]
    return out or [token.strip()]


def _split_page_into_paragraph_lines(
    token: str,
    *,
    recipe: Recipe,
    ctx: RenderContext,
    page_style: PageStyle,
) -> list[list[str]]:
    """Turn a pages-mode token into the ``list[list[str]]`` shape
    :func:`render_page` expects.

    Each outer entry is one paragraph; each inner entry is one
    rendered line of that paragraph. We re-use the same wrap path as
    paragraphs mode (:func:`_split_paragraph_into_lines`) per inner
    paragraph, threading the page's pre-sampled paragraph style so
    fit_lines measures against the same font + pixel size the page
    renderer paints with.

    ``layout.paragraph_indent_px`` is forwarded to the wrap-fitter so
    each paragraph's first-line budget shrinks by the indent the page
    renderer will apply — without this an indented first line would
    overflow ``max_width_px`` by exactly ``paragraph_indent_px``
    pixels (and a justified first line would absorb the indent into
    its slack budget, painting beyond the user's wrap target).
    """

    paragraphs = _split_page_into_paragraphs(token)
    para_style = page_style.paragraph_style
    indent_px = recipe.layout.paragraph_indent_px or 0
    return [
        _split_paragraph_into_lines(
            p,
            recipe=recipe,
            ctx=ctx,
            style=para_style,
            first_line_indent_px=indent_px,
        )
        for p in paragraphs
    ]


def _render_sample(
    *,
    recipe: Recipe,
    text: str,
    ctx: RenderContext,
    apply_degrade: bool,
) -> tuple[object | None, list[dict[str, object]], str | None, dict[str, object]]:
    """Render + degrade one sample.

    Returns ``(sample_or_none, applied_degradations, skip_reason, details)``.
    ``sample_or_none`` is the rendered Pillow-backed sample on
    success, ``None`` on skip. ``applied_degradations`` is the list
    actually executed for the manifest. ``details`` carries
    skip-specific info (missing codepoints, error message).
    """

    try:
        if recipe.layout.mode == "lines":
            sample = render_line(text, recipe=recipe, ctx=ctx)
        elif recipe.layout.mode == "paragraphs":
            # Pre-sample the paragraph style **once** so wrap-fitting
            # measures against the same font + pixel size the renderer
            # will paint with. ``render_paragraph`` honors the
            # ``presampled`` arg and skips its own RNG draws when
            # given, keeping bit-identical output to the un-pre-
            # sampled path on the same seed.
            style = sample_paragraph_style(recipe, ctx)
            lines = _split_paragraph_into_lines(text, recipe=recipe, ctx=ctx, style=style)
            if not lines:
                return (
                    None,
                    [],
                    "render_error",
                    {"text": text, "message": "paragraph token has no non-empty lines"},
                )
            sample = render_paragraph(lines, recipe=recipe, ctx=ctx, presampled=style)
        elif recipe.layout.mode == "pages":
            # Same pre-sample dance as paragraphs, lifted to the page
            # level: one ``PageStyle`` per page, threaded through both
            # the wrap-fitter (so fit_lines measures against the
            # exact same font + pixel size the renderer paints with)
            # and ``render_page`` (so its internal RNG draws are
            # short-circuited and the result is bit-identical to the
            # un-pre-sampled path).
            page_style = sample_page_style(recipe, ctx)
            paragraph_lines = _split_page_into_paragraph_lines(
                text, recipe=recipe, ctx=ctx, page_style=page_style
            )
            if not paragraph_lines or not all(paragraph_lines):
                return (
                    None,
                    [],
                    "render_error",
                    {"text": text, "message": "page token has no non-empty paragraphs"},
                )
            sample = render_page(paragraph_lines, recipe=recipe, ctx=ctx, presampled=page_style)
        else:
            sample = render_word_crop(text, recipe=recipe, ctx=ctx)
    except MissingGlyphError as exc:
        return (
            None,
            [],
            "missing_glyph",
            {
                "text": text,
                "missing_codepoints": [f"U+{cp:04X}" for cp in sorted(exc.missing)],
                "font_path": str(exc.font_path),
            },
        )
    except RenderError as exc:
        return (
            None,
            [],
            "render_error",
            {"text": text, "message": str(exc)},
        )

    applied: list[dict[str, object]] = []
    if apply_degrade and recipe.degradation:
        try:
            sample = apply_degradation(sample, list(recipe.degradation), rng=ctx.rng)
            # The pipeline doesn't yet surface per-stage telemetry; we
            # report which stages were configured. Per-roll outcomes
            # land when the degradation pipeline grows that surface
            # (M09, when bbox-aware stages need it for detection).
            #
            # Spec 07 "Common keys" lists ``name`` as an optional label
            # recorded in the manifest. ``DegradationStage`` uses
            # ``extra="allow"`` so ``name`` (when present in the recipe)
            # rides along as a pydantic extra; ``getattr`` with a
            # ``None`` fallback keeps the manifest record stable for
            # stages that don't declare a label.
            applied = []
            for s in recipe.degradation:
                record: dict[str, object] = {
                    "kind": s.kind,
                    "probability": s.probability,
                }
                stage_name = getattr(s, "name", None)
                if stage_name is not None:
                    record["name"] = str(stage_name)
                applied.append(record)
        except DegradationError as exc:
            return (
                None,
                [],
                "degradation_error",
                {"text": text, "message": str(exc)},
            )
    return sample, applied, None, {}


# ---------------------------------------------------------------------------
# Serial path
# ---------------------------------------------------------------------------


def _drive_serial(
    *,
    pending: Sequence[tuple[int, str]],
    recipe: Recipe,
    writer: RecognitionWriter | DetectionWriter,
    seed: int,
    progress: _ProgressReporter,
) -> None:
    render_ctx = RenderContext.for_seed(seed)
    for index, token in pending:
        render_ctx.reseed_for_sample(index)
        sample, applied, reason, details = _render_sample(
            recipe=recipe,
            text=token,
            ctx=render_ctx,
            apply_degrade=True,
        )
        if sample is None:
            _writer_write_skipped(writer, index, reason=str(reason), text=token, details=details)
        else:
            _writer_write_rendered(writer, index, sample, text=token, applied=applied)
        progress.tick()


def _writer_write_rendered(
    writer: RecognitionWriter | DetectionWriter,
    index: int,
    sample: object,
    *,
    text: str,
    applied: list[dict[str, object]],
) -> None:
    """Call ``write_rendered`` with the kwargs each writer accepts.

    ``RecognitionWriter`` keys the on-disk label off the caller's
    ``text``; ``DetectionWriter`` derives per-line text from the
    sample's ``line_boxes`` so it doesn't accept a ``text`` kwarg.
    Centralizing the branch here keeps the drive loop writer-agnostic.
    """

    if isinstance(writer, RecognitionWriter):
        writer.write_rendered(index, sample, text=text, applied_degradations=applied)
    else:
        writer.write_rendered(index, sample, applied_degradations=applied)


def _writer_write_skipped(
    writer: RecognitionWriter | DetectionWriter,
    index: int,
    *,
    reason: str,
    text: str,
    details: dict[str, object],
) -> None:
    """Mirror of :func:`_writer_write_rendered` for skip records."""

    if isinstance(writer, RecognitionWriter):
        writer.write_skipped(index, reason=reason, text=text, details=details)
    else:
        # Detection writer's skip record has no top-level ``text`` slot
        # (the sample-shaped payload would normally come from line_boxes
        # on success). We still surface the source token in ``details``
        # so the manifest carries provenance — same precedent as the
        # ``missing_glyph`` details dict.
        merged = {"text": text, **details} if text and "text" not in details else dict(details)
        writer.write_skipped(index, reason=reason, details=merged)


# ---------------------------------------------------------------------------
# Parallel path
# ---------------------------------------------------------------------------
# The worker re-loads the recipe from disk inside its process (font
# handles aren't safe to share across forks anyway). The parent
# collects (index, render_payload) tuples and writes via the writer
# in the order they arrive — writer is index-addressed so completion
# order doesn't change on-disk results.


_WORKER_RECIPE: Recipe | None = None
_WORKER_CTX: RenderContext | None = None
_WORKER_SEED: int = 0


def _worker_init(recipe_path: str, seed: int) -> None:
    from pd_ocr_synth.recipe import load_recipe

    global _WORKER_RECIPE, _WORKER_CTX, _WORKER_SEED
    _WORKER_RECIPE = load_recipe(recipe_path)
    _WORKER_CTX = RenderContext.for_seed(seed)
    _WORKER_SEED = seed


def _worker_render(payload: tuple[int, str]) -> tuple[int, dict[str, object]]:
    index, token = payload
    assert _WORKER_RECIPE is not None
    assert _WORKER_CTX is not None
    _WORKER_CTX.reseed_for_sample(index)

    sample, applied, reason, details = _render_sample(
        recipe=_WORKER_RECIPE,
        text=token,
        ctx=_WORKER_CTX,
        apply_degrade=True,
    )
    if sample is None:
        return index, {
            "status": "skipped",
            "reason": reason,
            "text": token,
            "details": details,
        }

    # Pickle the PIL image's raw bytes + mode + size — Pillow Image
    # objects can be pickled directly, but bytes-explicit is faster
    # and lets us avoid pulling in the fork-safety footguns of the
    # PIL internal handles.
    from io import BytesIO

    buf = BytesIO()
    sample.image.save(buf, format="PNG")
    return index, {
        "status": "rendered",
        "text": token,
        "png_bytes": buf.getvalue(),
        "font_path": str(sample.font_path),
        "font_size_pt": float(sample.font_size_pt),
        "dpi": int(sample.dpi),
        "ink_color": list(sample.ink_color),
        "background_color": list(sample.background_color),
        "size": list(sample.size),
        "bbox": list(sample.bbox),
        "word_boxes": [{"text": wb.text, "bbox": list(wb.bbox)} for wb in sample.word_boxes],
        # ``line_boxes`` ride alongside ``word_boxes`` so the parent
        # process can rebuild a paragraph-shaped sample shim for the
        # detection writer. Empty for layouts that don't emit line GT
        # (``word_crops`` / ``lines``) — same shape as the in-process
        # path.
        "line_boxes": [
            {"text": lb.text, "bbox": list(lb.bbox)}
            for lb in getattr(sample, "line_boxes", ()) or ()
        ],
        # ``paragraph_boxes`` mirror the ``line_boxes`` round-trip so
        # the parent process can rebuild per-paragraph ground truth
        # for ``pages``-mode samples (and the single-entry degenerate
        # case from ``paragraphs`` mode). Empty for layouts that
        # don't emit paragraph GT.
        "paragraph_boxes": [
            {"text": pb.text, "bbox": list(pb.bbox)}
            for pb in getattr(sample, "paragraph_boxes", ()) or ()
        ],
        "applied_degradations": applied,
    }


def _drive_parallel(
    *,
    pending: Sequence[tuple[int, str]],
    recipe: Recipe,
    writer: RecognitionWriter | DetectionWriter,
    recipe_path: Path,
    seed: int,
    workers: int,
    progress: _ProgressReporter,
) -> None:
    if not pending:
        return
    ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.get_context()
    )
    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(str(recipe_path), seed),
    ) as pool:
        for index, payload in pool.imap_unordered(_worker_render, list(pending), chunksize=1):
            if payload["status"] == "skipped":
                _writer_write_skipped(
                    writer,
                    index,
                    reason=str(payload["reason"]),
                    text=str(payload.get("text") or ""),
                    details=dict(payload.get("details") or {}),
                )
            else:
                _write_parallel_rendered(writer, index, payload)
            progress.tick()


def _write_parallel_rendered(
    writer: RecognitionWriter | DetectionWriter, index: int, payload: dict
) -> None:
    """Decode the worker's pickled PNG bytes back into a sample stub.

    The writer expects something duck-typed as a ``RenderedSample``;
    we build a minimal ad-hoc object so the writer can ``image.save(...)``
    and read the recorded metadata. Both ``word_boxes`` and
    ``line_boxes`` are reconstructed from the payload so paragraph
    detection runs round-trip per-line ground truth through the worker
    boundary the same way recognition rounds-trip per-word boxes.
    """

    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(payload["png_bytes"]))
    image.load()  # detach from the BytesIO

    class _ParallelSample:
        pass

    from pd_ocr_synth.render.sample import LineBox, ParagraphBox, WordBox

    s = _ParallelSample()
    s.image = image  # type: ignore[attr-defined]
    s.text = payload["text"]  # type: ignore[attr-defined]
    s.font_path = Path(payload["font_path"])  # type: ignore[attr-defined]
    s.font_size_pt = payload["font_size_pt"]  # type: ignore[attr-defined]
    s.dpi = payload["dpi"]  # type: ignore[attr-defined]
    s.ink_color = tuple(payload["ink_color"])  # type: ignore[attr-defined]
    s.background_color = tuple(payload["background_color"])  # type: ignore[attr-defined]
    s.size = tuple(payload["size"])  # type: ignore[attr-defined]
    s.bbox = tuple(payload["bbox"])  # type: ignore[attr-defined]
    s.glyph_runs = ()  # type: ignore[attr-defined]
    s.word_boxes = tuple(  # type: ignore[attr-defined]
        WordBox(text=str(wb["text"]), bbox=tuple(wb["bbox"]))  # type: ignore[arg-type]
        for wb in payload.get("word_boxes") or ()
    )
    s.line_boxes = tuple(  # type: ignore[attr-defined]
        LineBox(text=str(lb["text"]), bbox=tuple(lb["bbox"]))  # type: ignore[arg-type]
        for lb in payload.get("line_boxes") or ()
    )
    s.paragraph_boxes = tuple(  # type: ignore[attr-defined]
        ParagraphBox(text=str(pb["text"]), bbox=tuple(pb["bbox"]))  # type: ignore[arg-type]
        for pb in payload.get("paragraph_boxes") or ()
    )

    _writer_write_rendered(
        writer,
        index,
        s,
        text=payload["text"],
        applied=payload.get("applied_degradations") or [],
    )


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------


class _ProgressReporter:
    """Minimal stderr progress reporter — no tqdm dep.

    Updates at most every 0.5s so a tight loop doesn't flood stderr.
    Always emits a final summary line on ``done()`` so the user sees
    the rate even on a fast run that never hit the throttle.
    """

    _MIN_INTERVAL_S = 0.5

    def __init__(self, *, total: int, enabled: bool = True) -> None:
        self.total = total
        self.enabled = enabled and sys.stderr.isatty()
        # Always-on (non-tty) line at the end via ``done()``; the
        # in-progress refresh only fires for tty.
        self._count = 0
        self._started = time.monotonic()
        self._last_report = self._started

    def tick(self) -> None:
        self._count += 1
        if not self.enabled:
            return
        now = time.monotonic()
        if now - self._last_report < self._MIN_INTERVAL_S and self._count < self.total:
            return
        self._last_report = now
        self._render_inline(now)

    def done(self) -> None:
        elapsed = time.monotonic() - self._started
        rate = self._count / elapsed if elapsed > 0 else 0.0
        if self.enabled:
            # Newline to drop off the carriage-return line.
            print(file=sys.stderr)
        print(
            f"rendered {self._count}/{self.total} in {elapsed:.1f}s ({rate:.1f} samples/s)",
            file=sys.stderr,
        )

    def _render_inline(self, now: float) -> None:
        elapsed = now - self._started
        rate = self._count / elapsed if elapsed > 0 else 0.0
        # Carriage-return-only update so successive lines overwrite.
        bar_total = max(1, self.total)
        pct = self._count / bar_total
        width = 30
        filled = int(width * pct)
        bar = "#" * filled + "-" * (width - filled)
        print(
            f"\r[{bar}] {self._count}/{self.total} ({pct * 100:5.1f}%) {rate:.1f} samples/s ",
            file=sys.stderr,
            end="",
            flush=True,
        )
