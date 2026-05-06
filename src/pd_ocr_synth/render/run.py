"""End-to-end render orchestration for recognition mode (M07).

``run_recipe`` ties M03-M06 together into the trainer-consumable
profile layout described in :mod:`pd_ocr_synth.output.recognition`:

1. Run corpus providers + recipe-level text transforms.
2. Tokenize for the recipe's layout mode.
3. Pick ``count`` deterministic samples.
4. Render + degrade each sample, streaming results into a
   :class:`RecognitionWriter`.
5. Honor ``--force`` / ``--resume`` semantics, including snapshot
   integrity checks.

Word-crop layout is the only mode supported in M07 — line, paragraph,
and page layouts (and detection-mode output) are M09. Anything else
raises :class:`RenderError` up front so callers fail loudly.

Determinism is the same as :mod:`pd_ocr_synth.render.preview`: the
per-token pick is keyed on ``seed ^ 0xC0FFEE``; per-sample render +
degradation RNGs are reseeded by sample index. Worker count is not
load-bearing on output bytes.
"""

from __future__ import annotations

import multiprocessing
import random
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pd_ocr_synth.corpus import CacheStore, ProviderContext, default_cache_root
from pd_ocr_synth.corpus.runner import collect_corpus_text
from pd_ocr_synth.degradation import DegradationError, apply_degradation
from pd_ocr_synth.output import RecognitionWriter
from pd_ocr_synth.render.context import RenderContext
from pd_ocr_synth.render.word_crop import (
    MissingGlyphError,
    RenderError,
    render_word_crop,
)
from pd_ocr_synth.tokenization import tokenize

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
) -> RunPlan:
    """Compute the dry-run plan without writing.

    Touches the network only insofar as the corpus runner does (to
    determine total chars). Callers that want a fully offline plan
    should pre-fetch first via ``pd-ocr-synth fetch``.
    """

    if recipe.layout.mode != "word_crops":
        raise RenderError(
            f"render only supports layout.mode=word_crops in M07; got {recipe.layout.mode!r}. "
            "Lines / paragraphs / pages land in M09."
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
    text = collect_corpus_text(recipe, ctx=ctx)

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
) -> RunResult:
    """Render the full dataset for ``recipe`` into ``output_dir``.

    Parameters mirror the CLI flags. ``count`` overrides
    ``recipe.output.count`` for smoke tests; ``seed`` overrides the
    recipe seed; ``workers`` is the multiprocessing pool size (1 =
    in-process serial path); ``cache_dir`` overrides the corpus cache
    root. ``force`` clears the destination first; ``resume`` continues
    from any existing samples after validating the snapshot.

    Determinism contract: same recipe + same effective seed + same
    sample index → identical output bytes regardless of ``workers``.
    """

    if recipe.layout.mode != "word_crops":
        raise RenderError(
            f"render only supports layout.mode=word_crops in M07; got {recipe.layout.mode!r}"
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

    text = collect_corpus_text(recipe, ctx=ctx)
    tokens = tokenize(text, mode=recipe.layout.mode)
    if not tokens:
        raise RenderError("corpus produced no tokens after tokenization")

    pick_rng = random.Random(effective_seed ^ 0xC0FFEE)
    chosen: list[str] = [pick_rng.choice(tokens) for _ in range(effective_count)]

    started = time.monotonic()
    progress_reporter = _ProgressReporter(total=effective_count, enabled=progress)

    with RecognitionWriter.open(
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

    progress_reporter.done()

    return RunResult(
        output_dir=str(output_dir),
        rendered=writer.stats.samples_written,
        skipped=writer.stats.samples_skipped,
        skip_reasons=dict(writer.stats.skip_reasons),
        wall_time_seconds=writer.stats.wall_time_seconds,
    )


# ---------------------------------------------------------------------------
# Per-sample render core (shared between serial + parallel paths)
# ---------------------------------------------------------------------------


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
            applied = [{"kind": s.kind, "probability": s.probability} for s in recipe.degradation]
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
    writer: RecognitionWriter,
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
            writer.write_skipped(index, reason=str(reason), text=token, details=details)
        else:
            writer.write_rendered(
                index,
                sample,
                text=token,
                applied_degradations=applied,
            )
        progress.tick()


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
        "applied_degradations": applied,
    }


def _drive_parallel(
    *,
    pending: Sequence[tuple[int, str]],
    recipe: Recipe,
    writer: RecognitionWriter,
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
                writer.write_skipped(
                    index,
                    reason=str(payload["reason"]),
                    text=str(payload.get("text") or ""),
                    details=dict(payload.get("details") or {}),
                )
            else:
                _write_parallel_rendered(writer, index, payload)
            progress.tick()


def _write_parallel_rendered(writer: RecognitionWriter, index: int, payload: dict) -> None:
    """Decode the worker's pickled PNG bytes back into a sample stub.

    The writer expects something duck-typed as a ``RenderedSample``;
    we build a minimal ad-hoc object so the writer can ``image.save(...)``
    and read the recorded metadata.
    """

    from io import BytesIO

    from PIL import Image

    image = Image.open(BytesIO(payload["png_bytes"]))
    image.load()  # detach from the BytesIO

    class _ParallelSample:  # noqa: N801 - one-shot data shim
        pass

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

    writer.write_rendered(
        index,
        s,
        text=payload["text"],
        applied_degradations=payload.get("applied_degradations") or [],
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
