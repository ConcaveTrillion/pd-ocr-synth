"""Preview-mode dataset loop for ``pd-ocr-synth preview``.

Renders N samples to a directory, no degradation, no trainer-format
output adapter — just enough to spot-check the render layer.

Output layout (per docs/roadmap/05-rendering.md "Validation criteria"):

```
<output>/
    images/           # PNG files, one per rendered sample
    manifest.jsonl    # one JSON record per attempted sample
    stats.json        # summary: counts + skip reasons
```

Each manifest line carries either ``status: "rendered"`` (with the
full ground-truth payload) or ``status: "skipped"`` with a
``reason`` (currently ``missing_glyph`` or ``render_error``).

## Parallelism

When ``workers > 1`` the per-sample render fan out across a
``multiprocessing.Pool``. Each worker initializes its own
``RenderContext`` once via ``Pool(initializer=...)`` so freetype /
HarfBuzz font handles cache across that worker's samples.

Determinism is preserved: PNG file names and per-sample RNG state
are both keyed on the sample index (via ``branched_seed`` +
``RenderContext.reseed_for_sample``), so worker count and
completion order do not influence output bytes. Manifest lines are
collected by index and written in sorted order, regardless of the
order in which workers finish.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pd_ocr_synth.corpus import CacheStore, ProviderContext, default_cache_root
from pd_ocr_synth.corpus.runner import collect_corpus_text
from pd_ocr_synth.degradation import apply_degradation
from pd_ocr_synth.render.context import RenderContext
from pd_ocr_synth.render.word_crop import (
    MissingGlyphError,
    RenderError,
    render_word_crop,
)
from pd_ocr_synth.tokenization import tokenize

if TYPE_CHECKING:
    from pd_ocr_synth.recipe import Recipe


DEFAULT_PREVIEW_COUNT = 50

#: Hard cap on the auto-resolved worker count. Eight is a sensible
#: ceiling for a CPU-bound rendering job — beyond that, contention
#: on the GIL-released C extensions (freetype / HarfBuzz / Pillow)
#: and disk I/O usually outweighs the parallelism win on a dev box.
_DEFAULT_WORKER_CAP = 8


def resolve_workers(requested: int | None) -> int:
    """Pick a sensible worker count.

    - ``requested is None`` → auto: ``max(1, min(cpu_count - 1, 8))``.
      We leave one core free on multi-core machines so the dev box
      stays usable while a large render is in flight.
    - ``requested`` set → use it verbatim, clamped to ``>= 1``.
    """

    if requested is not None:
        return max(1, int(requested))
    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return max(1, cpu_count)
    return max(1, min(cpu_count - 1, _DEFAULT_WORKER_CAP))


@dataclass(frozen=True, slots=True)
class PreviewStats:
    """Summary counters returned to the CLI."""

    recipe: str
    seed: int
    count: int
    rendered: int
    skipped: int
    skip_reasons: dict[str, int] = field(default_factory=dict)
    output_dir: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


def run_preview(
    recipe: Recipe,
    *,
    output_dir: Path,
    count: int = DEFAULT_PREVIEW_COUNT,
    seed: int | None = None,
    cache_dir: Path | None = None,
    workers: int = 1,
    apply_degrade: bool = True,
    no_cache: bool = False,
) -> PreviewStats:
    """Render ``count`` samples from ``recipe`` into ``output_dir``.

    The seed defaults to the recipe seed; pass ``seed=...`` to override.
    Cache root defaults to ``$PD_OCR_SYNTH_CACHE`` / the per-user cache
    if ``cache_dir`` is omitted.

    ``workers`` defaults to 1 (in-process serial path; matches the
    pre-parallelism behavior). When ``workers > 1`` the per-sample
    render fans out across a ``multiprocessing.Pool``. Output is
    deterministic regardless of worker count: PNG file names and
    per-sample RNG state are keyed on the sample index, and manifest
    lines are written in sample-index order.

    ``apply_degrade`` (default ``True``) runs the recipe's
    ``degradation`` pipeline against each rendered sample. Pass
    ``False`` to inspect the raw render output (useful when
    debugging the renderer itself).

    ``no_cache`` (default ``False``) wires through the ``--no-cache``
    CLI flag: when ``True`` the corpus runner bypasses the on-disk
    cache and re-fetches every web/wikisource entry from upstream.
    """

    if recipe.layout.mode != "word_crops":
        # Lines / paragraphs / pages are M09. Surface a clear error
        # rather than silently mis-rendering.
        raise RenderError(
            f"preview only supports layout.mode=word_crops in M05; got {recipe.layout.mode!r}"
        )

    effective_seed = recipe.seed if seed is None else int(seed)

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    cache_root = cache_dir or default_cache_root()
    if recipe.source_path is None:
        raise RenderError("recipe has no source_path; load it via load_recipe(path)")
    ctx = ProviderContext(recipe_dir=recipe.source_path.parent, cache=CacheStore(root=cache_root))

    text = collect_corpus_text(recipe, ctx=ctx, no_cache=no_cache)
    tokens = tokenize(text, mode=recipe.layout.mode)
    if not tokens:
        raise RenderError("corpus produced no tokens after tokenization")

    pick_rng = random.Random(effective_seed ^ 0xC0FFEE)
    chosen: list[str] = [pick_rng.choice(tokens) for _ in range(count)]

    worker_count = max(1, int(workers))
    if worker_count == 1:
        records = _render_serial(
            chosen,
            recipe=recipe,
            images_dir=images_dir,
            seed=effective_seed,
            apply_degrade=apply_degrade,
        )
    else:
        records = _render_parallel(
            chosen,
            recipe_path=recipe.source_path,
            images_dir=images_dir,
            seed=effective_seed,
            workers=worker_count,
            apply_degrade=apply_degrade,
        )

    rendered = 0
    skipped = 0
    skip_reasons: dict[str, int] = {}
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in records:  # already in sample-index order
            if record["status"] == "rendered":
                rendered += 1
            else:
                skipped += 1
                reason = record.get("reason", "unknown")
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = PreviewStats(
        recipe=recipe.name,
        seed=effective_seed,
        count=count,
        rendered=rendered,
        skipped=skipped,
        skip_reasons=skip_reasons,
        output_dir=str(output_dir),
    )
    (output_dir / "stats.json").write_text(
        json.dumps(stats.as_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return stats


# ---------------------------------------------------------------------------
# Serial path (workers == 1) — kept identical to the pre-parallelism flow
# so single-worker stays simple and debuggable (no Pool overhead, full
# tracebacks, no fork dance).
# ---------------------------------------------------------------------------


def _render_serial(
    chosen: Sequence[str],
    *,
    recipe: Recipe,
    images_dir: Path,
    seed: int,
    apply_degrade: bool,
) -> list[dict]:
    render_ctx = RenderContext.for_seed(seed)
    out: list[dict] = []
    for index, token in enumerate(chosen):
        render_ctx.reseed_for_sample(index)
        out.append(
            _render_one(
                token,
                index=index,
                recipe=recipe,
                ctx=render_ctx,
                images_dir=images_dir,
                seed=seed,
                apply_degrade=apply_degrade,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Parallel path (workers > 1) — multiprocessing.Pool with imap_unordered.
# Each worker loads the recipe once via the pool initializer and builds
# its own RenderContext (font handles are not safe to share across
# processes). Records come back keyed by sample index and are sorted
# before manifest writing, so manifest order is independent of worker
# completion order.
# ---------------------------------------------------------------------------


# Per-worker module globals populated by ``_worker_init``. Keeping the
# heavy state out of the per-task payload means each ``imap`` call
# only ships a tiny ``(index, token)`` tuple to the worker.
_WORKER_RECIPE: Recipe | None = None
_WORKER_CTX: RenderContext | None = None
_WORKER_IMAGES_DIR: Path | None = None
_WORKER_SEED: int = 0
_WORKER_APPLY_DEGRADE: bool = True


def _worker_init(recipe_path: str, seed: int, images_dir: str, apply_degrade: bool) -> None:
    """Pool initializer — runs once per worker process.

    Re-loads the recipe from disk inside the worker rather than
    relying on pickle to copy it. Same end state, more explicit, and
    matches how the worker would naturally be re-entered if we ever
    swap to spawn-based pools on a non-fork platform.
    """

    # Imported here so the parent process doesn't pay yaml/pydantic
    # startup cost when running serially.
    from pd_ocr_synth.recipe import load_recipe

    global _WORKER_RECIPE, _WORKER_CTX, _WORKER_IMAGES_DIR, _WORKER_SEED, _WORKER_APPLY_DEGRADE
    _WORKER_RECIPE = load_recipe(recipe_path)
    _WORKER_CTX = RenderContext.for_seed(seed)
    _WORKER_IMAGES_DIR = Path(images_dir)
    _WORKER_SEED = seed
    _WORKER_APPLY_DEGRADE = apply_degrade


def _worker_render(payload: tuple[int, str]) -> tuple[int, dict]:
    index, token = payload
    assert _WORKER_RECIPE is not None
    assert _WORKER_CTX is not None
    assert _WORKER_IMAGES_DIR is not None

    _WORKER_CTX.reseed_for_sample(index)
    record = _render_one(
        token,
        index=index,
        recipe=_WORKER_RECIPE,
        ctx=_WORKER_CTX,
        images_dir=_WORKER_IMAGES_DIR,
        seed=_WORKER_SEED,
        apply_degrade=_WORKER_APPLY_DEGRADE,
    )
    return index, record


def _render_parallel(
    chosen: Sequence[str],
    *,
    recipe_path: Path,
    images_dir: Path,
    seed: int,
    workers: int,
    apply_degrade: bool,
) -> list[dict]:
    payloads = list(enumerate(chosen))
    # Per-record slot keyed by sample index, so we can write the
    # manifest in deterministic order regardless of completion order.
    results: list[dict | None] = [None] * len(payloads)

    # Use ``fork`` where available (Linux) — it's both faster and lets
    # the child see any module-level state the parent already set up.
    # Fall back to the platform default elsewhere.
    ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.get_context()
    )

    with ctx.Pool(
        processes=workers,
        initializer=_worker_init,
        initargs=(str(recipe_path), seed, str(images_dir), apply_degrade),
    ) as pool:
        # ``chunksize=1`` keeps load balancing tight; per-sample render
        # is heavy enough (font shaping + rasterization + PNG encode)
        # that the per-task IPC overhead is negligible.
        for index, record in pool.imap_unordered(_worker_render, payloads, chunksize=1):
            results[index] = record

    # All slots must be filled; assert defensively in case a worker
    # raised and ``imap_unordered`` swallowed it (shouldn't happen,
    # but it's a quick sanity check).
    for i, rec in enumerate(results):
        if rec is None:  # pragma: no cover - defensive
            raise RenderError(f"worker pool produced no record for sample index {i}")
    return [rec for rec in results if rec is not None]


def _render_one(
    text: str,
    *,
    index: int,
    recipe: Recipe,
    ctx: RenderContext,
    images_dir: Path,
    seed: int,
    apply_degrade: bool,
) -> dict:
    image_name = f"{seed:08x}_{index:06d}.png"
    image_path = images_dir / image_name
    try:
        sample = render_word_crop(text, recipe=recipe, ctx=ctx)
    except MissingGlyphError as exc:
        return {
            "index": index,
            "text": text,
            "status": "skipped",
            "reason": "missing_glyph",
            "missing_codepoints": [f"U+{cp:04X}" for cp in sorted(exc.missing)],
            "font_path": str(exc.font_path),
        }
    except RenderError as exc:
        return {
            "index": index,
            "text": text,
            "status": "skipped",
            "reason": "render_error",
            "message": str(exc),
        }

    # Run the recipe's degradation pipeline on the rendered sample.
    # ``ctx.rng`` is the per-sample-branched RNG; degradation continues
    # to draw from it so determinism keys (recipe + seed + index) stay
    # the same. We surface degradation errors as a render skip rather
    # than crashing the whole batch — a malformed stage shouldn't
    # poison the rest of the preview.
    if apply_degrade and recipe.degradation:
        from pd_ocr_synth.degradation import DegradationError

        try:
            sample = apply_degradation(sample, list(recipe.degradation), rng=ctx.rng)
        except DegradationError as exc:
            return {
                "index": index,
                "text": text,
                "status": "skipped",
                "reason": "degradation_error",
                "message": str(exc),
            }

    sample.image.save(image_path, format="PNG")
    return {
        "index": index,
        "image": f"images/{image_name}",
        "text": sample.text,
        "status": "rendered",
        "font_path": str(sample.font_path),
        "font_size_pt": sample.font_size_pt,
        "dpi": sample.dpi,
        "ink_color": list(sample.ink_color),
        "background_color": list(sample.background_color),
        "size": list(sample.size),
        "bbox": list(sample.bbox),
        "glyph_runs": [{"cluster": g.cluster, "bbox": list(g.bbox)} for g in sample.glyph_runs],
    }


__all__: Sequence[str] = (
    "DEFAULT_PREVIEW_COUNT",
    "PreviewStats",
    "resolve_workers",
    "run_preview",
)
