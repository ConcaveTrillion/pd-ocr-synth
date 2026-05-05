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
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from pd_ocr_synth.corpus import CacheStore, ProviderContext, default_cache_root
from pd_ocr_synth.corpus.runner import collect_corpus_text
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
) -> PreviewStats:
    """Render ``count`` samples from ``recipe`` into ``output_dir``.

    The seed defaults to the recipe seed; pass ``seed=...`` to override.
    Cache root defaults to ``$PD_OCR_SYNTH_CACHE`` / the per-user cache
    if ``cache_dir`` is omitted.
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

    text = collect_corpus_text(recipe, ctx=ctx)
    tokens = tokenize(text, mode=recipe.layout.mode)
    if not tokens:
        raise RenderError("corpus produced no tokens after tokenization")

    pick_rng = random.Random(effective_seed ^ 0xC0FFEE)
    chosen: list[str] = [pick_rng.choice(tokens) for _ in range(count)]

    render_ctx = RenderContext.for_seed(effective_seed)

    manifest_path = output_dir / "manifest.jsonl"
    rendered = 0
    skipped = 0
    skip_reasons: dict[str, int] = {}

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, token in enumerate(chosen):
            render_ctx.reseed_for_sample(index)
            record = _render_one(
                token,
                index=index,
                recipe=recipe,
                ctx=render_ctx,
                images_dir=images_dir,
                seed=effective_seed,
            )
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


def _render_one(
    text: str,
    *,
    index: int,
    recipe: Recipe,
    ctx: RenderContext,
    images_dir: Path,
    seed: int,
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


__all__: Sequence[str] = ("PreviewStats", "DEFAULT_PREVIEW_COUNT", "run_preview")
