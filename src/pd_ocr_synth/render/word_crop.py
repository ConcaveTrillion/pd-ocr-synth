"""Render a single ``word_crops``-mode sample.

Pipeline per sample:

1. Pick a font by recipe-weighted choice.
2. Draw size_pt + dpi from the recipe rendering distributions.
3. Shape the text via uharfbuzz: glyph IDs + per-glyph (x, y, advance).
4. For each glyph, ask freetype to rasterize its bitmap at the same
   pixel size, then composite onto a Pillow canvas.
5. Tight-crop and add per-side padding sampled from ``layout.padding_px``.

The renderer assumes ``shaping_engine: harfbuzz``. Pillow-only fallback
(no shaping) lands later if a recipe demands it.
"""

from __future__ import annotations

from pathlib import Path
from random import Random
from typing import TYPE_CHECKING

import freetype
import uharfbuzz as hb
from PIL import Image

from pd_ocr_synth.fonts import open_font
from pd_ocr_synth.render.context import RenderContext
from pd_ocr_synth.render.sample import GlyphRun, RenderedSample
from pd_ocr_synth.render.sampling import sample_color, sample_value

if TYPE_CHECKING:
    from pd_ocr_synth.recipe import Recipe


# Default OpenType features when neither the font nor the recipe
# overrides. Mirrors the spec ("liga and calt on by default").
_DEFAULT_FEATURES: dict[str, bool] = {"liga": True, "calt": True}


# Cache codepoint sets per resolved font path so the per-sample
# coverage check is O(1) after the first probe.
_COVERAGE_CACHE: dict[str, frozenset[int]] = {}


class RenderError(Exception):
    """Raised when a sample cannot be rendered (e.g. all glyphs missing)."""


class MissingGlyphError(RenderError):
    """Raised when the chosen font does not cover one of the input codepoints.

    The dataset loop catches this and records ``missing_glyph`` as the
    skip reason in the manifest (per docs/specs/06-rendering.md and
    docs/roadmap/05-rendering.md).
    """

    def __init__(self, text: str, font_path: Path, missing: set[int]) -> None:
        self.text = text
        self.font_path = font_path
        self.missing = missing
        codepoints = ", ".join(f"U+{cp:04X}" for cp in sorted(missing))
        super().__init__(f"font {font_path.name} does not cover {codepoints} for text {text!r}")


def render_word_crop(
    text: str,
    *,
    recipe: Recipe,
    ctx: RenderContext,
) -> RenderedSample:
    """Render ``text`` as one tight word-crop sample.

    Raises:
        MissingGlyphError: if the picked font doesn't cover every
            codepoint in ``text``. (Pre-shape: skips the cost of the
            HarfBuzz call when we already know it would emit
            ``.notdef`` glyphs.)
        RenderError: for other render failures (no usable fonts,
            shaping returned no glyphs).
    """

    font = _pick_font(recipe, ctx.rng)
    font_size_pt = float(sample_value(recipe.rendering.font_size_pt, ctx.rng))
    dpi = int(sample_value(recipe.rendering.dpi, ctx.rng))
    ink = sample_color(recipe.rendering.ink_color, ctx.rng)
    bg = sample_color(recipe.rendering.background_color, ctx.rng)
    padding = int(sample_value(recipe.layout.padding_px or 0, ctx.rng))

    # Codepoint-coverage check before we spin up shaping. Cheaper to
    # skip a missing-glyph sample here than to render a row of tofu.
    missing = _missing_codepoints(font.path, text)
    if missing:
        raise MissingGlyphError(text, font.path, missing)

    handles = ctx.font_handles(font.path)
    pixel_size = max(1, round(font_size_pt * dpi / 72.0))
    handles.ft_face.set_pixel_sizes(pixel_size, pixel_size)

    info_glyphs, positions = _shape(handles.hb_face, text, pixel_size, font.features)
    if not info_glyphs:
        raise RenderError(f"shaping returned no glyphs for {text!r}")

    bitmaps = _rasterize_glyphs(handles.ft_face, info_glyphs)
    image, bbox, runs = _composite(bitmaps, info_glyphs, positions, ink, bg, padding)

    return RenderedSample(
        text=text,
        image=image,
        bbox=bbox,
        font_path=font.path,
        font_size_pt=font_size_pt,
        dpi=dpi,
        ink_color=ink,
        background_color=bg,
        glyph_runs=tuple(runs),
    )


# ---------------------------------------------------------------------------
# Font selection
# ---------------------------------------------------------------------------


def _missing_codepoints(font_path: Path, text: str) -> set[int]:
    """Return codepoints in ``text`` not covered by ``font_path``.

    The chosen font's full codepoint set is cached on the first probe
    so subsequent samples are a cheap set-difference.
    """

    key = str(font_path.resolve())
    coverage = _COVERAGE_CACHE.get(key)
    if coverage is None:
        info = open_font(font_path)
        coverage = info.codepoints
        _COVERAGE_CACHE[key] = coverage
    return {ord(ch) for ch in text if ord(ch) not in coverage}


def _pick_font(recipe: Recipe, rng: Random):
    """Choose one ``Font`` entry by weight, skipping missing optional fonts."""

    eligible = [f for f in recipe.fonts if f.path.exists()]
    if not eligible:
        raise RenderError("no usable fonts (every entry is missing on disk)")
    weights = [max(f.weight, 0.0) for f in eligible]
    total = sum(weights)
    if total <= 0:
        return rng.choice(eligible)
    pick = rng.uniform(0.0, total)
    acc = 0.0
    for entry, weight in zip(eligible, weights, strict=True):
        acc += weight
        if pick <= acc:
            return entry
    return eligible[-1]


# ---------------------------------------------------------------------------
# Shaping
# ---------------------------------------------------------------------------


def _shape(
    hb_face: hb.Face,
    text: str,
    pixel_size: int,
    feature_overrides: dict | None,
) -> tuple[list[hb.GlyphInfo], list[hb.GlyphPosition]]:
    hb_font = hb.Font(hb_face)
    hb_font.scale = (pixel_size * 64, pixel_size * 64)
    hb.ot_font_set_funcs(hb_font)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()

    features = dict(_DEFAULT_FEATURES)
    if feature_overrides:
        features.update(feature_overrides)
    hb.shape(hb_font, buf, features)

    return list(buf.glyph_infos), list(buf.glyph_positions)


# ---------------------------------------------------------------------------
# Rasterization
# ---------------------------------------------------------------------------


def _rasterize_glyphs(face: freetype.Face, glyphs: list[hb.GlyphInfo]):
    """Return a list of (bitmap_array, left, top, width, height) tuples."""

    out = []
    for info in glyphs:
        face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        bmp = face.glyph.bitmap
        out.append(
            {
                "buffer": bytes(bmp.buffer),
                "width": bmp.width,
                "rows": bmp.rows,
                "left": face.glyph.bitmap_left,
                "top": face.glyph.bitmap_top,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def _composite(
    bitmaps,
    glyphs,
    positions,
    ink: tuple[int, int, int],
    bg: tuple[int, int, int],
    padding: int,
):
    # Pen position in pixel coords. Start at (0, 0); we tight-crop at
    # the end so the pen origin can be arbitrary.
    pen_x = 0.0
    pen_y = 0.0
    placements = []  # (bm, x_int, y_int)

    for bm, pos, info in zip(bitmaps, positions, glyphs, strict=True):
        x_offset = pos.x_offset / 64.0
        y_offset = pos.y_offset / 64.0
        x = pen_x + x_offset + bm["left"]
        # Pillow uses top-left origin; freetype's bitmap_top is the
        # baseline-to-top distance, so glyph top edge is pen_y -
        # bitmap_top (with HarfBuzz y-offsets in font coords).
        y = pen_y - y_offset - bm["top"]
        placements.append((bm, round(x), round(y), info.cluster))
        pen_x += pos.x_advance / 64.0
        pen_y += pos.y_advance / 64.0

    if not placements:
        raise RenderError("nothing to composite")

    # Compute pre-padding bbox of inked region.
    min_x = min(x for _, x, _, _ in placements)
    min_y = min(y for _, _, y, _ in placements)
    max_x = max(x + bm["width"] for bm, x, _, _ in placements)
    max_y = max(y + bm["rows"] for bm, _, y, _ in placements)

    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)

    img_w = width + 2 * padding
    img_h = height + 2 * padding

    canvas = Image.new("RGB", (img_w, img_h), color=bg)
    runs: list[GlyphRun] = []

    for bm, x, y, cluster in placements:
        if bm["width"] == 0 or bm["rows"] == 0:
            continue
        # Build a single-channel coverage image, then paste using the
        # ink color and the coverage mask.
        coverage = Image.frombytes("L", (bm["width"], bm["rows"]), bm["buffer"])
        ink_swatch = Image.new("RGB", coverage.size, color=ink)
        paste_x = x - min_x + padding
        paste_y = y - min_y + padding
        canvas.paste(ink_swatch, (paste_x, paste_y), mask=coverage)
        runs.append(
            GlyphRun(
                cluster=cluster,
                bbox=(
                    paste_x,
                    paste_y,
                    paste_x + bm["width"],
                    paste_y + bm["rows"],
                ),
            )
        )

    bbox = (padding, padding, padding + width, padding + height)
    return canvas, bbox, runs
