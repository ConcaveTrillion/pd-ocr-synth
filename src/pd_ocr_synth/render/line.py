"""Render a single ``lines``-mode sample.

A ``lines`` sample is N words on one baseline, output as a single
image whose ground-truth text is the full line. Per spec 06:

    layout:
      mode: lines
      max_width_px: 800
      padding_px: { min: 4, max: 12 }
      word_spacing: { min: 1.0, max: 1.4 }
      baseline_jitter_px: { min: -2, max: 2 }

The renderer shapes the *entire line* in one HarfBuzz buffer so any
cross-word contextual shaping rules (e.g. Arabic joining, Latin
contextual alternates) still apply. Per-word bounding boxes are
recovered after the fact by mapping each glyph's ``cluster`` (the
input codepoint offset that produced it) back to a word index.

The recipe-level ``word_spacing`` knob from spec 06 is **not yet
plumbed in** — that field is not on the recipe model as of this
chunk. Native space-glyph advance is used. Adding ``word_spacing``
is a separate (small) follow-up: the recipe model needs the field
plus a multiplier on space x_advance below.

``max_width_px`` is also intentionally not enforced here. Per the
spec it's the wrap budget for the layout engine that decides how
many words fit; in this chunk the caller is expected to hand
``render_line`` a pre-fitted line. The wrap-fitting logic (the
``LineLayout`` packer) lands in the next M09 chunk.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from pd_ocr_synth.render.context import RenderContext
from pd_ocr_synth.render.sample import GlyphRun, RenderedSample, WordBox
from pd_ocr_synth.render.sampling import sample_color, sample_value
from pd_ocr_synth.render.word_crop import (
    MissingGlyphError,
    RenderError,
    _missing_codepoints,
    _pick_font,
    _rasterize_glyphs,
    _shape,
)

if TYPE_CHECKING:
    from pd_ocr_synth.recipe import Recipe


def render_line(
    text: str,
    *,
    recipe: Recipe,
    ctx: RenderContext,
) -> RenderedSample:
    """Render ``text`` as one ``lines``-mode sample.

    ``text`` is the full line — multiple words separated by ASCII
    spaces. The resulting :class:`RenderedSample` carries:

    - ``text`` = the input line (unchanged).
    - ``image`` = a tight rectangular crop around the inked region
      with per-side padding sampled from ``layout.padding_px``.
    - ``glyph_runs`` = per-cluster bboxes (same as ``word_crops``).
    - ``word_boxes`` = per-word text + tight bbox, in input order.

    Raises:
        MissingGlyphError: if any non-whitespace codepoint in ``text``
            is not covered by the chosen font. (Whitespace gaps are
            permitted to render as empty space; the font almost
            certainly has a space glyph anyway.)
        RenderError: for other render failures (no usable fonts,
            shaping returned no glyphs, or a line containing only
            whitespace).
    """

    # Reject pathological inputs early. Empty / whitespace-only
    # strings are not meaningful samples and would produce zero-glyph
    # composites that downstream code would have to special-case.
    if not text or not text.strip():
        raise RenderError(f"render_line requires non-empty, non-whitespace text; got {text!r}")

    font = _pick_font(recipe, ctx.rng)
    font_size_pt = float(sample_value(recipe.rendering.font_size_pt, ctx.rng))  # pyright: ignore[reportArgumentType]
    dpi = int(sample_value(recipe.rendering.dpi, ctx.rng))  # pyright: ignore[reportArgumentType]
    ink = sample_color(recipe.rendering.ink_color, ctx.rng)
    bg = sample_color(recipe.rendering.background_color, ctx.rng)
    padding = int(sample_value(recipe.layout.padding_px or 0, ctx.rng))  # pyright: ignore[reportArgumentType]

    # Coverage check excludes whitespace: a missing space glyph would
    # render as nothing, which is also what we want (just an advance).
    non_ws_text = "".join(ch for ch in text if not ch.isspace())
    missing = _missing_codepoints(font.path, non_ws_text)
    if missing:
        raise MissingGlyphError(text, font.path, missing)

    handles = ctx.font_handles(font.path)
    pixel_size = max(1, round(font_size_pt * dpi / 72.0))
    handles.ft_face.set_pixel_sizes(pixel_size, pixel_size)

    info_glyphs, positions = _shape(handles.hb_face, text, pixel_size, font.features)
    if not info_glyphs:
        raise RenderError(f"shaping returned no glyphs for {text!r}")

    bitmaps = _rasterize_glyphs(handles.ft_face, info_glyphs)
    image, sample_bbox, runs, word_boxes = _composite_line(
        bitmaps=bitmaps,
        glyphs=info_glyphs,
        positions=positions,
        text=text,
        ink=ink,
        bg=bg,
        padding=padding,
    )

    return RenderedSample(
        text=text,
        image=image,
        bbox=sample_bbox,
        font_path=font.path,
        font_size_pt=font_size_pt,
        dpi=dpi,
        ink_color=ink,
        background_color=bg,
        glyph_runs=tuple(runs),
        word_boxes=tuple(word_boxes),
    )


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def _composite_line(
    *,
    bitmaps,
    glyphs,
    positions,
    text: str,
    ink: tuple[int, int, int],
    bg: tuple[int, int, int],
    padding: int,
) -> tuple[Image.Image, tuple[int, int, int, int], list[GlyphRun], list[WordBox]]:
    """Composite shaped glyphs onto a canvas + recover per-word boxes.

    Symmetric with :func:`pd_ocr_synth.render.word_crop._composite`,
    plus a second pass that groups glyph placements into words by
    matching each glyph's ``cluster`` (input codepoint offset) to a
    span derived from whitespace boundaries in ``text``.
    """

    pen_x = 0.0
    pen_y = 0.0
    placements: list[tuple[dict, int, int, int]] = []  # (bm, x_int, y_int, cluster)

    for bm, pos, info in zip(bitmaps, positions, glyphs, strict=True):
        x_offset = pos.x_offset / 64.0
        y_offset = pos.y_offset / 64.0
        x = pen_x + x_offset + bm["left"]
        y = pen_y - y_offset - bm["top"]
        placements.append((bm, round(x), round(y), info.cluster))
        pen_x += pos.x_advance / 64.0
        pen_y += pos.y_advance / 64.0

    if not placements:
        raise RenderError("nothing to composite")

    # Tight pre-padding bbox of all inked glyphs (skip zero-area
    # bitmaps such as space, which carry advance but no ink).
    inked = [(bm, x, y, c) for bm, x, y, c in placements if bm["width"] > 0 and bm["rows"] > 0]
    if not inked:
        raise RenderError(f"line shaped to zero inked glyphs: {text!r}")

    min_x = min(x for _, x, _, _ in inked)
    min_y = min(y for _, _, y, _ in inked)
    max_x = max(x + bm["width"] for bm, x, _, _ in inked)
    max_y = max(y + bm["rows"] for bm, _, y, _ in inked)

    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)

    img_w = width + 2 * padding
    img_h = height + 2 * padding

    canvas = Image.new("RGB", (img_w, img_h), color=bg)
    runs: list[GlyphRun] = []

    # Per-glyph paste position (post-padding). We store
    # (cluster, x0, y0, x1, y1) to feed the per-word bbox grouping.
    cluster_boxes: list[tuple[int, int, int, int, int]] = []

    for bm, x, y, cluster in placements:
        if bm["width"] == 0 or bm["rows"] == 0:
            continue
        coverage = Image.frombytes("L", (bm["width"], bm["rows"]), bm["buffer"])
        ink_swatch = Image.new("RGB", coverage.size, color=ink)
        paste_x = x - min_x + padding
        paste_y = y - min_y + padding
        canvas.paste(ink_swatch, (paste_x, paste_y), mask=coverage)
        x0, y0 = paste_x, paste_y
        x1, y1 = paste_x + bm["width"], paste_y + bm["rows"]
        runs.append(GlyphRun(cluster=cluster, bbox=(x0, y0, x1, y1)))
        cluster_boxes.append((cluster, x0, y0, x1, y1))

    sample_bbox = (padding, padding, padding + width, padding + height)
    word_boxes = _group_clusters_into_words(text, cluster_boxes)

    return canvas, sample_bbox, runs, word_boxes


def _group_clusters_into_words(
    text: str, cluster_boxes: list[tuple[int, int, int, int, int]]
) -> list[WordBox]:
    """Group per-glyph placements into per-word bboxes.

    Splits ``text`` on contiguous-whitespace runs to derive word
    spans, each defined by a half-open codepoint-offset range
    ``[start, end)``. A glyph's ``cluster`` is assumed to be the
    input codepoint offset of the cluster it came from (uharfbuzz's
    default when a buffer is populated with ``add_str``).

    Glyphs whose cluster falls inside a whitespace run are dropped
    (those would be space glyphs with zero ink anyway, but defensive
    is cheap). Words that end up with zero inked glyphs are dropped
    rather than emitted with a degenerate bbox.
    """

    spans = _word_spans(text)
    out: list[WordBox] = []
    for word_text, start, end in spans:
        # Pick the inked glyph boxes that came from this word's
        # codepoint range.
        members = [(x0, y0, x1, y1) for c, x0, y0, x1, y1 in cluster_boxes if start <= c < end]
        if not members:
            continue
        x0 = min(b[0] for b in members)
        y0 = min(b[1] for b in members)
        x1 = max(b[2] for b in members)
        y1 = max(b[3] for b in members)
        out.append(WordBox(text=word_text, bbox=(x0, y0, x1, y1)))
    return out


def _word_spans(text: str) -> list[tuple[str, int, int]]:
    """Yield ``(word_text, start_offset, end_offset)`` per word in ``text``.

    A "word" is a maximal run of non-whitespace codepoints. Offsets
    are codepoint indices (0-based), end-exclusive. Punctuation
    attached to a word stays with the word — matching the
    tokenizer's ``word_crops`` behaviour: this keeps the rendered
    GT in lock-step with what the recognition model would learn
    from word-level samples.
    """

    out: list[tuple[str, int, int]] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        out.append((text[i:j], i, j))
        i = j
    return out
