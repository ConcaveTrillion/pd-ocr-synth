"""Render a ``paragraphs``-mode sample.

A ``paragraphs`` sample is N lines stacked vertically with consistent
font, size, ink, and background — emitted as a single image whose
ground truth carries per-line and per-word bboxes plus the joined
paragraph text. Per spec 06:

    layout:
      mode: paragraphs
      max_width_px: 800
      max_lines: 8
      line_spacing: { min: 1.1, max: 1.5 }
      alignment: justify
      paragraph_indent_em: 0

This module implements the **paragraph compositing primitive only**.
The caller is expected to hand it a pre-fitted ``list[str]`` of lines
(one rendered line per element). The wrap-fitter that turns a free-
form word stream into a fitted line list lands in a separate M09
chunk; same for ``alignment`` and ``paragraph_indent_em``, which
require the wrap-fitter to be meaningful.

What this primitive *does* enforce:

- Font, size, dpi, ink, and background are sampled **once** per
  paragraph and reused for every line. (Mixing fonts mid-paragraph
  would be a deliberate stylistic choice, not the default; if a
  recipe ever wants it, it's a follow-up.)
- ``layout.line_spacing`` (a recipe scalar / range / weighted choice
  with a unit of "× the font's nominal line height") drives the
  vertical advance between baselines. Sampled once per paragraph so
  spacing is uniform within the sample.
- Per-line bboxes are computed from the inked region of each line,
  shifted into the paragraph canvas's coordinate frame.
- Per-word bboxes are computed per line via the same cluster→word
  mapping as ``render_line``, then shifted into the paragraph frame
  and concatenated in reading order.

The returned :class:`RenderedSample` has:

- ``text`` = ``"\\n".join(lines)`` — the paragraph as a single string,
  with each input line on its own line. Newlines in input lines are
  rejected (they'd confuse the cluster→word mapping).
- ``image`` = a single tight-cropped paragraph image with per-side
  padding sampled from ``layout.padding_px``.
- ``glyph_runs`` = per-cluster bboxes from every line, concatenated
  in reading order, all shifted into the paragraph frame.
- ``word_boxes`` = per-word text + tight bbox, in reading order
  (line 0 left-to-right, then line 1, etc.).
- ``line_boxes`` = per-line text + tight bbox, in input order.

Raises :class:`MissingGlyphError` if the chosen font lacks any
non-whitespace codepoint across the union of input lines (line-by-
line raise would be wasteful: skipping a font for one paragraph is
a paragraph-wide decision).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from pd_ocr_synth.render.context import RenderContext
from pd_ocr_synth.render.line import _group_clusters_into_words
from pd_ocr_synth.render.sample import (
    GlyphRun,
    LineBox,
    ParagraphBox,
    RenderedSample,
    WordBox,
)
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


# Default multiplier when ``layout.line_spacing`` is not set on the
# recipe. ``1.2`` is the historical "sensible default" for body text
# and matches what most word processors apply when "Single" line
# spacing is selected. The unit is "× the font's nominal line height"
# (face.height in font-units, scaled to pixels), so the same value
# behaves reasonably across font sizes.
_DEFAULT_LINE_SPACING_MULTIPLIER = 1.2


@dataclass(frozen=True, slots=True)
class ParagraphStyle:
    """RNG-sampled style for one paragraph render.

    Captures every random draw ``render_paragraph`` would make from
    its ``ctx.rng`` so a caller can pre-sample once, run the wrap-
    fitter against the same font + pixel size, and hand the result
    back into ``render_paragraph`` without consuming a second helping
    of RNG state. This is the seam the M09 wrap-fitter uses to make
    the wrap budget match what the renderer actually paints.

    Returned by :func:`sample_paragraph_style`. When passed to
    :func:`render_paragraph` via ``presampled``, the renderer skips
    its internal sampling block and uses these values verbatim.
    """

    font_path: Path
    font_features: dict | None
    font_size_pt: float
    dpi: int
    ink_color: tuple[int, int, int]
    background_color: tuple[int, int, int]
    padding_px: int
    spacing_multiplier: float
    pixel_size: int


def sample_paragraph_style(recipe: Recipe, ctx: RenderContext) -> ParagraphStyle:
    """Pre-sample every RNG-driven knob :func:`render_paragraph` reads.

    Consumes RNG state from ``ctx.rng`` in the **exact same order** as
    :func:`render_paragraph`'s internal sampling, so a caller that
    pre-samples and then passes the result back in via ``presampled``
    sees a render bit-identical to the un-pre-sampled path.

    Raises :class:`RenderError` if no usable font is available, same
    as :func:`render_paragraph`.
    """

    font = _pick_font(recipe, ctx.rng)
    font_size_pt = float(sample_value(recipe.rendering.font_size_pt, ctx.rng))
    dpi = int(sample_value(recipe.rendering.dpi, ctx.rng))
    ink = sample_color(recipe.rendering.ink_color, ctx.rng)
    bg = sample_color(recipe.rendering.background_color, ctx.rng)
    padding = int(sample_value(recipe.layout.padding_px or 0, ctx.rng))
    spacing_mul = float(
        sample_value(
            recipe.layout.line_spacing
            if recipe.layout.line_spacing is not None
            else _DEFAULT_LINE_SPACING_MULTIPLIER,
            ctx.rng,
        )
    )
    pixel_size = max(1, int(round(font_size_pt * dpi / 72.0)))
    return ParagraphStyle(
        font_path=font.path,
        font_features=font.features,
        font_size_pt=font_size_pt,
        dpi=dpi,
        ink_color=ink,
        background_color=bg,
        padding_px=padding,
        spacing_multiplier=spacing_mul,
        pixel_size=pixel_size,
    )


def render_paragraph(
    lines: list[str],
    *,
    recipe: Recipe,
    ctx: RenderContext,
    presampled: ParagraphStyle | None = None,
    first_line_indent_px: int = 0,
) -> RenderedSample:
    """Render ``lines`` as one ``paragraphs``-mode sample.

    See module docstring for the full contract.

    If ``presampled`` is provided, the renderer skips its internal
    RNG-driven style sampling and reuses those values. This is how
    the M09 wrap-fitter call site (``run_recipe`` paragraphs dispatch)
    keeps the wrap budget aligned with what the renderer paints —
    the same font + pixel size measured by ``fit_lines`` is the one
    used here. Direct callers (tests, the preview UI) can omit the
    kwarg and get the historical behavior.

    ``first_line_indent_px`` shifts the **first** line of the
    paragraph horizontally to the right by that many pixels. Used by
    :func:`render_page` to apply the recipe's
    ``layout.paragraph_indent_px`` per-paragraph (paragraphs-mode
    callers should leave this at the default ``0`` — the recipe
    validator already warns ``layout_key_unused`` if the field is set
    on a non-pages mode). The first line's image strip, per-cluster
    boxes, per-word boxes, and per-line bbox all shift by
    ``first_line_indent_px``; the canvas width grows to fit. Other
    lines are unaffected. ``0`` is a no-op (bit-identical to the
    historical render).

    Raises:
        RenderError: if ``lines`` is empty, or any line is empty /
            whitespace-only / contains an embedded newline, or
            shaping returns zero glyphs for any line, or no usable
            font is available, or ``first_line_indent_px`` is
            negative.
        MissingGlyphError: if the chosen font lacks any non-whitespace
            codepoint anywhere in the paragraph.
    """

    if first_line_indent_px < 0:
        raise RenderError(
            f"render_paragraph: first_line_indent_px must be >= 0, got {first_line_indent_px}"
        )
    _validate_lines(lines)

    if presampled is None:
        style = sample_paragraph_style(recipe, ctx)
    else:
        style = presampled

    font_path = style.font_path
    font_features = style.font_features
    font_size_pt = style.font_size_pt
    dpi = style.dpi
    ink = style.ink_color
    bg = style.background_color
    padding = style.padding_px
    spacing_mul = style.spacing_multiplier
    pixel_size = style.pixel_size

    # Coverage check across the whole paragraph: a font that misses a
    # codepoint on line 3 is just as unusable as one that misses on
    # line 0, and we'd rather fail fast than render half a paragraph.
    joined_non_ws = "".join(ch for line in lines for ch in line if not ch.isspace())
    missing = _missing_codepoints(font_path, joined_non_ws)
    if missing:
        raise MissingGlyphError("\n".join(lines), font_path, missing)

    handles = ctx.font_handles(font_path)
    handles.ft_face.set_pixel_sizes(pixel_size, pixel_size)

    # The face's nominal line height in pixels at this size — the
    # natural unit for ``line_spacing``. We pull it from the freetype
    # face's size metrics (``height`` is in 26.6 fixed-point pixels
    # after ``set_pixel_sizes``).
    line_height_px = max(1, int(round(handles.ft_face.size.height / 64.0)))
    line_advance_px = max(1, int(round(line_height_px * spacing_mul)))

    # Per-line shaped + composited fragments (no padding around each).
    fragments = [
        _shape_and_composite_line(
            text=line,
            handles=handles,
            pixel_size=pixel_size,
            features=font_features,
            ink=ink,
            bg=bg,
        )
        for line in lines
    ]

    # Stack fragments on a single canvas. The first line's baseline
    # sits at ``padding + line_height_px`` (so a baseline anchor never
    # underflows the canvas); subsequent lines advance by
    # ``line_advance_px``. Each fragment carries its own
    # baseline-relative inked extent (see ``_LineFragment``).
    # When ``first_line_indent_px`` is non-zero, line 0 is shifted
    # right by that many pixels, so the canvas width must accommodate
    # ``frag[0].width + indent`` in addition to the natural maxes of
    # the other lines.
    paragraph_width = max(
        (frag.width + (first_line_indent_px if i == 0 else 0)) for i, frag in enumerate(fragments)
    )
    img_w = paragraph_width + 2 * padding
    # Vertical bounds: first line top = padding; last line bottom =
    # padding + (n-1)*line_advance_px + last fragment height.
    last_top = (len(fragments) - 1) * line_advance_px
    img_h = padding + last_top + fragments[-1].height + padding

    canvas = Image.new("RGB", (img_w, img_h), color=bg)

    glyph_runs: list[GlyphRun] = []
    word_boxes: list[WordBox] = []
    line_boxes: list[LineBox] = []

    sample_min_x = img_w
    sample_min_y = img_h
    sample_max_x = 0
    sample_max_y = 0

    for line_index, frag in enumerate(fragments):
        line_top_in_canvas = padding + line_index * line_advance_px
        # Indent only the first line — every subsequent line starts
        # at the natural left edge.
        indent = first_line_indent_px if line_index == 0 else 0
        line_left_in_canvas = padding + indent

        canvas.paste(frag.image, (line_left_in_canvas, line_top_in_canvas))

        # Shift this line's per-cluster boxes into paragraph coords.
        for run in frag.runs:
            x0, y0, x1, y1 = run.bbox
            glyph_runs.append(
                GlyphRun(
                    cluster=run.cluster,
                    bbox=(
                        x0 + line_left_in_canvas,
                        y0 + line_top_in_canvas,
                        x1 + line_left_in_canvas,
                        y1 + line_top_in_canvas,
                    ),
                )
            )

        for wb in frag.word_boxes:
            x0, y0, x1, y1 = wb.bbox
            word_boxes.append(
                WordBox(
                    text=wb.text,
                    bbox=(
                        x0 + line_left_in_canvas,
                        y0 + line_top_in_canvas,
                        x1 + line_left_in_canvas,
                        y1 + line_top_in_canvas,
                    ),
                )
            )

        # Per-line bbox: tight inked bbox of *this* line shifted into
        # paragraph coords. Use the fragment's own inked extent (not
        # its full image, which is a tight crop already, but the
        # fragment's image carries no padding so the two coincide).
        lx0 = line_left_in_canvas + frag.inked_x0
        ly0 = line_top_in_canvas + frag.inked_y0
        lx1 = line_left_in_canvas + frag.inked_x1
        ly1 = line_top_in_canvas + frag.inked_y1
        line_boxes.append(LineBox(text=lines[line_index], bbox=(lx0, ly0, lx1, ly1)))

        sample_min_x = min(sample_min_x, lx0)
        sample_min_y = min(sample_min_y, ly0)
        sample_max_x = max(sample_max_x, lx1)
        sample_max_y = max(sample_max_y, ly1)

    # ``text`` joins lines with a literal newline. Recognition writers
    # may opt into a different join (e.g. " " for single-line
    # transcripts) but the canonical paragraph payload preserves the
    # line structure.
    paragraph_text = "\n".join(lines)

    # ``paragraph_boxes`` carries one entry — this paragraph's tight
    # inked union — so a downstream consumer (the M09 ``pages``
    # renderer, the detection writer's per-paragraph polygons) can
    # treat single-paragraph and multi-paragraph samples uniformly.
    # The bbox is exactly the union of ``line_boxes``, which by
    # construction equals the sample's own bbox here.
    paragraph_boxes = (
        ParagraphBox(
            text=paragraph_text,
            bbox=(sample_min_x, sample_min_y, sample_max_x, sample_max_y),
        ),
    )

    return RenderedSample(
        text=paragraph_text,
        image=canvas,
        bbox=(sample_min_x, sample_min_y, sample_max_x, sample_max_y),
        font_path=font_path,
        font_size_pt=font_size_pt,
        dpi=dpi,
        ink_color=ink,
        background_color=bg,
        glyph_runs=tuple(glyph_runs),
        word_boxes=tuple(word_boxes),
        line_boxes=tuple(line_boxes),
        paragraph_boxes=paragraph_boxes,
    )


# ---------------------------------------------------------------------------
# Per-line shaping + compositing into a tight, padding-free fragment image
# ---------------------------------------------------------------------------


class _LineFragment:
    """Internal: the inked bitmap + per-cluster + per-word boxes for one line.

    All bboxes are in the **fragment image's** own pixel coords (top-
    left = (0, 0)). The caller (``render_paragraph``) shifts them into
    the paragraph canvas's frame.

    ``inked_x0..inked_y1`` is the fragment's own tight inked bbox —
    which, for a padding-free fragment, is just (0, 0, width, height)
    by construction. We store it explicitly anyway so a later change
    that introduces per-line padding (e.g. for alignment slack) keeps
    the line-bbox calculation honest.
    """

    __slots__ = (
        "image",
        "width",
        "height",
        "inked_x0",
        "inked_y0",
        "inked_x1",
        "inked_y1",
        "runs",
        "word_boxes",
    )

    def __init__(
        self,
        image: Image.Image,
        runs: list[GlyphRun],
        word_boxes: list[WordBox],
        inked_bbox: tuple[int, int, int, int],
    ) -> None:
        self.image = image
        self.width = image.size[0]
        self.height = image.size[1]
        self.runs = runs
        self.word_boxes = word_boxes
        self.inked_x0, self.inked_y0, self.inked_x1, self.inked_y1 = inked_bbox


def _shape_and_composite_line(
    *,
    text: str,
    handles,
    pixel_size: int,
    features,
    ink: tuple[int, int, int],
    bg: tuple[int, int, int],
) -> _LineFragment:
    """Shape, rasterize, and composite one line into a tight fragment.

    Reuses the M09 ``render_line`` shaping + glyph rasterization path,
    but emits a padding-free fragment so the paragraph compositor can
    decide its own per-line offsets.
    """

    info_glyphs, positions = _shape(handles.hb_face, text, pixel_size, features)
    if not info_glyphs:
        raise RenderError(f"shaping returned no glyphs for {text!r}")

    bitmaps = _rasterize_glyphs(handles.ft_face, info_glyphs)

    # Place each glyph relative to a (0, 0) pen origin.
    pen_x = 0.0
    pen_y = 0.0
    placements: list[tuple[dict, int, int, int]] = []
    for bm, pos, info in zip(bitmaps, positions, info_glyphs, strict=True):
        x_offset = pos.x_offset / 64.0
        y_offset = pos.y_offset / 64.0
        x = pen_x + x_offset + bm["left"]
        y = pen_y - y_offset - bm["top"]
        placements.append((bm, int(round(x)), int(round(y)), info.cluster))
        pen_x += pos.x_advance / 64.0
        pen_y += pos.y_advance / 64.0

    inked = [(bm, x, y, c) for bm, x, y, c in placements if bm["width"] > 0 and bm["rows"] > 0]
    if not inked:
        raise RenderError(f"line shaped to zero inked glyphs: {text!r}")

    min_x = min(x for _, x, _, _ in inked)
    min_y = min(y for _, _, y, _ in inked)
    max_x = max(x + bm["width"] for bm, x, _, _ in inked)
    max_y = max(y + bm["rows"] for bm, _, y, _ in inked)

    width = max(1, max_x - min_x)
    height = max(1, max_y - min_y)

    canvas = Image.new("RGB", (width, height), color=bg)
    runs: list[GlyphRun] = []
    cluster_boxes: list[tuple[int, int, int, int, int]] = []

    for bm, x, y, cluster in placements:
        if bm["width"] == 0 or bm["rows"] == 0:
            continue
        coverage = Image.frombytes("L", (bm["width"], bm["rows"]), bm["buffer"])
        ink_swatch = Image.new("RGB", coverage.size, color=ink)
        paste_x = x - min_x
        paste_y = y - min_y
        canvas.paste(ink_swatch, (paste_x, paste_y), mask=coverage)
        x0, y0 = paste_x, paste_y
        x1, y1 = paste_x + bm["width"], paste_y + bm["rows"]
        runs.append(GlyphRun(cluster=cluster, bbox=(x0, y0, x1, y1)))
        cluster_boxes.append((cluster, x0, y0, x1, y1))

    word_boxes = _group_clusters_into_words(text, cluster_boxes)
    return _LineFragment(
        image=canvas,
        runs=runs,
        word_boxes=word_boxes,
        inked_bbox=(0, 0, width, height),
    )


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_lines(lines: list[str]) -> None:
    if not lines:
        raise RenderError("render_paragraph requires at least one line")
    for index, line in enumerate(lines):
        if not line or not line.strip():
            raise RenderError(
                f"render_paragraph: line {index} is empty or whitespace-only: {line!r}"
            )
        if "\n" in line or "\r" in line:
            raise RenderError(
                f"render_paragraph: line {index} contains an embedded newline: {line!r}"
            )
