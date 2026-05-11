"""Render a ``pages``-mode sample.

A ``pages`` sample is N **paragraphs** stacked vertically with consistent
font, size, ink, and background — emitted as a single image whose
ground truth carries per-paragraph, per-line, and per-word bboxes plus
the joined page text. Per spec 06:

    layout:
      mode: pages
      page_size_px: [1200, 1800]
      margins_px: { min: 60, max: 140 }
      paragraphs_per_page: { min: 3, max: 8 }
      heading_probability: 0.2
      drop_cap_probability: 0.1

This module implements the **page compositing primitive only**.
The caller is expected to hand it a pre-fitted
``Sequence[Sequence[str]]`` — outer sequence is paragraphs, inner is
the lines of each paragraph (one rendered line per element). The
wrap-fitter / paragraph-splitter that turns a free-form word stream
into this nested shape is upstream concern (a separate M09 chunk);
same for ``alignment``, ``paragraph_indent_em``, headings, and drop
caps — all of which lay on top of this primitive.

Optional ``recipe.layout.page_size_px`` produces a fixed-size canvas
by padding the natural composition with the sampled background colour
(top-left placement). Content larger than the requested page size in
either dimension raises :class:`RenderError`; we never silently
truncate annotations.

What this primitive *does* enforce:

- Font, size, dpi, ink, background, padding, and **line_spacing** are
  sampled **once** per page and reused for every paragraph. This is
  the same single-font invariant that ``render_paragraph`` enforces
  within a paragraph, lifted to the page level.
- ``layout.paragraph_spacing`` (a recipe scalar / range / weighted
  choice with units of "x the font's nominal line height") drives the
  vertical gap *between* paragraphs. Sampled once per page so the gap
  is uniform across the page. Defaults to ``1.0`` when the recipe
  field is unset — one extra line height between paragraphs, which
  matches typical body-text typography.
- Per-paragraph composition delegates to :func:`render_paragraph` via
  a pre-sampled :class:`ParagraphStyle` with ``padding_px=0``, so
  paragraphs are rendered tight (page padding wraps the whole page
  exactly once, not once per paragraph).
- Per-paragraph, per-line, and per-word bboxes are shifted into the
  page canvas's coordinate frame after stacking. This mirrors the
  pattern :func:`render_paragraph` uses to shift per-line bboxes into
  the paragraph frame.

The returned :class:`RenderedSample` has:

- ``text`` = ``"\\n\\n".join(paragraph_text)`` — the page as a single
  string with a blank line between paragraphs (matching the
  :func:`render_paragraph` ``text`` convention extended to the page
  level).
- ``image`` = a single tight-cropped page image with per-side padding
  sampled from ``layout.padding_px``.
- ``glyph_runs`` = per-cluster bboxes from every paragraph,
  concatenated in reading order, all shifted into the page frame.
- ``word_boxes`` = per-word text + tight bbox, in reading order
  (paragraph 0 line 0 left-to-right, then paragraph 0 line 1, ...,
  then paragraph 1 line 0, ...).
- ``line_boxes`` = per-line text + tight bbox, in input order across
  all paragraphs (flattened paragraph-by-paragraph).
- ``paragraph_boxes`` = per-paragraph text + tight bbox, one entry
  per input paragraph, in input order.

Raises :class:`MissingGlyphError` if the chosen font lacks any non-
whitespace codepoint across the union of all input paragraphs (a font
that misses a codepoint on paragraph 3 is just as unusable as one
that misses on paragraph 0; we'd rather fail fast than render half a
page).

Raises :class:`RenderError` for empty paragraph list, empty inner
paragraphs, or any line that violates :func:`render_paragraph`'s line-
shape contract (empty / whitespace-only / embedded newline).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

from pd_ocr_synth.render.context import RenderContext
from pd_ocr_synth.render.paragraph import (
    ParagraphStyle,
    render_paragraph,
)
from pd_ocr_synth.render.sample import (
    GlyphRun,
    LineBox,
    ParagraphBox,
    RenderedSample,
    WordBox,
)
from pd_ocr_synth.render.sampling import sample_value
from pd_ocr_synth.render.word_crop import (
    MissingGlyphError,
    RenderError,
    _missing_codepoints,
)

if TYPE_CHECKING:
    from pd_ocr_synth.recipe import Recipe


# Default multiplier when ``layout.paragraph_spacing`` is not set on
# the recipe. ``1.0`` means "one extra nominal line height of gap
# between paragraphs", which matches typical body-text typography
# (a blank line between paragraphs). The unit is "x the font's
# nominal line height", same as ``line_spacing``.
_DEFAULT_PARAGRAPH_SPACING_MULTIPLIER = 1.0


@dataclass(frozen=True, slots=True)
class PageStyle:
    """RNG-sampled style for one page render.

    Captures every random draw :func:`render_page` makes from
    ``ctx.rng`` so a caller can pre-sample once (e.g. for a future
    paragraph-splitter that needs to measure with the same metrics
    the renderer paints with) and hand the result back in via
    ``presampled`` without consuming a second helping of RNG state.

    The single-font invariant is enforced by reusing a single
    :class:`ParagraphStyle` across every paragraph on the page.
    ``paragraph_spacing_multiplier`` is the new degree of freedom
    introduced at the page level.

    Returned by :func:`sample_page_style`. When passed to
    :func:`render_page` via ``presampled``, the renderer skips its
    internal sampling block and uses these values verbatim.
    """

    paragraph_style: ParagraphStyle
    paragraph_spacing_multiplier: float


def sample_page_style(recipe: Recipe, ctx: RenderContext) -> PageStyle:
    """Pre-sample every RNG-driven knob :func:`render_page` reads.

    Consumes RNG state from ``ctx.rng`` in the **exact same order** as
    :func:`render_page`'s internal sampling, so a caller that pre-
    samples and then passes the result back in via ``presampled`` sees
    a render bit-identical to the un-pre-sampled path.

    Sampling order: paragraph style first (a delegating call to
    :func:`sample_paragraph_style`), then the paragraph-spacing
    multiplier. This ordering matters for determinism — see test
    ``test_render_page_is_deterministic_per_seed_and_index``.

    Raises :class:`RenderError` if no usable font is available, same
    as :func:`render_paragraph`.
    """

    # Avoid an import cycle at module load by importing the paragraph
    # sampler lazily.
    from pd_ocr_synth.render.paragraph import sample_paragraph_style

    paragraph_style = sample_paragraph_style(recipe, ctx)
    paragraph_spacing_mul = float(
        sample_value(
            recipe.layout.paragraph_spacing
            if recipe.layout.paragraph_spacing is not None
            else _DEFAULT_PARAGRAPH_SPACING_MULTIPLIER,
            ctx.rng,
        )
    )
    return PageStyle(
        paragraph_style=paragraph_style,
        paragraph_spacing_multiplier=paragraph_spacing_mul,
    )


def render_page(
    paragraphs: Sequence[Sequence[str]],
    *,
    recipe: Recipe,
    ctx: RenderContext,
    presampled: PageStyle | None = None,
) -> RenderedSample:
    """Render ``paragraphs`` as one ``pages``-mode sample.

    See module docstring for the full contract.

    If ``presampled`` is provided, the renderer skips its internal
    RNG-driven style sampling and reuses those values. This is the
    seam a future paragraph-splitter / wrap-fitter call site will use
    to keep its budget aligned with what the renderer paints.

    Args:
        paragraphs: outer sequence is paragraphs, inner is the pre-
            wrapped lines of each paragraph. Both sequences must be
            non-empty, and every line must satisfy
            :func:`render_paragraph`'s line contract.
        recipe: the recipe this render is for. ``recipe.layout`` is
            consulted for ``padding_px``, ``line_spacing``, and
            ``paragraph_spacing``.
        ctx: opened-font cache + per-render RNG.
        presampled: optional pre-sampled :class:`PageStyle`. When set,
            the renderer skips internal sampling.

    Returns:
        :class:`RenderedSample` with ``paragraph_boxes`` populated
        (one entry per input paragraph), ``line_boxes`` flattened
        across all paragraphs in input order, ``word_boxes`` and
        ``glyph_runs`` flattened in reading order, ``bbox`` = union
        of all paragraph bboxes.

    Raises:
        RenderError: if ``paragraphs`` is empty, any inner paragraph
            is empty, any line violates :func:`render_paragraph`'s
            line contract (delegated), no usable font is available, or
            ``recipe.layout.page_size_px`` is set and the natural-size
            content does not fit inside it.
        MissingGlyphError: if the chosen font lacks any non-
            whitespace codepoint anywhere on the page.
    """

    _validate_paragraphs(paragraphs)

    style = sample_page_style(recipe, ctx) if presampled is None else presampled

    para_style = style.paragraph_style
    padding = para_style.padding_px

    # Coverage check across the whole page in one go: a missing-glyph
    # on paragraph 3 is just as fatal as on paragraph 0, and shaping
    # half the page just to fail later is wasteful.
    joined_non_ws = "".join(
        ch for paragraph in paragraphs for line in paragraph for ch in line if not ch.isspace()
    )
    missing = _missing_codepoints(para_style.font_path, joined_non_ws)
    if missing:
        flat_text = "\n\n".join("\n".join(p) for p in paragraphs)
        raise MissingGlyphError(flat_text, para_style.font_path, missing)

    # Compute the page-level paragraph_spacing in pixels. The
    # multiplier is in units of "x nominal line height", same as
    # line_spacing. We use the paragraph style's pixel size to
    # derive line height the same way render_paragraph does.
    handles = ctx.font_handles(para_style.font_path)
    handles.ft_face.set_pixel_sizes(para_style.pixel_size, para_style.pixel_size)
    line_height_px = max(1, round(handles.ft_face.size.height / 64.0))
    paragraph_gap_px = max(0, round(line_height_px * style.paragraph_spacing_multiplier))

    # Render each paragraph with zero padding so the page wraps the
    # whole thing in padding exactly once, not once per paragraph.
    # ``presampled`` short-circuits render_paragraph's internal RNG
    # sampling, preserving determinism per page.
    #
    # ``layout.paragraph_indent_px`` (M09 first-line indent) is a
    # plain int, not RNG-sampled, so we read it directly from the
    # recipe and pass it through to every per-paragraph render. ``None``
    # → ``0`` is a no-op and preserves byte-identical output for
    # recipes that don't set the field.
    inner_para_style = _zero_padded(para_style)
    indent_px = recipe.layout.paragraph_indent_px or 0
    paragraph_samples = [
        render_paragraph(
            list(lines),
            recipe=recipe,
            ctx=ctx,
            presampled=inner_para_style,
            first_line_indent_px=indent_px,
        )
        for lines in paragraphs
    ]

    # Compose: page width = padding + max paragraph width + padding;
    # page height = padding + sum(paragraph heights) + (n-1) * gap +
    # padding.
    page_inner_w = max(s.size[0] for s in paragraph_samples)
    paragraph_heights = [s.size[1] for s in paragraph_samples]
    total_inner_h = sum(paragraph_heights) + paragraph_gap_px * (len(paragraph_samples) - 1)

    img_w = page_inner_w + 2 * padding
    img_h = total_inner_h + 2 * padding

    canvas = Image.new("RGB", (img_w, img_h), color=para_style.background_color)

    glyph_runs: list[GlyphRun] = []
    word_boxes: list[WordBox] = []
    line_boxes: list[LineBox] = []
    paragraph_boxes: list[ParagraphBox] = []

    page_min_x = img_w
    page_min_y = img_h
    page_max_x = 0
    page_max_y = 0

    cursor_y = padding
    for para_index, para_sample in enumerate(paragraph_samples):
        # Page-frame offsets for this paragraph.
        para_top = cursor_y
        para_left = padding

        canvas.paste(para_sample.image, (para_left, para_top))

        # Shift this paragraph's per-cluster boxes into page coords.
        for run in para_sample.glyph_runs:
            x0, y0, x1, y1 = run.bbox
            glyph_runs.append(
                GlyphRun(
                    cluster=run.cluster,
                    bbox=(x0 + para_left, y0 + para_top, x1 + para_left, y1 + para_top),
                )
            )

        for wb in para_sample.word_boxes:
            x0, y0, x1, y1 = wb.bbox
            word_boxes.append(
                WordBox(
                    text=wb.text,
                    bbox=(x0 + para_left, y0 + para_top, x1 + para_left, y1 + para_top),
                )
            )

        for lb in para_sample.line_boxes:
            x0, y0, x1, y1 = lb.bbox
            line_boxes.append(
                LineBox(
                    text=lb.text,
                    bbox=(x0 + para_left, y0 + para_top, x1 + para_left, y1 + para_top),
                )
            )

        # Paragraph-level bbox: shift the inner paragraph's own bbox
        # (which is already a tight inked union over its lines) into
        # page coords.
        px0, py0, px1, py1 = para_sample.bbox
        shifted_bbox = (px0 + para_left, py0 + para_top, px1 + para_left, py1 + para_top)
        paragraph_boxes.append(ParagraphBox(text=para_sample.text, bbox=shifted_bbox))

        page_min_x = min(page_min_x, shifted_bbox[0])
        page_min_y = min(page_min_y, shifted_bbox[1])
        page_max_x = max(page_max_x, shifted_bbox[2])
        page_max_y = max(page_max_y, shifted_bbox[3])

        # Advance cursor: paragraph height + inter-paragraph gap
        # (omitted after the last paragraph).
        cursor_y += paragraph_heights[para_index]
        if para_index != len(paragraph_samples) - 1:
            cursor_y += paragraph_gap_px

    # Page text: paragraphs separated by a blank line. Each
    # paragraph's own ``text`` already joins its lines with ``\n``.
    page_text = "\n\n".join(s.text for s in paragraph_samples)

    # Optional fixed-size canvas: pad the natural-size canvas to
    # ``recipe.layout.page_size_px``. Content stays at top-left; the
    # remainder is filled with the sampled background colour. Bbox
    # annotations are not shifted (offset = (0, 0)). If natural content
    # is larger than the requested canvas in either dimension, raise —
    # silent truncation would corrupt detection annotations.
    page_size_px = recipe.layout.page_size_px
    if page_size_px is not None:
        target_w, target_h = page_size_px
        if img_w > target_w or img_h > target_h:
            raise RenderError(
                f"render_page: natural content size ({img_w}x{img_h}) exceeds "
                f"requested page_size_px ({target_w}x{target_h}). "
                "Increase page_size_px, reduce content, or shrink padding."
            )
        if (img_w, img_h) != (target_w, target_h):
            padded = Image.new("RGB", (target_w, target_h), color=para_style.background_color)
            padded.paste(canvas, (0, 0))
            canvas = padded
            img_w, img_h = target_w, target_h
        # img_w / img_h are otherwise unchanged when content already
        # exactly fills the requested size.

    return RenderedSample(
        text=page_text,
        image=canvas,
        bbox=(page_min_x, page_min_y, page_max_x, page_max_y),
        font_path=para_style.font_path,
        font_size_pt=para_style.font_size_pt,
        dpi=para_style.dpi,
        ink_color=para_style.ink_color,
        background_color=para_style.background_color,
        glyph_runs=tuple(glyph_runs),
        word_boxes=tuple(word_boxes),
        line_boxes=tuple(line_boxes),
        paragraph_boxes=tuple(paragraph_boxes),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_padded(style: ParagraphStyle) -> ParagraphStyle:
    """Return a copy of ``style`` with ``padding_px = 0``.

    Inner paragraphs on a page render tight; the page wraps the whole
    composition in padding exactly once. Without this, every paragraph
    would carry its own padding and the inter-paragraph gap would be
    "padding + paragraph_spacing + padding", which is not what the
    spec means.
    """
    return ParagraphStyle(
        font_path=style.font_path,
        font_features=style.font_features,
        font_size_pt=style.font_size_pt,
        dpi=style.dpi,
        ink_color=style.ink_color,
        background_color=style.background_color,
        padding_px=0,
        spacing_multiplier=style.spacing_multiplier,
        pixel_size=style.pixel_size,
    )


def _validate_paragraphs(paragraphs: Sequence[Sequence[str]]) -> None:
    """Validate the outer + inner shape of the ``paragraphs`` argument.

    Each inner paragraph is delegated to :func:`render_paragraph`'s
    own validation, so we only check the page-level constraints
    here: outer non-empty + every inner non-empty. (An empty inner
    list would surface as ``RenderError`` from ``render_paragraph``
    too, but raising at the page level gives a more useful error
    message — "paragraph 2 is empty" beats "render_paragraph
    requires at least one line".)
    """
    if not paragraphs:
        raise RenderError("render_page requires at least one paragraph")
    for index, paragraph in enumerate(paragraphs):
        if not paragraph:
            raise RenderError(f"render_page: paragraph {index} is empty")


__all__ = [
    "PageStyle",
    "render_page",
    "sample_page_style",
]
