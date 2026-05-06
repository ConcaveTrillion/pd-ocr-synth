"""Tests for ``paragraphs``-mode rendering primitive (M09).

Mirrors the structure of ``test_render_line.py`` but exercises
:func:`pd_ocr_synth.render.render_paragraph` — multiple stacked
baselines, with per-line + per-word ground truth.

These tests skip cleanly when the bundled Bunchló GC font isn't
present (e.g. fresh checkout that hasn't run
``./scripts/fetch-fonts-gaelic.sh``).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from pd_ocr_synth.recipe import load_recipe
from pd_ocr_synth.render import (
    LineBox,
    MissingGlyphError,
    RenderContext,
    RenderError,
    WordBox,
    render_paragraph,
)

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; paragraph render tests skipped.")
    return _BUNDLED_FONT


_RECIPE_TEMPLATE = """\
schema_version: 1
name: para-smoke
seed: 42
output:
  format: pd-ocr-trainer/v1
  mode: detection
  destination: ./out
  count: 1
corpus:
  - type: local
    path: ./words.txt
fonts:
  - path: {font_path}
    weight: 1.0
rendering:
  font_size_pt: {{ min: 14, max: 22 }}
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: paragraphs
  padding_px: 6
  line_spacing: {{ min: 1.1, max: 1.4 }}
"""


def _make_recipe(tmp_path: Path) -> object:
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE_TEMPLATE.format(font_path=font), encoding="utf-8")
    (tmp_path / "words.txt").write_text("ḃeaḋ\n", encoding="utf-8")
    return load_recipe(rp)


# ---------------------------------------------------------------------------
# Smoke
# ---------------------------------------------------------------------------


def test_render_paragraph_produces_non_empty_png(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["ḃeaḋ saoġal", "agus mór"],
        recipe=recipe,
        ctx=ctx,
    )

    # Joined paragraph text preserves line structure.
    assert sample.text == "ḃeaḋ saoġal\nagus mór"
    assert sample.size[0] > 0 and sample.size[1] > 0
    # Tight inked bbox non-degenerate.
    assert sample.bbox[2] > sample.bbox[0]
    assert sample.bbox[3] > sample.bbox[1]
    assert sample.font_path == _BUNDLED_FONT
    assert sample.glyph_runs, "expected at least one glyph run"
    # PNG round-trip yields non-trivial bytes.
    buf = io.BytesIO()
    sample.image.save(buf, format="PNG")
    assert len(buf.getvalue()) > 100


# ---------------------------------------------------------------------------
# Per-line + per-word ground truth
# ---------------------------------------------------------------------------


def test_render_paragraph_emits_one_line_box_per_input_line(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["ḃeaḋ saoġal", "mór is beag", "aon dó"],
        recipe=recipe,
        ctx=ctx,
    )

    assert len(sample.line_boxes) == 3
    assert [lb.text for lb in sample.line_boxes] == [
        "ḃeaḋ saoġal",
        "mór is beag",
        "aon dó",
    ]
    for lb in sample.line_boxes:
        assert isinstance(lb, LineBox)
        x0, y0, x1, y1 = lb.bbox
        assert x1 > x0 > 0
        assert y1 > y0 > 0


def test_render_paragraph_word_boxes_cover_all_words_in_reading_order(
    tmp_path: Path,
) -> None:
    """Per-word ground truth is line 0 left-to-right, then line 1, etc."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["ḃeaḋ saoġal", "mór beag"],
        recipe=recipe,
        ctx=ctx,
    )

    assert [w.text for w in sample.word_boxes] == ["ḃeaḋ", "saoġal", "mór", "beag"]
    for wb in sample.word_boxes:
        assert isinstance(wb, WordBox)
        x0, y0, x1, y1 = wb.bbox
        assert x1 > x0 > 0
        assert y1 > y0 > 0


def test_render_paragraph_word_boxes_sit_inside_their_line_box(tmp_path: Path) -> None:
    """Each word's bbox is contained in its line's bbox (and so in the sample bbox)."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["ḃeaḋ saoġal", "mór beag"],
        recipe=recipe,
        ctx=ctx,
    )

    # Group words into lines using the line_boxes' y range. We don't
    # rely on a count-based slice (e.g. "first 2 words go on line 0")
    # because cluster→word mapping might in principle drop a word with
    # no inked glyphs; this matches words to their line by geometry.
    for line_idx, lb in enumerate(sample.line_boxes):
        lx0, ly0, lx1, ly1 = lb.bbox
        # At least one word sits inside this line.
        in_line = [wb for wb in sample.word_boxes if ly0 <= wb.bbox[1] and wb.bbox[3] <= ly1]
        assert in_line, f"line {line_idx} ({lb.text!r}) has no enclosed word boxes"
        for wb in in_line:
            wx0, wy0, wx1, wy1 = wb.bbox
            assert lx0 <= wx0 <= wx1 <= lx1, f"word {wb.text!r} x escapes line bbox"
            assert ly0 <= wy0 <= wy1 <= ly1, f"word {wb.text!r} y escapes line bbox"


def test_render_paragraph_lines_are_top_to_bottom_and_disjoint(tmp_path: Path) -> None:
    """Line bboxes don't overlap on y and run top→bottom in input order."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["alpha beta", "gamma delta", "epsilon"],
        recipe=recipe,
        ctx=ctx,
    )

    boxes = sample.line_boxes
    assert len(boxes) == 3
    for prev, curr in zip(boxes, boxes[1:], strict=False):
        # y0 strictly increasing (next line lower on canvas).
        assert prev.bbox[1] < curr.bbox[1], (
            f"line y0 not increasing: prev={prev.bbox} curr={curr.bbox}"
        )
        # No vertical overlap of inked regions: curr starts at or
        # below prev's bottom.
        assert prev.bbox[3] <= curr.bbox[1], (
            f"line bboxes overlap on y: prev={prev.bbox} curr={curr.bbox}"
        )


def test_render_paragraph_sample_bbox_is_union_of_line_bboxes(tmp_path: Path) -> None:
    """sample.bbox equals the tight union of all line_boxes."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["alpha beta", "gamma"],
        recipe=recipe,
        ctx=ctx,
    )

    union_x0 = min(lb.bbox[0] for lb in sample.line_boxes)
    union_y0 = min(lb.bbox[1] for lb in sample.line_boxes)
    union_x1 = max(lb.bbox[2] for lb in sample.line_boxes)
    union_y1 = max(lb.bbox[3] for lb in sample.line_boxes)
    assert sample.bbox == (union_x0, union_y0, union_x1, union_y1)


def test_render_paragraph_canvas_contains_every_inked_box(tmp_path: Path) -> None:
    """All bboxes (line, word, cluster) lie inside the canvas."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["aon dó", "trí ceithre"],
        recipe=recipe,
        ctx=ctx,
    )

    w, h = sample.size
    for lb in sample.line_boxes:
        x0, y0, x1, y1 = lb.bbox
        assert 0 <= x0 < x1 <= w, f"line bbox x out of canvas: {lb}"
        assert 0 <= y0 < y1 <= h, f"line bbox y out of canvas: {lb}"
    for wb in sample.word_boxes:
        x0, y0, x1, y1 = wb.bbox
        assert 0 <= x0 < x1 <= w, f"word bbox x out of canvas: {wb}"
        assert 0 <= y0 < y1 <= h, f"word bbox y out of canvas: {wb}"
    for run in sample.glyph_runs:
        x0, y0, x1, y1 = run.bbox
        assert 0 <= x0 < x1 <= w, f"glyph run x out of canvas: {run}"
        assert 0 <= y0 < y1 <= h, f"glyph run y out of canvas: {run}"


# ---------------------------------------------------------------------------
# Vertical stacking depends on line_spacing
# ---------------------------------------------------------------------------


def test_render_paragraph_taller_line_spacing_yields_taller_canvas(
    tmp_path: Path,
) -> None:
    """A bigger line_spacing multiplier stretches the canvas vertically.

    Build two recipes — one with ``line_spacing: 1.0`` and one with
    ``line_spacing: 2.0`` — and render the same lines with the same
    sample seed. The 2.0 paragraph must be strictly taller. (Width is
    line-content-driven and should be ~equal.)
    """

    font = _require_font()

    def _build(spacing: float) -> object:
        rp = tmp_path / f"recipe-{spacing}.yaml"
        words = tmp_path / f"words-{spacing}.txt"
        words.write_text("ḃeaḋ\n", encoding="utf-8")
        rp.write_text(
            "schema_version: 1\n"
            f"name: para-spacing-{spacing}\n"
            "seed: 42\n"
            "output:\n"
            "  format: pd-ocr-trainer/v1\n"
            "  mode: detection\n"
            "  destination: ./out\n"
            "  count: 1\n"
            "corpus:\n"
            f"  - type: local\n    path: {words}\n"
            "fonts:\n"
            f"  - path: {font}\n    weight: 1.0\n"
            "rendering:\n"
            "  font_size_pt: 18\n"
            "  dpi: 300\n"
            "  ink_color: { r: 10, g: 10, b: 10 }\n"
            "  background_color: { r: 240, g: 235, b: 220 }\n"
            "layout:\n"
            "  mode: paragraphs\n"
            "  padding_px: 6\n"
            f"  line_spacing: {spacing}\n",
            encoding="utf-8",
        )
        return load_recipe(rp)

    recipe_tight = _build(1.0)
    recipe_loose = _build(2.0)

    ctx_tight = RenderContext.for_seed(recipe_tight.seed)
    ctx_tight.reseed_for_sample(0)
    sample_tight = render_paragraph(["alpha", "beta", "gamma"], recipe=recipe_tight, ctx=ctx_tight)

    ctx_loose = RenderContext.for_seed(recipe_loose.seed)
    ctx_loose.reseed_for_sample(0)
    sample_loose = render_paragraph(["alpha", "beta", "gamma"], recipe=recipe_loose, ctx=ctx_loose)

    assert sample_loose.size[1] > sample_tight.size[1], (
        f"loose paragraph not taller: tight h={sample_tight.size[1]}, "
        f"loose h={sample_loose.size[1]}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_render_paragraph_is_deterministic_per_seed_and_index(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)

    def _render_at(index: int) -> bytes:
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(index)
        sample = render_paragraph(
            ["ḃeaḋ saoġal", "mór beag"],
            recipe=recipe,
            ctx=ctx,
        )
        buf = io.BytesIO()
        sample.image.save(buf, format="PNG")
        return buf.getvalue()

    assert _render_at(0) == _render_at(0)
    assert _render_at(7) == _render_at(7)
    assert _render_at(0) != _render_at(1)


# ---------------------------------------------------------------------------
# Single-font / single-size invariant within a paragraph
# ---------------------------------------------------------------------------


def test_render_paragraph_uses_single_font_size_throughout(tmp_path: Path) -> None:
    """Within one paragraph, every line shares the same font_size_pt
    (sampled once). Verifies by checking that *all* line bboxes have
    the same height (font height is the dominant component when
    glyphs share x-height; this won't be exactly equal across lines
    that have different inked-glyph maxima, so we assert the bboxes
    are within a small tolerance of each other).
    """

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    # Use lines with similar ascender/descender content so the inked
    # bbox heights are nearly equal. ``ḃeaḋ`` has a dotted-tall plus
    # descender; pair with similar shapes.
    sample = render_paragraph(
        ["ḃeaḋ", "ḋoḃ", "ċaḃ"],
        recipe=recipe,
        ctx=ctx,
    )

    heights = [lb.bbox[3] - lb.bbox[1] for lb in sample.line_boxes]
    # The single-size invariant means heights should cluster tightly.
    # Allow some slop because glyph metrics vary by character. A 50%
    # spread would indicate font-size sampling per line, which we
    # explicitly disallow.
    spread = (max(heights) - min(heights)) / max(heights)
    assert spread < 0.5, f"line heights vary too widely: {heights}"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_render_paragraph_rejects_empty_line_list(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(RenderError, match="at least one line"):
        render_paragraph([], recipe=recipe, ctx=ctx)


def test_render_paragraph_rejects_empty_string_line(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(RenderError, match="empty or whitespace-only"):
        render_paragraph(["alpha", "", "beta"], recipe=recipe, ctx=ctx)


def test_render_paragraph_rejects_whitespace_only_line(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(RenderError, match="empty or whitespace-only"):
        render_paragraph(["alpha", "   ", "beta"], recipe=recipe, ctx=ctx)


def test_render_paragraph_rejects_embedded_newline_in_line(tmp_path: Path) -> None:
    """Each list element is one line. Embedded ``\\n`` is a caller bug."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(RenderError, match="embedded newline"):
        render_paragraph(["alpha\nbeta"], recipe=recipe, ctx=ctx)


def test_render_paragraph_missing_glyph_anywhere_raises(tmp_path: Path) -> None:
    """A grinning-face emoji on line 2 of a 3-line paragraph still trips."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(MissingGlyphError) as exc_info:
        render_paragraph(["ḃeaḋ", "saoġal \U0001f600", "mór"], recipe=recipe, ctx=ctx)
    assert 0x1F600 in exc_info.value.missing


# ---------------------------------------------------------------------------
# Single-line degenerate case — paragraph with one line works
# ---------------------------------------------------------------------------


def test_render_paragraph_single_line_input_works(tmp_path: Path) -> None:
    """A one-element list is a valid (degenerate) paragraph.

    The output has one ``line_box`` and the joined paragraph text is
    just that line (no trailing newline).
    """

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(["ḃeaḋ saoġal"], recipe=recipe, ctx=ctx)

    assert sample.text == "ḃeaḋ saoġal"
    assert len(sample.line_boxes) == 1
    assert sample.line_boxes[0].text == "ḃeaḋ saoġal"
    assert len(sample.word_boxes) == 2


# ---------------------------------------------------------------------------
# First-line indent (internal kwarg used by render_page)
# ---------------------------------------------------------------------------


def test_render_paragraph_first_line_indent_zero_is_default(tmp_path: Path) -> None:
    """``first_line_indent_px=0`` matches the default (no kwarg) bit-identically."""

    recipe = _make_recipe(tmp_path)

    def _png(indent_kwarg: dict) -> bytes:
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(0)
        sample = render_paragraph(
            ["alpha beta", "gamma delta"],
            recipe=recipe,
            ctx=ctx,
            **indent_kwarg,
        )
        buf = io.BytesIO()
        sample.image.save(buf, format="PNG")
        return buf.getvalue()

    assert _png({}) == _png({"first_line_indent_px": 0})


def test_render_paragraph_first_line_indent_shifts_only_line_zero(tmp_path: Path) -> None:
    """Non-zero indent shifts the first line; subsequent lines unchanged."""

    indent = 30
    recipe = _make_recipe(tmp_path)

    ctx_no = RenderContext.for_seed(recipe.seed)
    ctx_no.reseed_for_sample(0)
    sample_no = render_paragraph(
        ["alpha beta", "gamma delta"],
        recipe=recipe,
        ctx=ctx_no,
    )

    ctx_in = RenderContext.for_seed(recipe.seed)
    ctx_in.reseed_for_sample(0)
    sample_in = render_paragraph(
        ["alpha beta", "gamma delta"],
        recipe=recipe,
        ctx=ctx_in,
        first_line_indent_px=indent,
    )

    # Line 0's bbox shifts right by exactly `indent`.
    assert sample_in.line_boxes[0].bbox[0] - sample_no.line_boxes[0].bbox[0] == indent
    # Line 1 unchanged.
    assert sample_in.line_boxes[1].bbox[0] == sample_no.line_boxes[1].bbox[0]
    # Word "alpha" (line 0) shifts; word "gamma" (line 1) doesn't.
    no_words = {wb.text: wb for wb in sample_no.word_boxes}
    in_words = {wb.text: wb for wb in sample_in.word_boxes}
    assert in_words["alpha"].bbox[0] - no_words["alpha"].bbox[0] == indent
    assert in_words["gamma"].bbox[0] == no_words["gamma"].bbox[0]


def test_render_paragraph_negative_first_line_indent_raises(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    with pytest.raises(RenderError, match="first_line_indent_px"):
        render_paragraph(
            ["alpha"],
            recipe=recipe,
            ctx=ctx,
            first_line_indent_px=-1,
        )


# ---------------------------------------------------------------------------
# Paragraph alignment (left + center)
# ---------------------------------------------------------------------------


def _make_aligned_recipe(tmp_path: Path, alignment: str | None) -> object:
    """Build a paragraphs-mode recipe with the given ``paragraph_alignment``.

    ``alignment=None`` omits the field entirely (preserving the
    historical un-aligned output bytes); a string writes
    ``paragraph_alignment: <value>`` into the layout block.
    """
    font = _require_font()
    rp = tmp_path / f"recipe-align-{alignment}.yaml"
    words = tmp_path / f"words-align-{alignment}.txt"
    words.write_text("ḃeaḋ\n", encoding="utf-8")
    align_line = "" if alignment is None else f"  paragraph_alignment: {alignment}\n"
    rp.write_text(
        "schema_version: 1\n"
        f"name: para-align-{alignment}\n"
        "seed: 42\n"
        "output:\n"
        "  format: pd-ocr-trainer/v1\n"
        "  mode: detection\n"
        "  destination: ./out\n"
        "  count: 1\n"
        "corpus:\n"
        f"  - type: local\n    path: {words}\n"
        "fonts:\n"
        f"  - path: {font}\n    weight: 1.0\n"
        "rendering:\n"
        "  font_size_pt: 18\n"
        "  dpi: 300\n"
        "  ink_color: { r: 10, g: 10, b: 10 }\n"
        "  background_color: { r: 240, g: 235, b: 220 }\n"
        "layout:\n"
        "  mode: paragraphs\n"
        "  padding_px: 6\n"
        "  line_spacing: 1.0\n"
        f"{align_line}",
        encoding="utf-8",
    )
    return load_recipe(rp)


def test_render_paragraph_alignment_none_is_bit_identical_to_left(tmp_path: Path) -> None:
    """``paragraph_alignment = None`` and ``= "left"`` produce identical PNG bytes."""

    paragraphs = ["alpha beta", "gamma delta", "x"]

    recipe_none = _make_aligned_recipe(tmp_path, None)
    recipe_left = _make_aligned_recipe(tmp_path, "left")

    def _png(recipe) -> bytes:
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(0)
        sample = render_paragraph(paragraphs, recipe=recipe, ctx=ctx)
        buf = io.BytesIO()
        sample.image.save(buf, format="PNG")
        return buf.getvalue()

    assert _png(recipe_none) == _png(recipe_left)


def test_render_paragraph_alignment_left_matches_default_layout(tmp_path: Path) -> None:
    """Left alignment leaves all bbox coordinates identical to the un-aligned default."""

    paragraphs = ["alpha beta", "gamma delta", "x"]

    recipe_none = _make_aligned_recipe(tmp_path, None)
    recipe_left = _make_aligned_recipe(tmp_path, "left")

    ctx_no = RenderContext.for_seed(recipe_none.seed)
    ctx_no.reseed_for_sample(0)
    sample_no = render_paragraph(paragraphs, recipe=recipe_none, ctx=ctx_no)

    ctx_left = RenderContext.for_seed(recipe_left.seed)
    ctx_left.reseed_for_sample(0)
    sample_left = render_paragraph(paragraphs, recipe=recipe_left, ctx=ctx_left)

    assert sample_left.size == sample_no.size
    assert sample_left.bbox == sample_no.bbox
    assert [lb.bbox for lb in sample_left.line_boxes] == [lb.bbox for lb in sample_no.line_boxes]
    assert [wb.bbox for wb in sample_left.word_boxes] == [wb.bbox for wb in sample_no.word_boxes]


def test_render_paragraph_alignment_center_centers_short_lines(tmp_path: Path) -> None:
    """``paragraph_alignment = "center"`` centers each line within ``paragraph_width``.

    For a paragraph whose lines have different widths, the centering
    offset for line *i* equals ``(paragraph_width - line_i_width) // 2``,
    where ``paragraph_width`` is the width of the longest line. The
    longest line gets offset 0; shorter lines slide right.
    """

    # First line is the long one; second line is short — so the
    # second line should be shifted right while the first line sits
    # at the natural left edge.
    paragraphs = ["alpha beta gamma delta", "x"]

    recipe_left = _make_aligned_recipe(tmp_path, "left")
    recipe_center = _make_aligned_recipe(tmp_path, "center")

    ctx_left = RenderContext.for_seed(recipe_left.seed)
    ctx_left.reseed_for_sample(0)
    sample_left = render_paragraph(paragraphs, recipe=recipe_left, ctx=ctx_left)

    ctx_center = RenderContext.for_seed(recipe_center.seed)
    ctx_center.reseed_for_sample(0)
    sample_center = render_paragraph(paragraphs, recipe=recipe_center, ctx=ctx_center)

    # Canvas size matches: alignment is purely a per-line horizontal
    # offset within the same paragraph width, no canvas growth.
    assert sample_center.size == sample_left.size

    # Long line (line 0) sits at the same left edge in both samples.
    assert sample_center.line_boxes[0].bbox[0] == sample_left.line_boxes[0].bbox[0]

    # Short line (line 1) moves right by a positive offset.
    delta = sample_center.line_boxes[1].bbox[0] - sample_left.line_boxes[1].bbox[0]
    assert delta > 0, (
        f"short line did not shift right under center alignment: "
        f"left.x0={sample_left.line_boxes[1].bbox[0]}, "
        f"center.x0={sample_center.line_boxes[1].bbox[0]}"
    )


def test_render_paragraph_alignment_center_word_boxes_track_line(tmp_path: Path) -> None:
    """Word boxes shift by the same per-line offset as their line."""

    paragraphs = ["alpha beta gamma delta", "x y"]

    recipe_left = _make_aligned_recipe(tmp_path, "left")
    recipe_center = _make_aligned_recipe(tmp_path, "center")

    ctx_left = RenderContext.for_seed(recipe_left.seed)
    ctx_left.reseed_for_sample(0)
    sample_left = render_paragraph(paragraphs, recipe=recipe_left, ctx=ctx_left)

    ctx_center = RenderContext.for_seed(recipe_center.seed)
    ctx_center.reseed_for_sample(0)
    sample_center = render_paragraph(paragraphs, recipe=recipe_center, ctx=ctx_center)

    # Match words by text + line index. ``alpha`` is on line 0
    # (longest line, zero offset); ``x`` and ``y`` are on line 1
    # (short, gets the centering offset). The line-1 words must shift
    # by the same delta as the line-1 bbox.
    line1_delta = sample_center.line_boxes[1].bbox[0] - sample_left.line_boxes[1].bbox[0]
    assert line1_delta > 0

    def _word(sample, text: str) -> WordBox:
        return next(wb for wb in sample.word_boxes if wb.text == text)

    # Line-0 words: zero offset.
    for word in ("alpha", "beta", "gamma", "delta"):
        assert _word(sample_center, word).bbox[0] == _word(sample_left, word).bbox[0], (
            f"line-0 word {word!r} shifted unexpectedly under center"
        )

    # Line-1 words: shift by the same delta as the line.
    for word in ("x", "y"):
        delta = _word(sample_center, word).bbox[0] - _word(sample_left, word).bbox[0]
        assert delta == line1_delta, (
            f"line-1 word {word!r} delta {delta} != line delta {line1_delta}"
        )


def test_render_paragraph_alignment_center_glyph_runs_track_line(tmp_path: Path) -> None:
    """Per-cluster glyph_runs shift by the same per-line offset as their line."""

    paragraphs = ["alpha beta gamma delta", "x"]

    recipe_left = _make_aligned_recipe(tmp_path, "left")
    recipe_center = _make_aligned_recipe(tmp_path, "center")

    ctx_left = RenderContext.for_seed(recipe_left.seed)
    ctx_left.reseed_for_sample(0)
    sample_left = render_paragraph(paragraphs, recipe=recipe_left, ctx=ctx_left)

    ctx_center = RenderContext.for_seed(recipe_center.seed)
    ctx_center.reseed_for_sample(0)
    sample_center = render_paragraph(paragraphs, recipe=recipe_center, ctx=ctx_center)

    line1_left = sample_left.line_boxes[1]
    line1_center = sample_center.line_boxes[1]
    line1_delta = line1_center.bbox[0] - line1_left.bbox[0]
    assert line1_delta > 0

    # Filter glyph_runs to line 1 by y-range (line 0 is fully above
    # line 1 in both renders).
    def _line1_runs(sample) -> list:
        ly0, ly1 = (
            sample.line_boxes[1].bbox[1],
            sample.line_boxes[1].bbox[3],
        )
        return sorted(
            (r for r in sample.glyph_runs if ly0 <= r.bbox[1] and r.bbox[3] <= ly1),
            key=lambda r: r.bbox[0],
        )

    runs_left = _line1_runs(sample_left)
    runs_center = _line1_runs(sample_center)
    assert len(runs_left) == len(runs_center) > 0
    for r_left, r_center in zip(runs_left, runs_center, strict=True):
        assert r_center.bbox[0] - r_left.bbox[0] == line1_delta
        assert r_center.bbox[2] - r_left.bbox[2] == line1_delta
        # y unchanged
        assert r_center.bbox[1] == r_left.bbox[1]
        assert r_center.bbox[3] == r_left.bbox[3]


def test_render_paragraph_alignment_center_longest_line_unchanged(tmp_path: Path) -> None:
    """The longest line gets offset 0 — its bbox matches the left-aligned render."""

    # Make line 1 the longest so we exercise "longest is not line 0".
    paragraphs = ["x", "alpha beta gamma delta", "y"]

    recipe_left = _make_aligned_recipe(tmp_path, "left")
    recipe_center = _make_aligned_recipe(tmp_path, "center")

    ctx_left = RenderContext.for_seed(recipe_left.seed)
    ctx_left.reseed_for_sample(0)
    sample_left = render_paragraph(paragraphs, recipe=recipe_left, ctx=ctx_left)

    ctx_center = RenderContext.for_seed(recipe_center.seed)
    ctx_center.reseed_for_sample(0)
    sample_center = render_paragraph(paragraphs, recipe=recipe_center, ctx=ctx_center)

    # Long line (line 1) bbox left edge unchanged.
    assert sample_center.line_boxes[1].bbox[0] == sample_left.line_boxes[1].bbox[0]
    # Short lines (lines 0 + 2) shift right.
    assert sample_center.line_boxes[0].bbox[0] > sample_left.line_boxes[0].bbox[0]
    assert sample_center.line_boxes[2].bbox[0] > sample_left.line_boxes[2].bbox[0]


def test_render_paragraph_alignment_unsupported_value_raises(tmp_path: Path) -> None:
    """An out-of-band alignment value (bypassing pydantic) raises RenderError.

    The recipe ``Layout`` model's
    ``Literal["left", "center", "right"]`` rejects unknown values at
    load time. This test pokes the validation layer directly by
    mutating an already-loaded recipe via ``model_copy(update=...)`` —
    the renderer's own defensive check must still fire.
    """
    recipe = _make_aligned_recipe(tmp_path, "left")
    bad_layout = recipe.layout.model_copy(update={"paragraph_alignment": "justify"})
    bad_recipe = recipe.model_copy(update={"layout": bad_layout})
    ctx = RenderContext.for_seed(bad_recipe.seed)
    ctx.reseed_for_sample(0)
    with pytest.raises(RenderError, match="paragraph_alignment"):
        render_paragraph(["alpha"], recipe=bad_recipe, ctx=ctx)


# ---------------------------------------------------------------------------
# Paragraph alignment: right
# ---------------------------------------------------------------------------


def test_render_paragraph_alignment_right_flushes_short_lines_to_right_edge(
    tmp_path: Path,
) -> None:
    """``paragraph_alignment = "right"`` flushes each line to ``paragraph_width``.

    For a paragraph whose lines have different widths, the right-
    alignment offset for line *i* equals
    ``paragraph_width - line_i_natural_width`` where
    ``paragraph_width`` is the width of the longest line. The longest
    line gets offset 0; shorter lines slide right by the full
    difference (twice as far as under ``"center"``).
    """

    paragraphs = ["alpha beta gamma delta", "x"]

    recipe_left = _make_aligned_recipe(tmp_path, "left")
    recipe_right = _make_aligned_recipe(tmp_path, "right")

    ctx_left = RenderContext.for_seed(recipe_left.seed)
    ctx_left.reseed_for_sample(0)
    sample_left = render_paragraph(paragraphs, recipe=recipe_left, ctx=ctx_left)

    ctx_right = RenderContext.for_seed(recipe_right.seed)
    ctx_right.reseed_for_sample(0)
    sample_right = render_paragraph(paragraphs, recipe=recipe_right, ctx=ctx_right)

    # Canvas size matches: alignment is purely a per-line horizontal
    # offset within the same paragraph width, no canvas growth.
    assert sample_right.size == sample_left.size

    # Long line (line 0) sits at the same left edge in both samples
    # (offset 0 because it equals paragraph_width).
    assert sample_right.line_boxes[0].bbox[0] == sample_left.line_boxes[0].bbox[0]

    # Short line (line 1) right edge is flush with the long line's
    # right edge — that's the defining property of right-alignment.
    assert sample_right.line_boxes[1].bbox[2] == sample_right.line_boxes[0].bbox[2]


def test_render_paragraph_alignment_right_offset_equals_full_difference(
    tmp_path: Path,
) -> None:
    """Right-alignment delta is exactly twice center-alignment delta for the same line.

    Under ``"center"`` a short line shifts by
    ``(paragraph_width - line_width) // 2``; under ``"right"`` it
    shifts by the full ``paragraph_width - line_width``. So
    ``right_delta`` equals ``2 * center_delta`` (modulo a one-pixel
    rounding when the difference is odd, which we avoid by picking a
    text that produces a deterministic gap).
    """

    paragraphs = ["alpha beta gamma delta", "x"]

    recipe_left = _make_aligned_recipe(tmp_path, "left")
    recipe_center = _make_aligned_recipe(tmp_path, "center")
    recipe_right = _make_aligned_recipe(tmp_path, "right")

    def _render(recipe):
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(0)
        return render_paragraph(paragraphs, recipe=recipe, ctx=ctx)

    sample_left = _render(recipe_left)
    sample_center = _render(recipe_center)
    sample_right = _render(recipe_right)

    center_delta = sample_center.line_boxes[1].bbox[0] - sample_left.line_boxes[1].bbox[0]
    right_delta = sample_right.line_boxes[1].bbox[0] - sample_left.line_boxes[1].bbox[0]
    # Right delta is the full paragraph_width - line_width gap; center
    # delta is half of it (rounded down). Allow a 1-px slack for
    # integer rounding when the gap is odd.
    assert right_delta >= 2 * center_delta
    assert right_delta <= 2 * center_delta + 1


def test_render_paragraph_alignment_right_word_boxes_track_line(tmp_path: Path) -> None:
    """Word boxes shift by the same per-line offset as their line under right-align."""

    paragraphs = ["alpha beta gamma delta", "x y"]

    recipe_left = _make_aligned_recipe(tmp_path, "left")
    recipe_right = _make_aligned_recipe(tmp_path, "right")

    ctx_left = RenderContext.for_seed(recipe_left.seed)
    ctx_left.reseed_for_sample(0)
    sample_left = render_paragraph(paragraphs, recipe=recipe_left, ctx=ctx_left)

    ctx_right = RenderContext.for_seed(recipe_right.seed)
    ctx_right.reseed_for_sample(0)
    sample_right = render_paragraph(paragraphs, recipe=recipe_right, ctx=ctx_right)

    line1_delta = sample_right.line_boxes[1].bbox[0] - sample_left.line_boxes[1].bbox[0]
    assert line1_delta > 0

    def _word(sample, text: str) -> WordBox:
        return next(wb for wb in sample.word_boxes if wb.text == text)

    # Line-0 words: zero offset (longest line).
    for word in ("alpha", "beta", "gamma", "delta"):
        assert _word(sample_right, word).bbox[0] == _word(sample_left, word).bbox[0], (
            f"line-0 word {word!r} shifted unexpectedly under right"
        )

    # Line-1 words: shift by the same delta as the line.
    for word in ("x", "y"):
        delta = _word(sample_right, word).bbox[0] - _word(sample_left, word).bbox[0]
        assert delta == line1_delta, (
            f"line-1 word {word!r} delta {delta} != line delta {line1_delta}"
        )


def test_render_paragraph_alignment_right_glyph_runs_track_line(tmp_path: Path) -> None:
    """Per-cluster glyph_runs shift by the same per-line offset as their line."""

    paragraphs = ["alpha beta gamma delta", "x"]

    recipe_left = _make_aligned_recipe(tmp_path, "left")
    recipe_right = _make_aligned_recipe(tmp_path, "right")

    ctx_left = RenderContext.for_seed(recipe_left.seed)
    ctx_left.reseed_for_sample(0)
    sample_left = render_paragraph(paragraphs, recipe=recipe_left, ctx=ctx_left)

    ctx_right = RenderContext.for_seed(recipe_right.seed)
    ctx_right.reseed_for_sample(0)
    sample_right = render_paragraph(paragraphs, recipe=recipe_right, ctx=ctx_right)

    line1_left = sample_left.line_boxes[1]
    line1_right = sample_right.line_boxes[1]
    line1_delta = line1_right.bbox[0] - line1_left.bbox[0]
    assert line1_delta > 0

    # Filter glyph_runs to line 1 by y-range (line 0 is fully above
    # line 1 in both renders).
    def _line1_runs(sample) -> list:
        ly0, ly1 = (
            sample.line_boxes[1].bbox[1],
            sample.line_boxes[1].bbox[3],
        )
        return sorted(
            (r for r in sample.glyph_runs if ly0 <= r.bbox[1] and r.bbox[3] <= ly1),
            key=lambda r: r.bbox[0],
        )

    runs_left = _line1_runs(sample_left)
    runs_right = _line1_runs(sample_right)
    assert len(runs_left) == len(runs_right) > 0
    for r_left, r_right in zip(runs_left, runs_right, strict=True):
        assert r_right.bbox[0] - r_left.bbox[0] == line1_delta
        assert r_right.bbox[2] - r_left.bbox[2] == line1_delta
        # y unchanged
        assert r_right.bbox[1] == r_left.bbox[1]
        assert r_right.bbox[3] == r_left.bbox[3]


def test_render_paragraph_alignment_right_longest_line_unchanged(tmp_path: Path) -> None:
    """The longest line gets offset 0 under right-align — bbox matches left-aligned."""

    # Make line 1 the longest so we exercise "longest is not line 0".
    paragraphs = ["x", "alpha beta gamma delta", "y"]

    recipe_left = _make_aligned_recipe(tmp_path, "left")
    recipe_right = _make_aligned_recipe(tmp_path, "right")

    ctx_left = RenderContext.for_seed(recipe_left.seed)
    ctx_left.reseed_for_sample(0)
    sample_left = render_paragraph(paragraphs, recipe=recipe_left, ctx=ctx_left)

    ctx_right = RenderContext.for_seed(recipe_right.seed)
    ctx_right.reseed_for_sample(0)
    sample_right = render_paragraph(paragraphs, recipe=recipe_right, ctx=ctx_right)

    # Long line (line 1) bbox left edge unchanged.
    assert sample_right.line_boxes[1].bbox[0] == sample_left.line_boxes[1].bbox[0]
    # Short lines (lines 0 + 2) shift right.
    assert sample_right.line_boxes[0].bbox[0] > sample_left.line_boxes[0].bbox[0]
    assert sample_right.line_boxes[2].bbox[0] > sample_left.line_boxes[2].bbox[0]
    # Both short lines flush to the long-line right edge.
    assert sample_right.line_boxes[0].bbox[2] == sample_right.line_boxes[1].bbox[2]
    assert sample_right.line_boxes[2].bbox[2] == sample_right.line_boxes[1].bbox[2]


def test_render_paragraph_alignment_right_equal_width_lines_unchanged(
    tmp_path: Path,
) -> None:
    """When all lines have equal width, right-align is a no-op vs left-align.

    With equal-width lines, ``paragraph_width - line_natural_width``
    is 0 for every line, so the offset is 0 under all three of
    ``left`` / ``center`` / ``right`` — bboxes match exactly.
    """

    paragraphs = ["alpha beta", "gamma delta"]
    # These two lines are not guaranteed equal-width — we only need a
    # case where each line equals paragraph_width to verify the
    # invariant. Use the one and only line that's by-construction
    # equal to paragraph_width: a single line.
    paragraphs = ["alpha beta gamma"]

    recipe_left = _make_aligned_recipe(tmp_path, "left")
    recipe_center = _make_aligned_recipe(tmp_path, "center")
    recipe_right = _make_aligned_recipe(tmp_path, "right")

    def _render(recipe):
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(0)
        return render_paragraph(paragraphs, recipe=recipe, ctx=ctx)

    sample_left = _render(recipe_left)
    sample_center = _render(recipe_center)
    sample_right = _render(recipe_right)

    # Every line equals paragraph_width here, so all three alignments
    # produce identical bboxes.
    assert [lb.bbox for lb in sample_left.line_boxes] == [
        lb.bbox for lb in sample_center.line_boxes
    ]
    assert [lb.bbox for lb in sample_left.line_boxes] == [lb.bbox for lb in sample_right.line_boxes]
    assert sample_left.size == sample_center.size == sample_right.size


def test_render_paragraph_alignment_right_word_boxes_stay_inside_canvas(
    tmp_path: Path,
) -> None:
    """Right-aligned word boxes still sit inside the paragraph canvas.

    A right-aligned line slides up to the right padding edge; if the
    offset arithmetic ever overshoots, ``x1`` would exceed the canvas
    width. Locks that in: every word box must fit in [0, w].
    """

    paragraphs = ["alpha beta gamma delta", "x y", "mu nu"]
    recipe = _make_aligned_recipe(tmp_path, "right")
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(paragraphs, recipe=recipe, ctx=ctx)

    w, h = sample.size
    for wb in sample.word_boxes:
        x0, y0, x1, y1 = wb.bbox
        assert 0 <= x0 < x1 <= w, f"word {wb.text!r} bbox out of canvas: {wb.bbox}, canvas={w}x{h}"
        assert 0 <= y0 < y1 <= h, f"word {wb.text!r} bbox out of canvas: {wb.bbox}, canvas={w}x{h}"
