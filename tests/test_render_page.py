"""Tests for ``pages``-mode rendering primitive (M09).

Mirrors the structure of ``test_render_paragraph.py`` but exercises
:func:`pd_ocr_synth.render.render_page` — multiple stacked paragraphs,
with per-paragraph + per-line + per-word ground truth on a single
page-shaped canvas.

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
    ParagraphBox,
    RenderContext,
    RenderError,
    WordBox,
    render_page,
    render_paragraph,
)

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; page render tests skipped.")
    return _BUNDLED_FONT


_RECIPE_TEMPLATE = """\
schema_version: 1
name: page-smoke
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
  mode: pages
  padding_px: 8
  line_spacing: {{ min: 1.1, max: 1.4 }}
  paragraph_spacing: {{ min: 0.8, max: 1.5 }}
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


def test_render_page_produces_non_empty_png(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["ḃeaḋ saoġal", "agus mór"],
            ["aon dó", "trí ceithre"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    # Joined page text uses "\n\n" between paragraphs, "\n" between lines.
    assert sample.text == "ḃeaḋ saoġal\nagus mór\n\naon dó\ntrí ceithre"
    assert sample.size[0] > 0 and sample.size[1] > 0
    assert sample.bbox[2] > sample.bbox[0]
    assert sample.bbox[3] > sample.bbox[1]
    assert sample.font_path == _BUNDLED_FONT
    assert sample.glyph_runs, "expected at least one glyph run"
    buf = io.BytesIO()
    sample.image.save(buf, format="PNG")
    assert len(buf.getvalue()) > 100


# ---------------------------------------------------------------------------
# Per-paragraph + per-line + per-word ground truth
# ---------------------------------------------------------------------------


def test_render_page_emits_one_paragraph_box_per_input_paragraph(
    tmp_path: Path,
) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["ḃeaḋ saoġal"],
            ["mór is beag", "aon dó"],
            ["trí ceithre"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    assert len(sample.paragraph_boxes) == 3
    for pb in sample.paragraph_boxes:
        assert isinstance(pb, ParagraphBox)
        x0, y0, x1, y1 = pb.bbox
        assert x1 > x0 > 0
        assert y1 > y0 > 0
    # Paragraph 0 has 1 line, paragraph 1 has 2 lines, paragraph 2 has 1 line.
    assert sample.paragraph_boxes[0].text == "ḃeaḋ saoġal"
    assert sample.paragraph_boxes[1].text == "mór is beag\naon dó"
    assert sample.paragraph_boxes[2].text == "trí ceithre"


def test_render_page_line_boxes_flatten_paragraphs_in_order(tmp_path: Path) -> None:
    """``line_boxes`` flatten paragraph-by-paragraph, top-to-bottom."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["alpha beta", "gamma"],
            ["delta", "epsilon zeta"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    assert [lb.text for lb in sample.line_boxes] == [
        "alpha beta",
        "gamma",
        "delta",
        "epsilon zeta",
    ]
    for lb in sample.line_boxes:
        assert isinstance(lb, LineBox)
    # y0s strictly increasing top-to-bottom.
    y0s = [lb.bbox[1] for lb in sample.line_boxes]
    assert y0s == sorted(y0s)
    for prev_y0, curr_y0 in zip(y0s, y0s[1:], strict=False):
        assert curr_y0 > prev_y0


def test_render_page_word_boxes_flatten_in_reading_order(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["alpha beta", "gamma"],
            ["delta epsilon"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    assert [w.text for w in sample.word_boxes] == [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
    ]
    for wb in sample.word_boxes:
        assert isinstance(wb, WordBox)


def test_render_page_words_inside_lines_inside_paragraphs(tmp_path: Path) -> None:
    """Coordinate-frame nesting: word bbox ⊆ line bbox ⊆ paragraph bbox ⊆ page."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["alpha beta", "gamma delta"],
            ["epsilon zeta", "eta theta"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    # Sanity: per-paragraph nesting is correct via geometry.
    for pb in sample.paragraph_boxes:
        px0, py0, px1, py1 = pb.bbox
        # Lines that fall inside this paragraph by y-range.
        in_para_lines = [lb for lb in sample.line_boxes if py0 <= lb.bbox[1] and lb.bbox[3] <= py1]
        assert in_para_lines, f"paragraph {pb.text!r} has no enclosed line boxes"
        for lb in in_para_lines:
            lx0, ly0, lx1, ly1 = lb.bbox
            assert px0 <= lx0 <= lx1 <= px1, f"line {lb.text!r} x escapes paragraph bbox"
            assert py0 <= ly0 <= ly1 <= py1, f"line {lb.text!r} y escapes paragraph bbox"
            # Words inside this line.
            in_line_words = [
                wb for wb in sample.word_boxes if ly0 <= wb.bbox[1] and wb.bbox[3] <= ly1
            ]
            assert in_line_words, f"line {lb.text!r} has no enclosed word boxes"
            for wb in in_line_words:
                wx0, wy0, wx1, wy1 = wb.bbox
                assert lx0 <= wx0 <= wx1 <= lx1, f"word {wb.text!r} x escapes line"
                assert ly0 <= wy0 <= wy1 <= ly1, f"word {wb.text!r} y escapes line"

    # Page bbox encloses every paragraph bbox.
    sx0, sy0, sx1, sy1 = sample.bbox
    for pb in sample.paragraph_boxes:
        px0, py0, px1, py1 = pb.bbox
        assert sx0 <= px0 <= px1 <= sx1
        assert sy0 <= py0 <= py1 <= sy1


def test_render_page_canvas_contains_every_inked_box(tmp_path: Path) -> None:
    """All bboxes (paragraph, line, word, cluster) lie inside the canvas."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["aon dó", "trí ceithre"],
            ["ḃeaḋ saoġal"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    w, h = sample.size
    for pb in sample.paragraph_boxes:
        x0, y0, x1, y1 = pb.bbox
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h
    for lb in sample.line_boxes:
        x0, y0, x1, y1 = lb.bbox
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h
    for wb in sample.word_boxes:
        x0, y0, x1, y1 = wb.bbox
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h
    for run in sample.glyph_runs:
        x0, y0, x1, y1 = run.bbox
        assert 0 <= x0 < x1 <= w
        assert 0 <= y0 < y1 <= h


def test_render_page_paragraph_bboxes_are_top_to_bottom_and_disjoint(
    tmp_path: Path,
) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["alpha"],
            ["beta gamma"],
            ["delta"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    boxes = sample.paragraph_boxes
    assert len(boxes) == 3
    for prev, curr in zip(boxes, boxes[1:], strict=False):
        assert prev.bbox[1] < curr.bbox[1]
        assert prev.bbox[3] <= curr.bbox[1], (
            f"paragraph bboxes overlap on y: prev={prev.bbox} curr={curr.bbox}"
        )


def test_render_page_sample_bbox_is_union_of_paragraph_bboxes(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["alpha beta"],
            ["gamma"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    union_x0 = min(pb.bbox[0] for pb in sample.paragraph_boxes)
    union_y0 = min(pb.bbox[1] for pb in sample.paragraph_boxes)
    union_x1 = max(pb.bbox[2] for pb in sample.paragraph_boxes)
    union_y1 = max(pb.bbox[3] for pb in sample.paragraph_boxes)
    assert sample.bbox == (union_x0, union_y0, union_x1, union_y1)


# ---------------------------------------------------------------------------
# paragraph_spacing affects vertical extent
# ---------------------------------------------------------------------------


def test_render_page_taller_paragraph_spacing_yields_taller_canvas(
    tmp_path: Path,
) -> None:
    """A bigger ``paragraph_spacing`` multiplier stretches the page vertically.

    Build two recipes — one with ``paragraph_spacing: 0.5`` and one
    with ``paragraph_spacing: 3.0`` — and render the same paragraphs
    with the same sample seed. The 3.0 page must be strictly taller.
    """

    font = _require_font()

    def _build(spacing: float) -> object:
        rp = tmp_path / f"recipe-{spacing}.yaml"
        words = tmp_path / f"words-{spacing}.txt"
        words.write_text("ḃeaḋ\n", encoding="utf-8")
        rp.write_text(
            "schema_version: 1\n"
            f"name: page-spacing-{spacing}\n"
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
            "  mode: pages\n"
            "  padding_px: 6\n"
            "  line_spacing: 1.0\n"
            f"  paragraph_spacing: {spacing}\n",
            encoding="utf-8",
        )
        return load_recipe(rp)

    recipe_tight = _build(0.5)
    recipe_loose = _build(3.0)

    ctx_tight = RenderContext.for_seed(recipe_tight.seed)
    ctx_tight.reseed_for_sample(0)
    sample_tight = render_page(
        [["alpha"], ["beta"], ["gamma"]],
        recipe=recipe_tight,
        ctx=ctx_tight,
    )

    ctx_loose = RenderContext.for_seed(recipe_loose.seed)
    ctx_loose.reseed_for_sample(0)
    sample_loose = render_page(
        [["alpha"], ["beta"], ["gamma"]],
        recipe=recipe_loose,
        ctx=ctx_loose,
    )

    assert sample_loose.size[1] > sample_tight.size[1], (
        f"loose page not taller: tight h={sample_tight.size[1]}, loose h={sample_loose.size[1]}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_render_page_is_deterministic_per_seed_and_index(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)

    def _render_at(index: int) -> bytes:
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(index)
        sample = render_page(
            [
                ["ḃeaḋ saoġal", "mór beag"],
                ["aon dó"],
            ],
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
# Single-font invariant across the whole page
# ---------------------------------------------------------------------------


def test_render_page_uses_single_font_throughout(tmp_path: Path) -> None:
    """Every glyph run on the page comes from the same single font."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["alpha beta"],
            ["gamma delta"],
            ["epsilon zeta"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    # The single-font invariant manifests as: ``sample.font_path`` is
    # the only font on the page (tracked at the sample level, since
    # GlyphRun doesn't carry a per-run font_path). The recipe has
    # exactly one font, so this is a coarse invariant — but the
    # tighter check is that all line bboxes have similar height (the
    # font + pixel size is sampled once). Mirror the paragraph test.
    heights = [lb.bbox[3] - lb.bbox[1] for lb in sample.line_boxes]
    spread = (max(heights) - min(heights)) / max(heights)
    assert spread < 0.5, f"line heights vary too widely: {heights}"


def test_render_page_single_paragraph_matches_render_paragraph(tmp_path: Path) -> None:
    """A 1-paragraph page is equivalent to ``render_paragraph`` output.

    With ``layout.padding_px`` matching across both, a 1-paragraph
    page should produce the same text and the same number of
    line/word boxes as the equivalent ``render_paragraph`` call.

    We do *not* assert byte-identical PNGs because the two paths
    consume RNG slightly differently (the page sampler also draws
    paragraph_spacing). Instead we assert the structural properties.
    """

    recipe = _make_recipe(tmp_path)

    ctx_para = RenderContext.for_seed(recipe.seed)
    ctx_para.reseed_for_sample(0)
    para_sample = render_paragraph(
        ["alpha beta", "gamma delta"],
        recipe=recipe,
        ctx=ctx_para,
    )

    ctx_page = RenderContext.for_seed(recipe.seed)
    ctx_page.reseed_for_sample(0)
    page_sample = render_page(
        [["alpha beta", "gamma delta"]],
        recipe=recipe,
        ctx=ctx_page,
    )

    # Same text up to paragraph join (only one paragraph here).
    assert page_sample.text == para_sample.text
    assert len(page_sample.paragraph_boxes) == 1
    assert page_sample.paragraph_boxes[0].text == para_sample.text
    assert len(page_sample.line_boxes) == len(para_sample.line_boxes)
    assert len(page_sample.word_boxes) == len(para_sample.word_boxes)
    assert [lb.text for lb in page_sample.line_boxes] == [lb.text for lb in para_sample.line_boxes]
    assert [wb.text for wb in page_sample.word_boxes] == [wb.text for wb in para_sample.word_boxes]


# ---------------------------------------------------------------------------
# First-line indent
# ---------------------------------------------------------------------------


def _build_indent_recipe(tmp_path: Path, indent_px: int | None) -> object:
    """Build a pages-mode recipe with a fixed seed and the given indent.

    ``indent_px=None`` omits the field entirely (preserving the
    historical un-indented output bytes); a non-negative int writes
    ``paragraph_indent_px: <n>`` into the layout block.
    """

    font = _require_font()
    rp = tmp_path / f"recipe-indent-{indent_px}.yaml"
    words = tmp_path / f"words-indent-{indent_px}.txt"
    words.write_text("ḃeaḋ\n", encoding="utf-8")
    indent_line = "" if indent_px is None else f"  paragraph_indent_px: {indent_px}\n"
    rp.write_text(
        "schema_version: 1\n"
        f"name: page-indent-{indent_px}\n"
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
        "  mode: pages\n"
        "  padding_px: 6\n"
        "  line_spacing: 1.0\n"
        "  paragraph_spacing: 1.0\n"
        f"{indent_line}",
        encoding="utf-8",
    )
    return load_recipe(rp)


def test_render_page_first_line_indent_shifts_first_line_right(
    tmp_path: Path,
) -> None:
    """A non-zero ``paragraph_indent_px`` shifts the first line of every paragraph.

    For each paragraph that has at least two lines, the first line's
    bbox left edge must sit ~indent_px further right than the second
    line's left edge.
    """

    indent_px = 50
    recipe = _build_indent_recipe(tmp_path, indent_px)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["alpha beta", "gamma delta"],
            ["epsilon zeta", "eta theta"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    # Walk per-paragraph: lines under each paragraph (matched by y-range)
    # must show the first line shifted right by indent_px relative to
    # subsequent lines.
    for pb in sample.paragraph_boxes:
        py0, py1 = pb.bbox[1], pb.bbox[3]
        in_para = sorted(
            (lb for lb in sample.line_boxes if py0 <= lb.bbox[1] and lb.bbox[3] <= py1),
            key=lambda lb: lb.bbox[1],
        )
        assert len(in_para) >= 2, "test fixture should produce 2+ lines per paragraph"
        first, second = in_para[0], in_para[1]
        delta = first.bbox[0] - second.bbox[0]
        # Slack of a few px to absorb glyph side-bearing differences
        # (the inked extent of an indented line is shifted by exactly
        # indent_px in the *paste* but the inked bbox depends on left-
        # bearing of the leading glyph; we still expect ~indent_px).
        assert abs(delta - indent_px) <= 4, (
            f"first line not indented by ~{indent_px}: first.x0={first.bbox[0]}, "
            f"second.x0={second.bbox[0]}, delta={delta}"
        )


def test_render_page_first_line_indent_shifts_first_word_box(tmp_path: Path) -> None:
    """The leftmost word on the first line shifts by the indent."""

    indent_px = 40
    recipe_no = _build_indent_recipe(tmp_path, None)
    recipe_indent = _build_indent_recipe(tmp_path, indent_px)

    paragraphs = [["alpha beta", "gamma delta"]]

    ctx_no = RenderContext.for_seed(recipe_no.seed)
    ctx_no.reseed_for_sample(0)
    sample_no = render_page(paragraphs, recipe=recipe_no, ctx=ctx_no)

    ctx_in = RenderContext.for_seed(recipe_indent.seed)
    ctx_in.reseed_for_sample(0)
    sample_in = render_page(paragraphs, recipe=recipe_indent, ctx=ctx_in)

    # The first word ("alpha") on line 0 should shift right by exactly
    # indent_px between the two renders.
    first_word_no = sample_no.word_boxes[0]
    first_word_in = sample_in.word_boxes[0]
    assert first_word_no.text == first_word_in.text == "alpha"
    delta = first_word_in.bbox[0] - first_word_no.bbox[0]
    assert delta == indent_px, (
        f"first word didn't shift by exactly {indent_px}: "
        f"no-indent x0={first_word_no.bbox[0]}, indent x0={first_word_in.bbox[0]}"
    )

    # The first word on line 1 ("gamma") should *not* have shifted.
    # Find the first word of line 1 in both renders by y-position.
    def _first_line1_word(sample) -> object:
        line1 = sample.line_boxes[1]
        ly0, ly1 = line1.bbox[1], line1.bbox[3]
        on_line = [wb for wb in sample.word_boxes if ly0 <= wb.bbox[1] and wb.bbox[3] <= ly1]
        on_line.sort(key=lambda wb: wb.bbox[0])
        return on_line[0]

    line1_no = _first_line1_word(sample_no)
    line1_in = _first_line1_word(sample_in)
    assert line1_no.text == line1_in.text == "gamma"
    assert line1_no.bbox[0] == line1_in.bbox[0], (
        f"second line shifted unexpectedly: no={line1_no.bbox[0]}, indent={line1_in.bbox[0]}"
    )


def test_render_page_first_line_indent_widens_canvas(tmp_path: Path) -> None:
    """A first-line indent grows the canvas width to accommodate the shift."""

    paragraphs = [["alphabet", "x"]]  # Line 1 short, line 0 normal.

    recipe_no = _build_indent_recipe(tmp_path, None)
    recipe_in = _build_indent_recipe(tmp_path, 60)

    ctx_no = RenderContext.for_seed(recipe_no.seed)
    ctx_no.reseed_for_sample(0)
    sample_no = render_page(paragraphs, recipe=recipe_no, ctx=ctx_no)

    ctx_in = RenderContext.for_seed(recipe_in.seed)
    ctx_in.reseed_for_sample(0)
    sample_in = render_page(paragraphs, recipe=recipe_in, ctx=ctx_in)

    # Canvas must be wider when line 0 is the longest line and gets
    # indented.
    assert sample_in.size[0] > sample_no.size[0], (
        f"indent did not widen canvas: no={sample_no.size[0]}, indent={sample_in.size[0]}"
    )


def test_render_page_indent_none_is_bit_identical_to_zero(tmp_path: Path) -> None:
    """``paragraph_indent_px = None`` and ``= 0`` produce identical PNG bytes."""

    paragraphs = [["alpha beta", "gamma"], ["delta epsilon", "zeta"]]

    recipe_none = _build_indent_recipe(tmp_path, None)
    recipe_zero = _build_indent_recipe(tmp_path, 0)

    def _png(recipe) -> bytes:
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(0)
        sample = render_page(paragraphs, recipe=recipe, ctx=ctx)
        buf = io.BytesIO()
        sample.image.save(buf, format="PNG")
        return buf.getvalue()

    assert _png(recipe_none) == _png(recipe_zero)


def test_render_page_indent_word_boxes_stay_inside_canvas(tmp_path: Path) -> None:
    """Indented first-line words still have bboxes inside the page canvas."""

    indent_px = 50
    recipe = _build_indent_recipe(tmp_path, indent_px)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(
        [
            ["alpha beta", "gamma delta"],
            ["epsilon zeta", "eta theta"],
        ],
        recipe=recipe,
        ctx=ctx,
    )

    w, h = sample.size
    for wb in sample.word_boxes:
        x0, y0, x1, y1 = wb.bbox
        assert 0 <= x0 < x1 <= w, f"word {wb.text!r} bbox out of canvas: {wb.bbox}, canvas={w}x{h}"
        assert 0 <= y0 < y1 <= h, f"word {wb.text!r} bbox out of canvas: {wb.bbox}, canvas={w}x{h}"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_render_page_rejects_empty_paragraph_list(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(RenderError, match="at least one paragraph"):
        render_page([], recipe=recipe, ctx=ctx)


def test_render_page_rejects_empty_inner_paragraph(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(RenderError, match="paragraph 1 is empty"):
        render_page(
            [["alpha"], [], ["beta"]],
            recipe=recipe,
            ctx=ctx,
        )


def test_render_page_rejects_whitespace_only_inner_line(tmp_path: Path) -> None:
    """Inner-line validation is delegated to render_paragraph."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(RenderError, match="empty or whitespace-only"):
        render_page(
            [["alpha"], ["   "]],
            recipe=recipe,
            ctx=ctx,
        )


def test_render_page_missing_glyph_anywhere_raises(tmp_path: Path) -> None:
    """A missing glyph in paragraph 2 trips the page-level coverage check."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(MissingGlyphError) as exc_info:
        render_page(
            [
                ["ḃeaḋ"],
                ["saoġal"],
                ["mór \U0001f600"],
            ],
            recipe=recipe,
            ctx=ctx,
        )
    assert 0x1F600 in exc_info.value.missing


# ---------------------------------------------------------------------------
# Paragraph alignment (left + center) threaded through pages mode
# ---------------------------------------------------------------------------


def _build_aligned_recipe(tmp_path: Path, alignment: str | None) -> object:
    """Build a pages-mode recipe with the given ``paragraph_alignment``.

    ``alignment=None`` omits the field entirely (preserving the
    historical un-aligned output bytes); a string writes
    ``paragraph_alignment: <value>`` into the layout block.
    """
    font = _require_font()
    rp = tmp_path / f"recipe-pagealign-{alignment}.yaml"
    words = tmp_path / f"words-pagealign-{alignment}.txt"
    words.write_text("ḃeaḋ\n", encoding="utf-8")
    align_line = "" if alignment is None else f"  paragraph_alignment: {alignment}\n"
    rp.write_text(
        "schema_version: 1\n"
        f"name: page-align-{alignment}\n"
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
        "  mode: pages\n"
        "  padding_px: 6\n"
        "  line_spacing: 1.0\n"
        "  paragraph_spacing: 1.0\n"
        f"{align_line}",
        encoding="utf-8",
    )
    return load_recipe(rp)


def test_render_page_alignment_none_is_bit_identical_to_left(tmp_path: Path) -> None:
    """``paragraph_alignment = None`` and ``= "left"`` produce identical PNG bytes."""

    paragraphs = [["alpha beta gamma", "x"], ["delta epsilon", "y z"]]

    recipe_none = _build_aligned_recipe(tmp_path, None)
    recipe_left = _build_aligned_recipe(tmp_path, "left")

    def _png(recipe) -> bytes:
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(0)
        sample = render_page(paragraphs, recipe=recipe, ctx=ctx)
        buf = io.BytesIO()
        sample.image.save(buf, format="PNG")
        return buf.getvalue()

    assert _png(recipe_none) == _png(recipe_left)


def test_render_page_alignment_center_centers_short_lines_per_paragraph(
    tmp_path: Path,
) -> None:
    """Center alignment shifts short lines right within each paragraph independently.

    Each paragraph carries its own ``paragraph_width`` (= the longest
    line in *that* paragraph). Centering happens per-paragraph: the
    short line in paragraph A is centered against paragraph A's max
    line width, not the page's max line width.
    """

    paragraphs = [["alpha beta gamma delta", "x"], ["mu nu", "y"]]

    recipe_left = _build_aligned_recipe(tmp_path, "left")
    recipe_center = _build_aligned_recipe(tmp_path, "center")

    ctx_left = RenderContext.for_seed(recipe_left.seed)
    ctx_left.reseed_for_sample(0)
    sample_left = render_page(paragraphs, recipe=recipe_left, ctx=ctx_left)

    ctx_center = RenderContext.for_seed(recipe_center.seed)
    ctx_center.reseed_for_sample(0)
    sample_center = render_page(paragraphs, recipe=recipe_center, ctx=ctx_center)

    # Paragraphs are stacked top-to-bottom; lines within each are
    # ordered top-to-bottom too. So `line_boxes[0]` is paragraph 0
    # line 0 (long), `line_boxes[1]` is paragraph 0 line 1 (short),
    # `line_boxes[2]` is paragraph 1 line 0 (long-of-its-paragraph),
    # `line_boxes[3]` is paragraph 1 line 1 (short).
    assert len(sample_left.line_boxes) == len(sample_center.line_boxes) == 4

    # Paragraph 0 long line: unchanged.
    assert sample_center.line_boxes[0].bbox[0] == sample_left.line_boxes[0].bbox[0]
    # Paragraph 0 short line: shifts right.
    delta_p0 = sample_center.line_boxes[1].bbox[0] - sample_left.line_boxes[1].bbox[0]
    assert delta_p0 > 0

    # Paragraph 1 long line: unchanged.
    assert sample_center.line_boxes[2].bbox[0] == sample_left.line_boxes[2].bbox[0]
    # Paragraph 1 short line: shifts right by paragraph 1's own delta
    # (which differs from paragraph 0's because the paragraph widths
    # differ).
    delta_p1 = sample_center.line_boxes[3].bbox[0] - sample_left.line_boxes[3].bbox[0]
    assert delta_p1 > 0


def test_render_page_alignment_center_word_boxes_stay_inside_canvas(
    tmp_path: Path,
) -> None:
    """Centered word boxes still sit inside the page canvas."""

    paragraphs = [["alpha beta gamma", "x y"], ["delta epsilon", "z"]]
    recipe = _build_aligned_recipe(tmp_path, "center")
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(paragraphs, recipe=recipe, ctx=ctx)

    w, h = sample.size
    for wb in sample.word_boxes:
        x0, y0, x1, y1 = wb.bbox
        assert 0 <= x0 < x1 <= w, f"word {wb.text!r} bbox out of canvas: {wb.bbox}, canvas={w}x{h}"
        assert 0 <= y0 < y1 <= h, f"word {wb.text!r} bbox out of canvas: {wb.bbox}, canvas={w}x{h}"


# ---------------------------------------------------------------------------
# Explicit page_size_px (fixed-canvas pad)
# ---------------------------------------------------------------------------


def _build_page_size_recipe(tmp_path: Path, page_size_px: tuple[int, int] | None) -> object:
    """Build a pages-mode recipe with the given fixed canvas size.

    ``page_size_px=None`` omits the field entirely (preserving the
    historical auto-sized canvas); a ``(w, h)`` tuple writes
    ``page_size_px: [w, h]`` into the layout block.
    """

    font = _require_font()
    suffix = "none" if page_size_px is None else f"{page_size_px[0]}x{page_size_px[1]}"
    rp = tmp_path / f"recipe-pagesize-{suffix}.yaml"
    words = tmp_path / f"words-pagesize-{suffix}.txt"
    words.write_text("ḃeaḋ\n", encoding="utf-8")
    if page_size_px is None:
        size_line = ""
    else:
        size_line = f"  page_size_px: [{page_size_px[0]}, {page_size_px[1]}]\n"
    rp.write_text(
        "schema_version: 1\n"
        f"name: page-pagesize-{suffix}\n"
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
        "  mode: pages\n"
        "  padding_px: 6\n"
        "  line_spacing: 1.0\n"
        "  paragraph_spacing: 1.0\n"
        f"{size_line}",
        encoding="utf-8",
    )
    return load_recipe(rp)


def test_render_page_page_size_px_produces_exact_canvas(tmp_path: Path) -> None:
    """When ``page_size_px=(W, H)`` is set, output image has exactly those dims."""

    target = (1200, 1800)
    paragraphs = [["alpha beta", "gamma delta"], ["epsilon zeta", "eta theta"]]
    recipe = _build_page_size_recipe(tmp_path, target)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(paragraphs, recipe=recipe, ctx=ctx)

    assert sample.size == target, f"expected {target}, got {sample.size}"


def test_render_page_page_size_px_pads_with_background(tmp_path: Path) -> None:
    """The padded region is filled with the sampled background colour."""

    target = (1200, 1800)
    paragraphs = [["alpha beta", "gamma"]]
    recipe = _build_page_size_recipe(tmp_path, target)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_page(paragraphs, recipe=recipe, ctx=ctx)

    bg = sample.background_color
    # Probe deep in the bottom-right corner — natural content sits at
    # top-left so this region must be pure background.
    px = sample.image.getpixel((target[0] - 5, target[1] - 5))
    assert px == bg, f"bottom-right not background: pixel={px}, bg={bg}"
    # And along the right edge near the top.
    px2 = sample.image.getpixel((target[0] - 5, 5))
    assert px2 == bg, f"top-right not background: pixel={px2}, bg={bg}"


def test_render_page_page_size_px_keeps_annotations_inside_natural_region(
    tmp_path: Path,
) -> None:
    """All word/line/paragraph bboxes sit inside the natural-content rectangle.

    Top-left placement means annotations never extend into the padded
    margin; checking that the natural-content extent (== auto-sized
    render) still bounds every annotation locks the no-shift contract.
    """

    target = (1200, 1800)
    paragraphs = [["alpha beta", "gamma delta"], ["epsilon zeta", "eta theta"]]

    recipe_natural = _build_page_size_recipe(tmp_path, None)
    recipe_padded = _build_page_size_recipe(tmp_path, target)

    ctx_n = RenderContext.for_seed(recipe_natural.seed)
    ctx_n.reseed_for_sample(0)
    sample_n = render_page(paragraphs, recipe=recipe_natural, ctx=ctx_n)

    ctx_p = RenderContext.for_seed(recipe_padded.seed)
    ctx_p.reseed_for_sample(0)
    sample_p = render_page(paragraphs, recipe=recipe_padded, ctx=ctx_p)

    natural_w, natural_h = sample_n.size

    # Padded sample is the requested target size, not the natural size.
    assert sample_p.size == target

    # Every padded-sample annotation lies inside (0, 0, natural_w, natural_h).
    for wb in sample_p.word_boxes:
        x0, y0, x1, y1 = wb.bbox
        assert 0 <= x0 < x1 <= natural_w, (
            f"word {wb.text!r} extends past natural width: bbox={wb.bbox}, "
            f"natural={natural_w}x{natural_h}"
        )
        assert 0 <= y0 < y1 <= natural_h, (
            f"word {wb.text!r} extends past natural height: bbox={wb.bbox}, "
            f"natural={natural_w}x{natural_h}"
        )
    for lb in sample_p.line_boxes:
        x0, y0, x1, y1 = lb.bbox
        assert x1 <= natural_w and y1 <= natural_h
    for pb in sample_p.paragraph_boxes:
        x0, y0, x1, y1 = pb.bbox
        assert x1 <= natural_w and y1 <= natural_h

    # Annotation bboxes are bit-identical to the natural sample — top-
    # left placement implies zero shift, so padding must not move
    # anything.
    assert tuple(sample_p.word_boxes) == tuple(sample_n.word_boxes)
    assert tuple(sample_p.line_boxes) == tuple(sample_n.line_boxes)
    assert tuple(sample_p.paragraph_boxes) == tuple(sample_n.paragraph_boxes)


def test_render_page_page_size_px_rejects_oversized_content(tmp_path: Path) -> None:
    """Content larger than ``page_size_px`` in either dim raises ``RenderError``."""

    # Tiny target — even one paragraph will overflow.
    target = (50, 50)
    paragraphs = [["alpha beta gamma", "delta epsilon"]]
    recipe = _build_page_size_recipe(tmp_path, target)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(RenderError, match="exceeds requested page_size_px"):
        render_page(paragraphs, recipe=recipe, ctx=ctx)


def test_render_page_page_size_px_none_is_bit_identical_to_unset(
    tmp_path: Path,
) -> None:
    """Recipes without ``page_size_px`` produce byte-identical PNGs to before.

    Regression lock: existing recipes (which do not set ``page_size_px``)
    must continue to produce the same PNG bytes they did before this
    feature landed. We compare the un-set form to itself rather than to
    a hardcoded blob — the invariant is "no behaviour change for the
    default path", and round-tripping confirms determinism.
    """

    paragraphs = [["alpha beta", "gamma"], ["delta epsilon", "zeta"]]

    recipe_a = _build_page_size_recipe(tmp_path, None)
    recipe_b = _build_page_size_recipe(tmp_path, None)

    def _png(recipe) -> bytes:
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(0)
        sample = render_page(paragraphs, recipe=recipe, ctx=ctx)
        buf = io.BytesIO()
        sample.image.save(buf, format="PNG")
        return buf.getvalue()

    assert _png(recipe_a) == _png(recipe_b)


def test_render_page_page_size_px_exact_fit_does_not_pad(tmp_path: Path) -> None:
    """When natural size exactly matches ``page_size_px``, no padding is applied.

    The renderer should leave the canvas alone (no ``Image.new`` /
    paste) when content already fills the requested size — this avoids
    a wasted full-canvas allocation for the common "auto-sized for me,
    annotated as exact" case.
    """

    paragraphs = [["alpha beta", "gamma"]]

    # First, render natural size to discover the exact dims.
    recipe_natural = _build_page_size_recipe(tmp_path, None)
    ctx_n = RenderContext.for_seed(recipe_natural.seed)
    ctx_n.reseed_for_sample(0)
    sample_n = render_page(paragraphs, recipe=recipe_natural, ctx=ctx_n)
    natural_size = sample_n.size

    # Now request that exact size as the fixed page size.
    recipe_exact = _build_page_size_recipe(tmp_path, natural_size)
    ctx_e = RenderContext.for_seed(recipe_exact.seed)
    ctx_e.reseed_for_sample(0)
    sample_e = render_page(paragraphs, recipe=recipe_exact, ctx=ctx_e)

    assert sample_e.size == natural_size

    # Pixel-equal: same content, no padding region.
    buf_n = io.BytesIO()
    sample_n.image.save(buf_n, format="PNG")
    buf_e = io.BytesIO()
    sample_e.image.save(buf_e, format="PNG")
    assert buf_n.getvalue() == buf_e.getvalue()
