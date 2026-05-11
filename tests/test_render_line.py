"""Tests for ``lines``-mode rendering (M09 layout chunk).

Mirrors the structure of ``test_render.py`` but exercises
:func:`pd_ocr_synth.render.render_line` — a single baseline with
multiple words, returning a :class:`RenderedSample` whose
``word_boxes`` carry per-word ground truth.

These tests skip cleanly when the bundled Bunchló GC font isn't
present (e.g. fresh checkout that hasn't run
``./scripts/fetch-fonts-gaelic.sh``).
"""

from __future__ import annotations

import io
import itertools
from pathlib import Path

import pytest

from pd_ocr_synth.recipe import load_recipe
from pd_ocr_synth.render import (
    MissingGlyphError,
    RenderContext,
    RenderError,
    WordBox,
    render_line,
)

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; line render tests skipped.")
    return _BUNDLED_FONT


_RECIPE_TEMPLATE = """\
schema_version: 1
name: line-smoke
seed: 42
output:
  format: pd-ocr-trainer/v1
  mode: recognition
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
  mode: lines
  padding_px: 6
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


def test_render_line_produces_non_empty_png(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_line("ḃeaḋ saoġal", recipe=recipe, ctx=ctx)

    assert sample.text == "ḃeaḋ saoġal"
    assert sample.size[0] > 0 and sample.size[1] > 0
    # Tight bbox is non-degenerate.
    assert sample.bbox[2] > sample.bbox[0]
    assert sample.bbox[3] > sample.bbox[1]
    assert sample.font_path == _BUNDLED_FONT
    assert sample.glyph_runs, "expected at least one glyph run"
    # PNG round-trip yields non-trivial bytes.
    buf = io.BytesIO()
    sample.image.save(buf, format="PNG")
    assert len(buf.getvalue()) > 100


# ---------------------------------------------------------------------------
# Per-word ground truth
# ---------------------------------------------------------------------------


def test_render_line_emits_one_word_box_per_input_word(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_line("ḃeaḋ saoġal mór", recipe=recipe, ctx=ctx)

    assert len(sample.word_boxes) == 3
    assert [w.text for w in sample.word_boxes] == ["ḃeaḋ", "saoġal", "mór"]
    for wb in sample.word_boxes:
        assert isinstance(wb, WordBox)
        x0, y0, x1, y1 = wb.bbox
        assert x1 > x0 > 0, f"word {wb.text!r} bbox not strictly inside canvas: {wb.bbox}"
        assert y1 > y0 > 0, f"word {wb.text!r} bbox not strictly inside canvas: {wb.bbox}"


def test_render_line_word_boxes_are_left_to_right_and_disjoint(tmp_path: Path) -> None:
    """For Latin / Cló Gaelach script, word bboxes don't overlap on x.

    This is a structural sanity check: words sit on a single
    baseline with intervening whitespace. If two word bboxes ever
    overlap on x, something's wrong with the cluster→word mapping
    or the line shaping.
    """

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_line("aon dó trí", recipe=recipe, ctx=ctx)

    boxes = sample.word_boxes
    assert len(boxes) == 3
    # Strictly increasing x0; previous x1 not greater than next x0.
    # ``strict=False`` because ``boxes[1:]`` is one shorter than ``boxes``
    # by construction; we want the (n-1) consecutive pairs.
    for prev, curr in itertools.pairwise(boxes):
        assert prev.bbox[0] < curr.bbox[0], f"x0 not increasing: {prev=} {curr=}"
        assert prev.bbox[2] <= curr.bbox[0], (
            f"word bboxes overlap on x: {prev.text!r}{prev.bbox} vs {curr.text!r}{curr.bbox}"
        )


def test_render_line_word_box_sits_inside_sample_bbox(tmp_path: Path) -> None:
    """Each word's bbox is contained in the sample's tight inked bbox."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_line("ḃeaḋ saoġal", recipe=recipe, ctx=ctx)

    sx0, sy0, sx1, sy1 = sample.bbox
    for wb in sample.word_boxes:
        wx0, wy0, wx1, wy1 = wb.bbox
        assert sx0 <= wx0 <= wx1 <= sx1, f"word x range {wb.bbox} escapes sample bbox {sample.bbox}"
        assert sy0 <= wy0 <= wy1 <= sy1, f"word y range {wb.bbox} escapes sample bbox {sample.bbox}"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_render_line_is_deterministic_per_seed_and_index(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)

    def _render_at(index: int) -> bytes:
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(index)
        sample = render_line("ḃeaḋ saoġal", recipe=recipe, ctx=ctx)
        buf = io.BytesIO()
        sample.image.save(buf, format="PNG")
        return buf.getvalue()

    assert _render_at(0) == _render_at(0)
    assert _render_at(7) == _render_at(7)
    assert _render_at(0) != _render_at(1)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_render_line_missing_glyph_in_mid_line_raises(tmp_path: Path) -> None:
    """A grinning-face emoji mid-line still trips the coverage check."""

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(MissingGlyphError) as exc_info:
        render_line("ḃeaḋ \U0001f600 saoġal", recipe=recipe, ctx=ctx)
    err = exc_info.value
    assert 0x1F600 in err.missing


def test_render_line_rejects_empty_text(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(RenderError):
        render_line("", recipe=recipe, ctx=ctx)


def test_render_line_rejects_whitespace_only_text(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(RenderError):
        render_line("   ", recipe=recipe, ctx=ctx)


# ---------------------------------------------------------------------------
# Single-word degenerate case
# ---------------------------------------------------------------------------


def test_render_line_single_word_still_emits_one_word_box(tmp_path: Path) -> None:
    """A line containing one word is a valid lines-mode sample.

    The word_box list has one entry and matches the word text;
    the image is essentially a word_crop equivalent.
    """

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_line("ḃeaḋ", recipe=recipe, ctx=ctx)

    assert len(sample.word_boxes) == 1
    assert sample.word_boxes[0].text == "ḃeaḋ"
    x0, y0, x1, y1 = sample.word_boxes[0].bbox
    assert x1 > x0 and y1 > y0
