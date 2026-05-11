"""Tests for the ``ParagraphBox`` dataclass + ``paragraph_boxes``
field on :class:`RenderedSample` (M09).

This is the foundational chunk that lands the per-paragraph ground-
truth shape ahead of the ``pages`` mode renderer. Locks:

- ``ParagraphBox`` exists and is shaped like ``LineBox`` /
  ``WordBox`` (immutable, ``text`` + ``bbox``).
- ``RenderedSample.paragraph_boxes`` defaults to an empty tuple, so
  every existing sample (``word_crops``, ``lines``) keeps emitting
  legal payloads with no code changes.
- ``render_paragraph`` populates ``paragraph_boxes`` with exactly one
  entry whose bbox equals the union of ``line_boxes`` (which by
  construction also equals ``sample.bbox`` for paragraph mode).
- ``ParagraphBox.text`` matches the paragraph's ``"\\n"``-joined
  text — the same convention used by
  :attr:`RenderedSample.text` for paragraphs.

Tests skip cleanly when the bundled Bunchló GC font isn't present.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from pd_ocr_synth.recipe import load_recipe
from pd_ocr_synth.render import (
    ParagraphBox,
    RenderContext,
    RenderedSample,
    render_paragraph,
    render_word_crop,
)

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; paragraph_box tests skipped.")
    return _BUNDLED_FONT


_PARAGRAPH_RECIPE_TEMPLATE = """\
schema_version: 1
name: para-box-test
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
  font_size_pt: 18
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: paragraphs
  padding_px: 6
  line_spacing: 1.2
"""


_WORD_CROP_RECIPE_TEMPLATE = """\
schema_version: 1
name: wc-para-box-test
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
  font_size_pt: 18
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: word_crops
  padding_px: 6
"""


def _make_paragraph_recipe(tmp_path: Path) -> object:
    font = _require_font()
    rp = tmp_path / "paragraph-recipe.yaml"
    rp.write_text(_PARAGRAPH_RECIPE_TEMPLATE.format(font_path=font), encoding="utf-8")
    (tmp_path / "words.txt").write_text("ḃeaḋ\n", encoding="utf-8")
    return load_recipe(rp)


def _make_word_crop_recipe(tmp_path: Path) -> object:
    font = _require_font()
    rp = tmp_path / "word-crop-recipe.yaml"
    rp.write_text(_WORD_CROP_RECIPE_TEMPLATE.format(font_path=font), encoding="utf-8")
    (tmp_path / "words.txt").write_text("ḃeaḋ\n", encoding="utf-8")
    return load_recipe(rp)


# ---------------------------------------------------------------------------
# Dataclass shape — mirrors LineBox / WordBox
# ---------------------------------------------------------------------------


def test_paragraph_box_constructs_with_text_and_bbox() -> None:
    pb = ParagraphBox(text="hello world", bbox=(0, 0, 10, 20))
    assert pb.text == "hello world"
    assert pb.bbox == (0, 0, 10, 20)


def test_paragraph_box_is_frozen() -> None:
    pb = ParagraphBox(text="hello", bbox=(0, 0, 10, 20))
    with pytest.raises((FrozenInstanceError, AttributeError)):
        pb.text = "changed"  # type: ignore[misc]


def test_paragraph_box_equality_is_value_based() -> None:
    a = ParagraphBox(text="hello", bbox=(0, 0, 10, 20))
    b = ParagraphBox(text="hello", bbox=(0, 0, 10, 20))
    c = ParagraphBox(text="world", bbox=(0, 0, 10, 20))
    assert a == b
    assert a != c


# ---------------------------------------------------------------------------
# RenderedSample default — every existing sample has empty paragraph_boxes
# ---------------------------------------------------------------------------


def test_rendered_sample_default_paragraph_boxes_is_empty_tuple() -> None:
    """Constructing a sample without ``paragraph_boxes`` yields ``()``.

    This locks the additive nature of the new field — every existing
    in-process and across-worker producer that doesn't know about
    ``paragraph_boxes`` keeps shipping legal samples.
    """

    from PIL import Image

    sample = RenderedSample(
        text="x",
        image=Image.new("RGB", (4, 4), color=(0, 0, 0)),
        bbox=(0, 0, 4, 4),
        font_path=Path("/nonexistent.otf"),
        font_size_pt=12.0,
        dpi=300,
        ink_color=(0, 0, 0),
        background_color=(255, 255, 255),
    )
    assert sample.paragraph_boxes == ()


def test_word_crop_sample_has_empty_paragraph_boxes(tmp_path: Path) -> None:
    """Word-crop mode emits ``paragraph_boxes=()`` — paragraphs aren't
    a concept at this layout level."""

    recipe = _make_word_crop_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_word_crop("ḃeaḋ", recipe=recipe, ctx=ctx)
    assert sample.paragraph_boxes == ()


# ---------------------------------------------------------------------------
# render_paragraph populates paragraph_boxes with a single entry
# ---------------------------------------------------------------------------


def test_render_paragraph_emits_single_paragraph_box(tmp_path: Path) -> None:
    """A ``paragraphs``-mode sample carries exactly one paragraph_box."""

    recipe = _make_paragraph_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["ḃeaḋ saoġal", "mór is beag"],
        recipe=recipe,
        ctx=ctx,
    )

    assert len(sample.paragraph_boxes) == 1
    assert isinstance(sample.paragraph_boxes[0], ParagraphBox)


def test_render_paragraph_paragraph_box_text_matches_joined_lines(
    tmp_path: Path,
) -> None:
    """``ParagraphBox.text`` is the ``"\\n"``-joined paragraph text."""

    recipe = _make_paragraph_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    lines = ["ḃeaḋ saoġal", "mór is beag", "aon dó"]
    sample = render_paragraph(lines, recipe=recipe, ctx=ctx)

    assert sample.paragraph_boxes[0].text == "\n".join(lines)
    # And it equals the sample's own text payload (paragraph-mode
    # convention), so a downstream consumer reading either field
    # gets the same string.
    assert sample.paragraph_boxes[0].text == sample.text


def test_render_paragraph_box_bbox_is_union_of_line_boxes(tmp_path: Path) -> None:
    """The single paragraph's bbox = union of its ``line_boxes``."""

    recipe = _make_paragraph_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["alpha beta", "gamma delta", "epsilon"],
        recipe=recipe,
        ctx=ctx,
    )

    union_x0 = min(lb.bbox[0] for lb in sample.line_boxes)
    union_y0 = min(lb.bbox[1] for lb in sample.line_boxes)
    union_x1 = max(lb.bbox[2] for lb in sample.line_boxes)
    union_y1 = max(lb.bbox[3] for lb in sample.line_boxes)
    assert sample.paragraph_boxes[0].bbox == (union_x0, union_y0, union_x1, union_y1)


def test_render_paragraph_box_bbox_equals_sample_bbox(tmp_path: Path) -> None:
    """For ``paragraphs`` mode, paragraph_box bbox == sample bbox."""

    recipe = _make_paragraph_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(["alpha beta", "gamma"], recipe=recipe, ctx=ctx)

    assert sample.paragraph_boxes[0].bbox == sample.bbox


def test_render_paragraph_single_line_paragraph_box(tmp_path: Path) -> None:
    """Degenerate single-line paragraph still carries one paragraph_box.

    The paragraph_box's text equals the line; its bbox equals the
    line's bbox.
    """

    recipe = _make_paragraph_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(["ḃeaḋ saoġal"], recipe=recipe, ctx=ctx)

    assert len(sample.paragraph_boxes) == 1
    pb = sample.paragraph_boxes[0]
    assert pb.text == "ḃeaḋ saoġal"
    # Single-line paragraph → paragraph bbox = line bbox.
    assert pb.bbox == sample.line_boxes[0].bbox


def test_render_paragraph_box_is_inside_canvas(tmp_path: Path) -> None:
    """The paragraph_box lives entirely inside the canvas."""

    recipe = _make_paragraph_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["ḃeaḋ saoġal", "mór beag"],
        recipe=recipe,
        ctx=ctx,
    )

    w, h = sample.size
    x0, y0, x1, y1 = sample.paragraph_boxes[0].bbox
    assert 0 <= x0 < x1 <= w
    assert 0 <= y0 < y1 <= h
