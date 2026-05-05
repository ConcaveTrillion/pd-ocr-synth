"""Tests for the M05 render layer.

Covers the three roadmap-required test cases for M05:

- Smoke: a small render-mode pipeline produces non-empty PNG output.
- Determinism: same recipe + seed + sample index → byte-identical PNG.
- Glyph-coverage skip: a token containing a codepoint absent from
  the chosen font raises ``MissingGlyphError`` with the missing
  codepoints set.

The bundled Bunchló GC font is reused for all three cases. Tests
skip cleanly when the font is not present (e.g. fresh checkout that
hasn't run ``./scripts/fetch-fonts-gaelic.sh``).
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from pd_ocr_synth.recipe import load_recipe
from pd_ocr_synth.render import (
    MissingGlyphError,
    RenderContext,
    render_word_crop,
)
from pd_ocr_synth.render.context import branched_seed

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; render tests skipped.")
    return _BUNDLED_FONT


_RECIPE_TEMPLATE = """\
schema_version: 1
name: render-smoke
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
  mode: word_crops
  padding_px: 6
"""


def _make_recipe(tmp_path: Path) -> object:
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE_TEMPLATE.format(font_path=font), encoding="utf-8")
    (tmp_path / "words.txt").write_text("ḃeaḋ\n", encoding="utf-8")
    return load_recipe(rp)


def test_render_word_crop_produces_non_empty_png(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_word_crop("ḃeaḋ", recipe=recipe, ctx=ctx)

    assert sample.size[0] > 0 and sample.size[1] > 0
    assert sample.bbox[2] > sample.bbox[0] and sample.bbox[3] > sample.bbox[1]
    assert sample.font_path == _BUNDLED_FONT
    assert 14.0 <= sample.font_size_pt <= 22.0
    assert sample.dpi == 300
    # Per-cluster glyph runs (one per codepoint cluster, not per glyph).
    assert sample.glyph_runs, "expected at least one glyph run"

    # Round-tripping through PNG must produce non-trivial bytes.
    buf = io.BytesIO()
    sample.image.save(buf, format="PNG")
    assert len(buf.getvalue()) > 100


def test_render_smoke_five_samples_all_render(tmp_path: Path) -> None:
    """Render 5 samples sequentially and confirm every one succeeds.

    Mirrors the roadmap's ``Smoke: render 5 samples from gaelic.yaml``
    deliverable but uses an inline tmp recipe so the test is hermetic.
    """

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)

    rendered = []
    for index in range(5):
        ctx.reseed_for_sample(index)
        sample = render_word_crop("ḃeaḋ", recipe=recipe, ctx=ctx)
        rendered.append(sample)

    assert len(rendered) == 5
    for s in rendered:
        assert s.size[0] > 0 and s.size[1] > 0


def test_render_is_deterministic_per_seed_and_index(tmp_path: Path) -> None:
    recipe = _make_recipe(tmp_path)

    def _render_at(index: int) -> bytes:
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(index)
        sample = render_word_crop("ḃeaḋ", recipe=recipe, ctx=ctx)
        buf = io.BytesIO()
        sample.image.save(buf, format="PNG")
        return buf.getvalue()

    # Same seed + sample_index → byte-identical PNG.
    assert _render_at(0) == _render_at(0)
    assert _render_at(7) == _render_at(7)

    # Different sample_index → different output (statistically; the
    # bunched RNG mix means the chance of collision is vanishingly
    # small with 16 px+ size variance, color variance, and padding
    # variance all live).
    assert _render_at(0) != _render_at(1)


def test_branched_seed_is_stable_and_index_sensitive() -> None:
    assert branched_seed(42, 0) == branched_seed(42, 0)
    assert branched_seed(42, 0) != branched_seed(42, 1)
    assert branched_seed(0, 5) != branched_seed(1, 5)


def test_missing_glyph_raises_missing_glyph_error(tmp_path: Path) -> None:
    """The chosen font does not cover U+1F600 (grinning face emoji).

    The renderer must detect this *before* shaping and raise
    ``MissingGlyphError`` so the dataset loop can record a
    ``missing_glyph`` skip reason in the manifest.
    """

    recipe = _make_recipe(tmp_path)
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)

    with pytest.raises(MissingGlyphError) as exc_info:
        render_word_crop("hello \U0001f600", recipe=recipe, ctx=ctx)

    err = exc_info.value
    assert 0x1F600 in err.missing
    assert err.font_path == _BUNDLED_FONT
    assert "U+1F600" in str(err)
