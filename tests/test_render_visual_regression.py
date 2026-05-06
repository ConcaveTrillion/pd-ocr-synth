"""Visual regression tests — pin canonical render output by sha256.

These tests lock the byte-for-byte PNG output of each layout-mode
entry point (``render_word_crop``, ``render_line``, ``render_paragraph``,
``render_page``) against checked-in sha256 digest constants. They
exist to catch *unintentional* drift from refactors, font upgrades,
or RNG-state shifts that the existing per-test determinism asserts
(same call → same bytes within one Python process) would happily
miss.

What this is **not**: a pixel-similarity / image-diff check. The
canonical recipe surface area is kept deliberately tiny (single
font, single corpus string, fixed seed, no degradations) so the
digest-equality check is robust against anything *except* a real
rendering-pipeline change.

Self-bootstrapping pattern
--------------------------

The first time a new digest pin is added — or when an intentional
pipeline change flips the bytes — these tests will fail with the
expected vs. actual digests both in the failure message. To update:

1. Run ``PD_OCR_SYNTH_REGEN_VISUAL_DIGESTS=1 uv run pytest \\
   tests/test_render_visual_regression.py`` (the env var prints the
   freshly-computed digests in a copy-pasteable form via
   ``pytest -s``-style stdout while still failing the assertion).
2. Eyeball-verify the output of the failing case is intentional
   (e.g., view the rendered PNG via the preview helpers, or compare
   to a reference image you trust).
3. Paste the new digest into ``_PINS`` below and re-run.

We intentionally do **not** auto-update the digest from inside the
test — every drift event should land in a commit that documents
*why* the bytes shifted (font upgrade, paragraph-spacing rounding
change, etc.).

Skip behavior matches the rest of the render test suite: if the
bundled Bunchló GC font is missing, every case here is skipped.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path

import pytest

from pd_ocr_synth.recipe import load_recipe
from pd_ocr_synth.render import (
    RenderContext,
    render_line,
    render_page,
    render_paragraph,
    render_word_crop,
)

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)
_REGEN_ENV = "PD_OCR_SYNTH_REGEN_VISUAL_DIGESTS"


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; visual regression tests skipped.")
    return _BUNDLED_FONT


# ---------------------------------------------------------------------------
# Canonical recipes — kept minimal on purpose.
#
# Each recipe pins:
#   - one font, one corpus string ``ḃeaḋ`` (same as other render tests).
#   - a fixed seed (1) so the per-sample RNG branch is deterministic.
#   - a *single* font_size_pt (no range) and a *single* color so the
#     output is invariant to any RNG-consumption-order change in the
#     scalar / range / color samplers.
#
# The tighter the surface, the lower the false-positive drift risk.
# ---------------------------------------------------------------------------

_RECIPE_WORD_CROP = """\
schema_version: 1
name: visreg-word-crop
seed: 1
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

_RECIPE_LINES = """\
schema_version: 1
name: visreg-lines
seed: 1
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
  mode: lines
  padding_px: 6
"""

_RECIPE_PARAGRAPHS = """\
schema_version: 1
name: visreg-paragraphs
seed: 1
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

_RECIPE_PAGES = """\
schema_version: 1
name: visreg-pages
seed: 1
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
  mode: pages
  padding_px: 8
  line_spacing: 1.2
  paragraph_spacing: 1.0
"""


def _write_recipe(tmp_path: Path, template: str, name: str) -> object:
    font = _require_font()
    rp = tmp_path / f"{name}.yaml"
    rp.write_text(template.format(font_path=font), encoding="utf-8")
    (tmp_path / "words.txt").write_text("ḃeaḋ\n", encoding="utf-8")
    return load_recipe(rp)


def _png_sha256(image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


# ---------------------------------------------------------------------------
# Pinned digests.
#
# To regenerate intentionally, set ``PD_OCR_SYNTH_REGEN_VISUAL_DIGESTS=1``,
# run the test (it will still fail, but will print the actual digest),
# then update the pin here and commit.
# ---------------------------------------------------------------------------

_PINS = {
    "word_crop:ḃeaḋ": "0d224650780a2a5eacbf9dba40e69062d5c977ecaeb134ba4ed092c2e51d7504",
    "line:ḃeaḋ saoġal": "68864f7c91ff4a079701e6564e403e14ab7ba967be65cf6b03c35bafb4e5dba9",
    "paragraph:two-lines": "e625fabba7d3d0f5cfea548a3a7ab9b5685d6b1b12a3998cf996ba4a92be90b5",
    "page:two-paragraphs": "6ae102edf3f1bcf9c38b926e19e095c30fe288c710b5017c03e4074c3cf514f9",
}


def _check_pin(label: str, image) -> None:
    """Assert the rendered ``image``'s PNG sha256 matches the pinned digest.

    When ``PD_OCR_SYNTH_REGEN_VISUAL_DIGESTS`` is set, prints the
    actual digest (so a regen run captures all four in one go) before
    asserting — the assertion still fires so the developer is forced
    to paste the new digest in deliberately.
    """

    actual = _png_sha256(image)
    expected = _PINS[label]
    if os.environ.get(_REGEN_ENV):
        print(f"\n[visreg-regen] {label!r}\n  actual:   {actual}\n  expected: {expected}")
    assert actual == expected, (
        f"visual regression for {label!r}:\n"
        f"  expected sha256: {expected}\n"
        f"  actual   sha256: {actual}\n"
        f"If this change is intentional, update _PINS in "
        f"tests/test_render_visual_regression.py and commit with a note "
        f"on why the bytes shifted (e.g. font upgrade, layout fix)."
    )


# ---------------------------------------------------------------------------
# One test per layout mode.
# ---------------------------------------------------------------------------


def test_visreg_word_crop_pinned_digest(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path, _RECIPE_WORD_CROP, "word_crop")
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_word_crop("ḃeaḋ", recipe=recipe, ctx=ctx)
    _check_pin("word_crop:ḃeaḋ", sample.image)


def test_visreg_line_pinned_digest(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path, _RECIPE_LINES, "lines")
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_line("ḃeaḋ saoġal", recipe=recipe, ctx=ctx)
    _check_pin("line:ḃeaḋ saoġal", sample.image)


def test_visreg_paragraph_pinned_digest(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path, _RECIPE_PARAGRAPHS, "paragraphs")
    ctx = RenderContext.for_seed(recipe.seed)
    ctx.reseed_for_sample(0)
    sample = render_paragraph(
        ["ḃeaḋ saoġal", "agus mór"],
        recipe=recipe,
        ctx=ctx,
    )
    _check_pin("paragraph:two-lines", sample.image)


def test_visreg_page_pinned_digest(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path, _RECIPE_PAGES, "pages")
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
    _check_pin("page:two-paragraphs", sample.image)


# ---------------------------------------------------------------------------
# Cross-call stability sanity check — the same rendered bytes hash
# the same on a second call. Cheap belt-and-braces against a future
# refactor that introduces non-determinism *between* calls (the
# existing per-mode determinism tests check intra-call, this checks
# stability of the digest itself).
# ---------------------------------------------------------------------------


def test_visreg_word_crop_digest_is_stable_across_two_calls(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path, _RECIPE_WORD_CROP, "word_crop")

    def _hash() -> str:
        ctx = RenderContext.for_seed(recipe.seed)
        ctx.reseed_for_sample(0)
        sample = render_word_crop("ḃeaḋ", recipe=recipe, ctx=ctx)
        return _png_sha256(sample.image)

    assert _hash() == _hash()
