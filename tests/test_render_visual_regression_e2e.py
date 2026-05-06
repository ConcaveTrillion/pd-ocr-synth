"""End-to-end visual regression — pin ``run_recipe`` output by sha256.

Companion to ``test_render_visual_regression.py``. The per-renderer
pins there lock the four layout-mode entry points
(``render_word_crop`` / ``render_line`` / ``render_paragraph`` /
``render_page``) plus one degraded-line case at the *function* level
— direct calls into the renderer bypassing ``run_recipe``, the
recognition writer, the corpus tokenizer, and the per-sample RNG-fork
machinery.

This file pins the next layer up: the output of ``run_recipe`` end
to end. Drift here can come from the per-renderer changes the lower
pins already cover (which is fine — both sets fail together) **or**
from a bug in the writer / tokenizer / sample-pick RNG that the
lower pins don't see. Specifically:

- ``RecognitionWriter`` PNG-save path (``Image.save(format="PNG")``
  rather than the ``BytesIO`` round-trip the unit tests use).
- The per-sample RNG fork inside ``run_recipe`` — token pick is
  ``random.Random(seed ^ 0xC0FFEE).choice(...)`` which the low-level
  tests don't exercise.
- The corpus-tokenizer split for the configured ``layout.mode``.
- ``labels.json`` ordering / encoding (pretty-printed, sorted).

What we don't pin
-----------------

``manifest.jsonl``, ``stats.json`` and ``recipe.snapshot.yaml`` carry
machine-dependent or time-dependent fields:

- ``manifest.jsonl`` records the *absolute* font path under
  ``font.path`` — different on every machine / CI runner.
- ``stats.json`` has ``wall_time_seconds`` (non-deterministic).
- ``recipe.snapshot.yaml`` round-trips absolute font paths the same
  way (the snapshot module documents this).

Pinning those would require a portability shim (relative-path
rewriting + wall-time mask) which is out of scope for the M10 polish
chunk. We pin the two artifacts that *are* portable — the per-image
PNG bytes and the labels.json content — and that already covers the
writer-side bug surface that motivated this test.

Self-bootstrap and Pillow notes mirror the per-renderer file; see
``test_render_visual_regression.py`` for the full regen procedure.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from pd_ocr_synth.recipe import load_recipe
from pd_ocr_synth.render import run_recipe

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)
_REGEN_ENV = "PD_OCR_SYNTH_REGEN_VISUAL_DIGESTS"


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; e2e visual regression tests skipped.")
    return _BUNDLED_FONT


# ---------------------------------------------------------------------------
# Canonical recipe — kept minimal on purpose.
#
# Pins:
#   - one font, multi-word corpus so the per-sample token pick produces
#     three *distinct* samples. A single-word corpus would render three
#     byte-identical images and silently fail to lock the per-sample
#     RNG fork inside ``run_recipe`` (the very surface this e2e pin
#     exists to cover).
#   - count=3 so we exercise the multi-sample loop (per-sample RNG
#     reseed, padded filename width across multiple indices) without
#     paying for a slow render.
#   - fixed seed (``1``) — the token-pick PRNG is
#     ``random.Random(seed ^ 0xC0FFEE).choice(tokens)``, so the seed
#     determines which words land at which index.
#   - single ``font_size_pt`` and single ``ink_color`` / ``background_color``.
#   - no degradations.
# ---------------------------------------------------------------------------

_RECIPE_E2E = """\
schema_version: 1
name: visreg-e2e-word-crops
seed: 1
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 3
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


def _write_recipe(tmp_path: Path) -> object:
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE_E2E.format(font_path=font), encoding="utf-8")
    # Multi-word corpus → three distinct token picks at fixed seed.
    # Whitespace-separated; the ``word_crops`` tokenizer splits on
    # whitespace so each whole token becomes a candidate sample.
    (tmp_path / "words.txt").write_text(
        "ḃeaḋ saoġal mór ċeann an ḃoṫair ḋuine ḟir\n", encoding="utf-8"
    )
    return load_recipe(rp)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Pinned digests.
#
# Per-image PNG sha256 covers the writer's ``Image.save(..., format="PNG")``
# path. Labels-json sha256 covers the writer's pretty-printed +
# trailing-newline encoding and its sorted-by-filename ordering.
#
# Regenerate by running with ``PD_OCR_SYNTH_REGEN_VISUAL_DIGESTS=1``.
# ---------------------------------------------------------------------------

_PINS = {
    # ``mór`` at sample 0.
    "image:0000000.png": "001dd2706a90ccb64722b2a6c4e0f8bbca7627bea488acd14a5bcdc42c0279be",
    # ``ḃoṫair`` at sample 1.
    "image:0000001.png": "1a8c913a7d34746c98a2c3760c13f9e29f5aa24ec50136835cfc725d23ba5771",
    # ``ḋuine`` at sample 2.
    "image:0000002.png": "4c65b14d679f4c17783641dad1674812470911316c8e767df23cb629f9161616",
    # ``labels.json`` is sorted-by-filename, pretty-printed
    # (``indent=2``), with a trailing newline. The pinned digest covers
    # all three of those formatting choices.
    "labels.json": "7fb4d992262df32b62059c3ca3c43c86bb146cac7de6e65834074b9a3c3c8515",
}


def _check_pin(label: str, digest: str) -> None:
    expected = _PINS[label]
    if os.environ.get(_REGEN_ENV):
        print(f"\n[visreg-e2e-regen] {label!r}\n  actual:   {digest}\n  expected: {expected}")
    assert digest == expected, (
        f"end-to-end visual regression for {label!r}:\n"
        f"  expected sha256: {expected}\n"
        f"  actual   sha256: {digest}\n"
        f"If this change is intentional, update _PINS in "
        f"tests/test_render_visual_regression_e2e.py and commit with a note "
        f"on why the bytes shifted (e.g. writer refactor, font upgrade)."
    )


# ---------------------------------------------------------------------------
# The single end-to-end pin test.
#
# Splitting per-image into separate test functions would pay for three
# ``run_recipe`` invocations to lock three samples; one test that
# checks all four digests off a single render run is faster and keeps
# the fixture surface small.
# ---------------------------------------------------------------------------


def test_visreg_e2e_run_recipe_word_crops_pinned(tmp_path: Path) -> None:
    recipe = _write_recipe(tmp_path)

    out_dir = tmp_path / "trainer-out"
    result = run_recipe(
        recipe,
        output_dir=out_dir,
        count=3,
        seed=1,
        workers=1,
        progress=False,
        audit=False,
    )
    assert result.rendered == 3, (
        f"e2e fixture expected 3 rendered samples, got {result.rendered} "
        f"(skipped={result.skipped} reasons={result.skip_reasons})"
    )

    images = sorted((out_dir / "images").glob("*.png"))
    assert [p.name for p in images] == [
        "0000000.png",
        "0000001.png",
        "0000002.png",
    ], f"unexpected image filenames: {[p.name for p in images]}"

    # Per-image PNG bytes — portable (no machine-dependent fields).
    for path in images:
        _check_pin(f"image:{path.name}", _file_sha256(path))

    # ``labels.json`` is portable (just filename → text); pin its
    # exact bytes including the trailing newline the writer emits.
    labels_path = out_dir / "labels.json"
    labels_bytes = labels_path.read_bytes()
    _check_pin("labels.json", _bytes_sha256(labels_bytes))

    # Sanity: the labels file we just hashed parses to the expected
    # dict shape. This means a hash collision would still fail loud
    # if someone managed to engineer one (vanishingly unlikely; this
    # is here so the test reads as a self-checking unit).
    parsed = json.loads(labels_bytes.decode("utf-8"))
    assert parsed == {
        "0000000.png": "mór",
        "0000001.png": "ḃoṫair",
        "0000002.png": "ḋuine",
    }


# ---------------------------------------------------------------------------
# Cross-call stability sanity — running the same recipe twice produces
# the same e2e digests. Belt-and-braces against a future writer / RNG
# refactor that introduces between-call non-determinism.
# ---------------------------------------------------------------------------


def test_visreg_e2e_digests_are_stable_across_two_runs(tmp_path: Path) -> None:
    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"
    run1_dir.mkdir()
    run2_dir.mkdir()
    recipe1 = _write_recipe(run1_dir)
    recipe2 = _write_recipe(run2_dir)

    out1 = run1_dir / "trainer-out"
    out2 = run2_dir / "trainer-out"

    for recipe, out in ((recipe1, out1), (recipe2, out2)):
        run_recipe(
            recipe,
            output_dir=out,
            count=3,
            seed=1,
            workers=1,
            progress=False,
            audit=False,
        )

    images1 = sorted((out1 / "images").glob("*.png"))
    images2 = sorted((out2 / "images").glob("*.png"))
    assert len(images1) == len(images2) == 3

    for a, b in zip(images1, images2, strict=True):
        assert _file_sha256(a) == _file_sha256(b), (
            f"e2e digest drift between two run_recipe invocations on "
            f"{a.name}: this means run_recipe's output is no longer "
            f"deterministic across invocations of the same recipe."
        )

    assert _bytes_sha256((out1 / "labels.json").read_bytes()) == _bytes_sha256(
        (out2 / "labels.json").read_bytes()
    )
