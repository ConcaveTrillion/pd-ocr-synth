"""Tests for the M06 degradation pipeline.

Roadmap deliverables (``docs/roadmap/06-degradation.md`` "Tests"):

- One per stage: input shape preserved (or transformed correctly for
  geometric), determinism under seed.
- Pipeline composition: stages run in order, probabilities honored
  across many seeds.
- Bbox round-trip on ``skew``: rendered text remains inside the
  reported bbox after rotation.

The fixtures here build a synthetic ``RenderedSample`` directly — we
don't need a real font to exercise the degradation layer because the
renderer's output is just a PIL Image plus metadata.
"""

from __future__ import annotations

import io
from pathlib import Path
from random import Random

import numpy as np
import pytest
from PIL import Image

from pd_ocr_synth.degradation import REGISTRY, DegradationError, apply_degradation
from pd_ocr_synth.recipe.models import DegradationStage
from pd_ocr_synth.render.sample import GlyphRun, RenderedSample

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _make_sample(
    *,
    width: int = 64,
    height: int = 32,
    bg: tuple[int, int, int] = (240, 235, 220),
    ink: tuple[int, int, int] = (20, 18, 16),
) -> RenderedSample:
    """Build a synthetic word-crop with a dark rectangle (inked region).

    The "text" is just a 4-px-bordered solid block in the center; that's
    enough to drive the bbox-aware geometry tests, and pixel stages
    don't care what's in the image as long as it's RGB.
    """

    img = Image.new("RGB", (width, height), color=bg)
    pad = 4
    inked = Image.new("RGB", (width - 2 * pad, height - 2 * pad), color=ink)
    img.paste(inked, (pad, pad))
    bbox = (pad, pad, width - pad, height - pad)
    return RenderedSample(
        text="dummy",
        image=img,
        bbox=bbox,
        font_path=None,  # type: ignore[arg-type]  # not used by pipeline
        font_size_pt=14.0,
        dpi=300,
        ink_color=ink,
        background_color=bg,
        glyph_runs=(GlyphRun(cluster=0, bbox=bbox),),
    )


def _stage(kind: str, probability: float = 1.0, **opts) -> DegradationStage:
    return DegradationStage(kind=kind, probability=probability, **opts)


# ---------------------------------------------------------------------------
# Stage protocol & registry
# ---------------------------------------------------------------------------


def test_registry_covers_m06_minimum_set() -> None:
    """Per docs/roadmap/06-degradation.md "Built-in stages (M06 minimum)".

    These must all be wired by the time M06 lands.
    """

    # Importing the package side-effects the registration; force it.
    apply_degradation(_make_sample(), [], rng=Random(0))

    must_have = {
        "skew",
        "blur",
        "noise",
        "brightness",
        "contrast",
        "gamma",
        "ink_bleed",
        "ink_thin",
        "jpeg",
        "webp",
        "grayscale",
    }
    assert must_have <= set(REGISTRY)


def test_unknown_kind_raises_degradation_error() -> None:
    sample = _make_sample()
    bogus = DegradationStage(kind="not_a_real_stage", probability=1.0)
    with pytest.raises(DegradationError):
        apply_degradation(sample, [bogus], rng=Random(0))


# ---------------------------------------------------------------------------
# Per-stage: shape preserved, determinism under seed
# ---------------------------------------------------------------------------


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


@pytest.mark.parametrize(
    "stage",
    [
        _stage("blur", filter="gaussian", sigma={"min": 0.5, "max": 1.5}),
        _stage("blur", filter="motion", motion_length_px=5, motion_angle_deg=10),
        _stage("blur", filter="defocus", sigma={"min": 0.5, "max": 1.5}),
        _stage("noise", noise_kind="gaussian", stddev={"min": 2, "max": 8}),
        _stage("noise", noise_kind="salt_pepper", amount={"min": 0.01, "max": 0.02}),
        _stage("noise", noise_kind="poisson"),
        _stage("noise", noise_kind="speckle", stddev={"min": 2, "max": 8}),
        _stage("brightness", factor={"min": 0.85, "max": 1.15}),
        _stage("contrast", factor={"min": 0.8, "max": 1.2}),
        _stage("gamma", gamma={"min": 0.7, "max": 1.3}),
        _stage("ink_bleed", iterations=2, kernel_size_px=1),
        _stage("ink_thin", iterations=1, kernel_size_px=1),
        _stage("jpeg", quality={"min": 60, "max": 80}),
        _stage("webp", quality={"min": 60, "max": 80}),
        _stage("grayscale", method="luminosity"),
    ],
    ids=lambda s: (
        f"{s.kind}-{(s.model_extra or {}).get('filter') or (s.model_extra or {}).get('noise_kind') or (s.model_extra or {}).get('method') or 'default'}"
    ),
)
def test_pixel_stage_preserves_shape_and_is_deterministic(stage: DegradationStage) -> None:
    base = _make_sample()

    out_a = apply_degradation(base, [stage], rng=Random(123))
    out_b = apply_degradation(base, [stage], rng=Random(123))

    # Pixel stages must not change image dimensions or bbox.
    assert out_a.image.size == base.image.size
    assert out_a.bbox == base.bbox
    # Same seed → byte-identical PNG round-trip.
    assert _png_bytes(out_a.image) == _png_bytes(out_b.image)

    # And we should actually have done *something* (avoid the noop trap).
    if stage.kind not in {"grayscale"}:  # grayscale on grayscale would be noop; here ink is colored
        assert _png_bytes(out_a.image) != _png_bytes(base.image), (
            f"stage {stage.kind} produced no change"
        )


def test_skew_updates_bbox_and_keeps_text_inside() -> None:
    """The bbox round-trip test: text remains inside the reported bbox after rotation."""

    base = _make_sample()
    # Force a non-trivial rotation.
    stage = _stage("skew", probability=1.0, angle_deg=5.0, fill="background")
    out = apply_degradation(base, [stage], rng=Random(0))

    # Image should be larger (PIL expand=True grows the canvas).
    assert out.image.size != base.image.size
    # The new bbox lives in the new image's coord space.
    bx0, by0, bx1, by1 = out.bbox
    nw, nh = out.image.size
    assert 0 <= bx0 < bx1 <= nw
    assert 0 <= by0 < by1 <= nh

    # All inked pixels (anything substantially darker than the bg) must
    # fall inside the reported bbox. We allow a small margin to absorb
    # bilinear-resample edge softening.
    arr = np.asarray(out.image, dtype=np.int16)
    bg = np.asarray(base.background_color, dtype=np.int16)
    diff = np.abs(arr - bg).sum(axis=-1)
    inked = diff > 60  # threshold well below ink/bg distance, above resample bleed
    ys, xs = np.where(inked)

    margin = 2
    assert xs.min() >= bx0 - margin
    assert xs.max() <= bx1 + margin
    assert ys.min() >= by0 - margin
    assert ys.max() <= by1 + margin


def test_skew_zero_angle_is_noop() -> None:
    base = _make_sample()
    stage = _stage("skew", probability=1.0, angle_deg=0.0)
    out = apply_degradation(base, [stage], rng=Random(0))
    assert out.image.size == base.image.size
    assert out.bbox == base.bbox


def test_skew_glyph_runs_track_image_resize() -> None:
    base = _make_sample()
    stage = _stage("skew", probability=1.0, angle_deg=4.0)
    out = apply_degradation(base, [stage], rng=Random(0))
    nw, nh = out.image.size
    for run in out.glyph_runs:
        x0, y0, x1, y1 = run.bbox
        assert 0 <= x0 < x1 <= nw
        assert 0 <= y0 < y1 <= nh


# ---------------------------------------------------------------------------
# Pipeline composition
# ---------------------------------------------------------------------------


def test_pipeline_runs_stages_in_order() -> None:
    """Order matters. Reversing two non-commuting stages must change output.

    blur → ink_bleed and ink_bleed → blur produce different results
    because ``MinFilter`` (ink_bleed) is non-linear and a Gaussian
    blur after vs before changes which neighborhood minima the
    second stage sees.
    """

    base = _make_sample(width=80, height=40)
    blur = _stage("blur", filter="gaussian", sigma=1.5)
    bleed = _stage("ink_bleed", iterations=1, kernel_size_px=1)

    a = apply_degradation(base, [blur, bleed], rng=Random(0))
    b = apply_degradation(base, [bleed, blur], rng=Random(0))

    arr_a = np.asarray(a.image, dtype=np.int16)
    arr_b = np.asarray(b.image, dtype=np.int16)
    # We only need *some* difference; a single-pixel mismatch is enough.
    assert np.any(arr_a != arr_b)


def test_pipeline_probability_zero_skips_stage() -> None:
    base = _make_sample()
    # probability=0 must skip even at maximally skewed RNG.
    stage = _stage("blur", probability=0.0, filter="gaussian", sigma=2.0)
    out = apply_degradation(base, [stage], rng=Random(0))
    assert _png_bytes(out.image) == _png_bytes(base.image)


def test_pipeline_probability_one_always_applies() -> None:
    base = _make_sample()
    # Strong blur, probability=1.0 — must always apply, regardless of seed.
    stage = _stage("blur", probability=1.0, filter="gaussian", sigma=2.5)
    for seed in (0, 1, 7, 42, 9999):
        out = apply_degradation(base, [stage], rng=Random(seed))
        assert _png_bytes(out.image) != _png_bytes(base.image)


def test_pipeline_probability_honored_across_many_seeds() -> None:
    """Empirical: at p=0.5 across 1000 seeds the apply rate is ~50%.

    Tolerance is generous (40 %–60 %) because the seeds are arbitrary
    and we don't want a flaky test, just one that catches a stage that
    silently always applies or always skips.
    """

    base = _make_sample()
    base_bytes = _png_bytes(base.image)
    stage = _stage("blur", probability=0.5, filter="gaussian", sigma=2.5)

    applied = 0
    n = 1000
    for seed in range(n):
        out = apply_degradation(base, [stage], rng=Random(seed))
        if _png_bytes(out.image) != base_bytes:
            applied += 1
    rate = applied / n
    assert 0.40 <= rate <= 0.60, f"applied rate {rate:.3f} not within [0.40, 0.60]"


def test_paper_texture_blends_with_directory(tmp_path) -> None:
    # Build two synthetic textures so paper_texture has something to pick.
    tex_dir = tmp_path / "textures"
    tex_dir.mkdir()
    rng_np = np.random.default_rng(0)
    for i in range(2):
        arr = (rng_np.random((64, 64, 3)) * 200 + 30).astype(np.uint8)
        Image.fromarray(arr, mode="RGB").save(tex_dir / f"t{i}.png")

    base = _make_sample(width=80, height=40)
    stage = _stage(
        "paper_texture",
        probability=1.0,
        directory=str(tex_dir),
        blend="multiply",
        opacity=0.4,
    )

    out_a = apply_degradation(base, [stage], rng=Random(7))
    out_b = apply_degradation(base, [stage], rng=Random(7))

    # Shape preserved; image is changed; deterministic.
    assert out_a.image.size == base.image.size
    assert _png_bytes(out_a.image) != _png_bytes(base.image)
    assert _png_bytes(out_a.image) == _png_bytes(out_b.image)


def test_paper_texture_uses_bundled_aged_paper() -> None:
    """Smoke: the bundled CC0 textures load and blend without errors."""

    bundled = (
        Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "textures" / "aged-paper"
    )
    if not bundled.exists() or not any(bundled.glob("*.png")):
        pytest.skip("Bundled paper textures not present.")
    base = _make_sample()
    stage = _stage(
        "paper_texture",
        probability=1.0,
        directory=str(bundled),
        blend="multiply",
        opacity={"min": 0.2, "max": 0.6},
        scale={"min": 0.5, "max": 1.5},
        rotate_deg={"min": -180, "max": 180},
    )
    out = apply_degradation(base, [stage], rng=Random(42))
    assert out.image.size == base.image.size
    assert _png_bytes(out.image) != _png_bytes(base.image)


def test_paper_texture_missing_directory_raises(tmp_path) -> None:
    base = _make_sample()
    stage = _stage(
        "paper_texture",
        probability=1.0,
        directory=str(tmp_path / "nope"),
        blend="multiply",
        opacity=0.4,
    )
    with pytest.raises(ValueError, match="directory does not exist"):
        apply_degradation(base, [stage], rng=Random(0))


def test_foxing_zero_count_is_noop() -> None:
    base = _make_sample()
    stage = _stage("foxing", probability=1.0, count=0, radius_px=4, opacity=0.3)
    out = apply_degradation(base, [stage], rng=Random(0))
    assert _png_bytes(out.image) == _png_bytes(base.image)


def test_foxing_adds_visible_spots() -> None:
    base = _make_sample(width=120, height=80)
    stage = _stage(
        "foxing",
        probability=1.0,
        count=5,
        radius_px=4,
        color=[120, 60, 30],
        opacity=0.5,
    )
    out_a = apply_degradation(base, [stage], rng=Random(11))
    out_b = apply_degradation(base, [stage], rng=Random(11))

    assert out_a.image.size == base.image.size
    assert _png_bytes(out_a.image) != _png_bytes(base.image)
    assert _png_bytes(out_a.image) == _png_bytes(out_b.image)


def test_pipeline_is_deterministic_under_seed() -> None:
    base = _make_sample()
    stages = [
        _stage("skew", probability=0.5, angle_deg={"min": -2, "max": 2}),
        _stage("blur", probability=0.5, filter="gaussian", sigma={"min": 0.0, "max": 1.0}),
        _stage("noise", probability=0.5, noise_kind="gaussian", stddev={"min": 0, "max": 6}),
        _stage("jpeg", probability=0.5, quality={"min": 70, "max": 90}),
    ]
    a = apply_degradation(base, stages, rng=Random(2024))
    b = apply_degradation(base, stages, rng=Random(2024))
    assert _png_bytes(a.image) == _png_bytes(b.image)
    assert a.bbox == b.bbox
