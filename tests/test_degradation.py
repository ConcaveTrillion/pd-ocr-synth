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
from pd_ocr_synth.render.sample import GlyphRun, LineBox, ParagraphBox, RenderedSample, WordBox

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

    Tolerance is generous (40 %\u201360 %) because the seeds are arbitrary
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


# ---------------------------------------------------------------------------
# M09: pixel-only stages must pass geometry through unchanged
# ---------------------------------------------------------------------------
#
# Per ``docs/roadmap/09-detection-mode.md`` (Bbox-aware degradation):
#
#   - Pixel-only stages pass bboxes through unchanged.
#   - Tests verify bbox round-trip on a fixed seed.
#
# We drive this directly off the live ``REGISTRY`` so that any future
# pixel-only stage added without thought to bbox/glyph_runs/word_boxes
# tracking still has to either (a) leave geometry untouched, or
# (b) get registered as a geometry stage and own that contract
# explicitly. A pixel stage that silently shifts text relative to its
# reported bboxes would break detection-mode annotations downstream;
# this test is the tripwire.


def _pixel_stage_options(
    kind: str, *, paper_texture_dir: Path | None = None
) -> dict[str, object] | None:
    """Options for ``kind`` that guarantee a non-noop application.

    Returning ``None`` means "no usable non-noop config without extra
    fixtures, skip"; the only such stage today is ``paper_texture``
    when ``paper_texture_dir`` is not supplied.

    Why force non-noop: a stage that becomes a noop (e.g. ``sigma=0``,
    ``factor=1.0``) trivially preserves every field — that's not what
    we want to lock down. We want to confirm that a stage which
    *actually* mutates pixels still leaves geometry alone.
    """

    if kind == "blur":
        return {"filter": "gaussian", "sigma": 1.5}
    if kind == "noise":
        return {"noise_kind": "gaussian", "stddev": 8}
    if kind == "brightness":
        return {"factor": 0.8}
    if kind == "contrast":
        return {"factor": 1.3}
    if kind == "gamma":
        return {"gamma": 1.4}
    if kind == "ink_bleed":
        return {"iterations": 1, "kernel_size_px": 1}
    if kind == "ink_thin":
        return {"iterations": 1, "kernel_size_px": 1}
    if kind == "jpeg":
        return {"quality": 60}
    if kind == "webp":
        return {"quality": 60}
    if kind == "grayscale":
        return {"method": "luminosity"}
    if kind == "foxing":
        return {"count": 3, "radius_px": 4, "color": [120, 60, 30], "opacity": 0.5}
    if kind == "paper_texture":
        if paper_texture_dir is None:
            return None
        return {
            "directory": str(paper_texture_dir),
            "blend": "multiply",
            "opacity": 0.4,
        }
    # Unknown pixel stage: no defaults, caller will skip.
    return None


def _make_textured_sample() -> RenderedSample:
    """Sample variant with a non-empty ``word_boxes`` and multi-cluster glyphs.

    The pixel-only invariant should hold for *every* geometry slot
    (``bbox``, ``glyph_runs``, ``word_boxes``), not just the top-level
    bbox. So the fixture has to populate all three.
    """

    width, height = 96, 32
    bg = (240, 235, 220)
    ink = (20, 18, 16)
    img = Image.new("RGB", (width, height), color=bg)
    pad = 4
    inked = Image.new("RGB", (width - 2 * pad, height - 2 * pad), color=ink)
    img.paste(inked, (pad, pad))
    bbox = (pad, pad, width - pad, height - pad)
    # Two synthetic word boxes + matching glyph runs. Coordinates are
    # arbitrary — they only need to round-trip byte-equal.
    word_boxes = (
        WordBox(text="alpha", bbox=(pad, pad, 40, height - pad)),
        WordBox(text="beta", bbox=(48, pad, width - pad, height - pad)),
    )
    glyph_runs = (
        GlyphRun(cluster=0, bbox=(pad, pad, 12, height - pad)),
        GlyphRun(cluster=1, bbox=(14, pad, 26, height - pad)),
        GlyphRun(cluster=2, bbox=(48, pad, 60, height - pad)),
        GlyphRun(cluster=3, bbox=(62, pad, 78, height - pad)),
    )
    return RenderedSample(
        text="alpha beta",
        image=img,
        bbox=bbox,
        font_path=None,  # type: ignore[arg-type]  # not used by pipeline
        font_size_pt=14.0,
        dpi=300,
        ink_color=ink,
        background_color=bg,
        glyph_runs=glyph_runs,
        word_boxes=word_boxes,
    )


def _registered_pixel_kinds() -> list[str]:
    # Force registration via a no-op call.
    apply_degradation(_make_sample(), [], rng=Random(0))
    return sorted(k for k, entry in REGISTRY.items() if entry.shape == "pixel")


def _build_paper_texture_dir(tmp_path: Path) -> Path:
    """Tiny synthetic texture so paper_texture has a dir to read from."""

    tex_dir = tmp_path / "textures-invariant"
    tex_dir.mkdir()
    rng_np = np.random.default_rng(0)
    arr = (rng_np.random((32, 32, 3)) * 200 + 30).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(tex_dir / "t0.png")
    return tex_dir


def test_pixel_stages_preserve_bbox_glyph_runs_and_word_boxes(tmp_path: Path) -> None:
    """Property test: every registered pixel stage leaves geometry untouched.

    Iterates the live ``REGISTRY``. For each pixel-shape entry we build
    a ``RenderedSample`` with non-empty ``bbox``, ``glyph_runs``, and
    ``word_boxes``, apply the stage with options that guarantee a real
    (non-noop) pixel mutation, and assert all geometry fields are
    byte-equal to the input. We also assert pixels actually changed —
    otherwise the test trivially passes via the noop path.
    """

    base = _make_textured_sample()
    base_pixels = _png_bytes(base.image)

    pixel_kinds = _registered_pixel_kinds()
    # Expect *all* M06 pixel stages plus foxing + paper_texture; if the
    # registry shrinks below this, we want a clear failure rather than
    # a silently shrunk property test.
    expected = {
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
        "foxing",
        "paper_texture",
    }
    assert expected <= set(pixel_kinds)

    paper_dir = _build_paper_texture_dir(tmp_path)
    seen_kinds: list[str] = []

    for kind in pixel_kinds:
        opts = _pixel_stage_options(kind, paper_texture_dir=paper_dir)
        if opts is None:
            pytest.fail(
                f"pixel stage {kind!r} has no non-noop options recipe in "
                f"_pixel_stage_options; add one so the bbox-invariant "
                f"property test covers it"
            )
        stage = _stage(kind, probability=1.0, **opts)
        out = apply_degradation(base, [stage], rng=Random(1234))

        # Geometry fields: byte-equal round-trip.
        assert out.bbox == base.bbox, f"{kind} mutated bbox"
        assert out.glyph_runs == base.glyph_runs, f"{kind} mutated glyph_runs"
        assert out.word_boxes == base.word_boxes, f"{kind} mutated word_boxes"
        assert out.image.size == base.image.size, f"{kind} mutated image size"

        # Sanity: the stage actually did something to the pixels.
        # (Otherwise the test is meaningless — a noop stage trivially
        # preserves every field.)
        assert _png_bytes(out.image) != base_pixels, (
            f"pixel stage {kind!r} did not modify image; "
            f"_pixel_stage_options must use a non-noop config"
        )
        seen_kinds.append(kind)

    # Belt-and-braces: confirm we exercised every pixel stage we found.
    assert sorted(seen_kinds) == pixel_kinds


def test_pixel_stages_preserve_geometry_with_empty_optional_fields() -> None:
    """The invariant must also hold when ``glyph_runs`` / ``word_boxes`` are empty.

    Word-crops mode (M07) does not populate ``word_boxes``; some
    callers may not populate ``glyph_runs`` either. Empty tuples must
    round-trip as empty tuples.
    """

    base = _make_sample()
    # Replace defaults with explicit empties to make the contract obvious.
    from dataclasses import replace as dc_replace

    base = dc_replace(base, glyph_runs=(), word_boxes=())

    stage = _stage("blur", probability=1.0, filter="gaussian", sigma=1.5)
    out = apply_degradation(base, [stage], rng=Random(0))

    assert out.glyph_runs == ()
    assert out.word_boxes == ()
    assert out.bbox == base.bbox


# ---------------------------------------------------------------------------
# M09: geometric stages must propagate the affine to every box collection
# ---------------------------------------------------------------------------
#
# Per ``docs/roadmap/09-detection-mode.md`` § "Residual M09 work →
# Geometric-stage detection-bbox propagation": ``_skew`` (and any
# future geometric stage) must rotate ``word_boxes`` / ``line_boxes`` /
# ``paragraph_boxes`` alongside ``bbox`` / ``glyph_runs``, otherwise a
# detection-mode recipe with `skew` enabled will write polygons that
# no longer match the rendered text. This is a complementary tripwire
# to the pixel-stage invariant test above.


def _make_paragraph_sample() -> RenderedSample:
    """Sample populated with all four bbox collections.

    Mimics what `paragraphs` / `pages` modes feed into the degradation
    pipeline: a paragraph containing two lines, each with two words,
    each word with one glyph cluster. Coordinates are arbitrary but
    consistent (every word fits inside its line, every line fits
    inside the paragraph, every box fits inside ``sample.bbox``) so
    the test can lock containment invariants pre- and post-skew.
    """

    width, height = 128, 64
    bg = (240, 235, 220)
    ink = (20, 18, 16)
    img = Image.new("RGB", (width, height), color=bg)
    pad = 8
    inked = Image.new("RGB", (width - 2 * pad, height - 2 * pad), color=ink)
    img.paste(inked, (pad, pad))
    bbox = (pad, pad, width - pad, height - pad)

    # Two lines vertically stacked inside the paragraph bbox.
    line0_bbox = (pad, pad, width - pad, pad + 24)
    line1_bbox = (pad, pad + 28, width - pad, height - pad)

    # Two words per line, side by side.
    word_a = WordBox(text="alpha", bbox=(pad, pad, pad + 48, pad + 24))
    word_b = WordBox(text="beta", bbox=(pad + 56, pad, width - pad, pad + 24))
    word_c = WordBox(text="gamma", bbox=(pad, pad + 28, pad + 56, height - pad))
    word_d = WordBox(text="delta", bbox=(pad + 64, pad + 28, width - pad, height - pad))

    line_boxes = (
        LineBox(text="alpha beta", bbox=line0_bbox),
        LineBox(text="gamma delta", bbox=line1_bbox),
    )
    paragraph_boxes = (ParagraphBox(text="alpha beta\ngamma delta", bbox=bbox),)
    glyph_runs = (
        GlyphRun(cluster=0, bbox=word_a.bbox),
        GlyphRun(cluster=1, bbox=word_b.bbox),
        GlyphRun(cluster=2, bbox=word_c.bbox),
        GlyphRun(cluster=3, bbox=word_d.bbox),
    )

    return RenderedSample(
        text="alpha beta\ngamma delta",
        image=img,
        bbox=bbox,
        font_path=None,  # type: ignore[arg-type]  # not used by pipeline
        font_size_pt=14.0,
        dpi=300,
        ink_color=ink,
        background_color=bg,
        glyph_runs=glyph_runs,
        word_boxes=(word_a, word_b, word_c, word_d),
        line_boxes=line_boxes,
        paragraph_boxes=paragraph_boxes,
    )


def _bbox_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], *, margin: int = 0
) -> bool:
    """True if ``inner`` lies inside ``outer`` (with ``margin`` slack)."""

    ox0, oy0, ox1, oy1 = outer
    ix0, iy0, ix1, iy1 = inner
    return (
        ix0 >= ox0 - margin and iy0 >= oy0 - margin and ix1 <= ox1 + margin and iy1 <= oy1 + margin
    )


def test_skew_propagates_to_every_box_collection() -> None:
    """``_skew`` must rotate every bbox collection, not just bbox+glyph_runs.

    Lock the M09 residual contract: a detection-mode recipe that
    enables `skew` must still emit per-word / per-line / per-paragraph
    polygons that line up with the rendered text. Each collection's
    bboxes should (a) survive (no entries dropped), (b) actually move
    (the affine ran), and (c) preserve the original containment
    relationships (words ⊆ lines ⊆ paragraphs ⊆ sample.bbox) within
    a small slack to absorb axis-aligned-bounding-of-rotated-quad
    rounding.
    """

    base = _make_paragraph_sample()
    stage = _stage("skew", probability=1.0, angle_deg=6.0, fill="background")
    out = apply_degradation(base, [stage], rng=Random(0))

    # (a) Survival: counts unchanged.
    assert len(out.word_boxes) == len(base.word_boxes)
    assert len(out.line_boxes) == len(base.line_boxes)
    assert len(out.paragraph_boxes) == len(base.paragraph_boxes)
    assert len(out.glyph_runs) == len(base.glyph_runs)

    # Texts round-trip — only bboxes should mutate.
    assert [w.text for w in out.word_boxes] == [w.text for w in base.word_boxes]
    assert [line.text for line in out.line_boxes] == [line.text for line in base.line_boxes]
    assert [p.text for p in out.paragraph_boxes] == [p.text for p in base.paragraph_boxes]

    # (b) Bboxes actually moved. PIL expand=True grows the canvas, so
    # *every* box should land at a new coordinate post-skew.
    assert out.image.size != base.image.size
    for old, new in zip(base.word_boxes, out.word_boxes, strict=True):
        assert old.bbox != new.bbox
    for old, new in zip(base.line_boxes, out.line_boxes, strict=True):
        assert old.bbox != new.bbox
    for old, new in zip(base.paragraph_boxes, out.paragraph_boxes, strict=True):
        assert old.bbox != new.bbox

    # (c) Containment relationships survive within a small slack. The
    # axis-aligned bbox of a rotated rectangle is strictly larger than
    # the rotated rectangle itself, so child boxes can spill past their
    # parents' axis-aligned hulls by a few pixels at small angles. Six
    # pixels is a generous upper bound for 6° on a 128 x 64 canvas.
    margin = 6
    paragraph_bbox = out.paragraph_boxes[0].bbox

    # word_boxes 0..1 belong to line 0; word_boxes 2..3 to line 1.
    line0, line1 = out.line_boxes
    word_to_line = {0: line0, 1: line0, 2: line1, 3: line1}
    for idx, word in enumerate(out.word_boxes):
        line = word_to_line[idx]
        assert _bbox_contains(line.bbox, word.bbox, margin=margin), (
            f"word {idx} {word.bbox!r} no longer inside line {line.bbox!r}"
        )

    for line in out.line_boxes:
        assert _bbox_contains(paragraph_bbox, line.bbox, margin=margin), (
            f"line {line.bbox!r} no longer inside paragraph {paragraph_bbox!r}"
        )

    # Sample.bbox should still enclose every word.
    for word in out.word_boxes:
        assert _bbox_contains(out.bbox, word.bbox, margin=margin), (
            f"word {word.bbox!r} no longer inside sample.bbox {out.bbox!r}"
        )

    # And every box must land on the new (expanded) canvas.
    nw, nh = out.image.size
    for collection in (out.word_boxes, out.line_boxes, out.paragraph_boxes):
        for entry in collection:
            x0, y0, x1, y1 = entry.bbox
            assert 0 <= x0 < x1 <= nw
            assert 0 <= y0 < y1 <= nh


def test_skew_zero_angle_preserves_every_box_collection() -> None:
    """Zero-angle ``_skew`` is a strict noop, including for new collections."""

    base = _make_paragraph_sample()
    stage = _stage("skew", probability=1.0, angle_deg=0.0)
    out = apply_degradation(base, [stage], rng=Random(0))
    assert out.bbox == base.bbox
    assert out.glyph_runs == base.glyph_runs
    assert out.word_boxes == base.word_boxes
    assert out.line_boxes == base.line_boxes
    assert out.paragraph_boxes == base.paragraph_boxes


def test_skew_preserves_empty_optional_box_collections() -> None:
    """``_skew`` on a word-crops-style sample must not invent box collections.

    M07 word-crops mode emits ``word_boxes`` / ``line_boxes`` /
    ``paragraph_boxes`` empty by construction (a single word *is* the
    sample). A geometric stage applied to such a sample must leave
    them empty — fabricating entries here would propagate downstream
    as bogus annotations.
    """

    base = _make_sample()  # default: empty word/line/paragraph collections
    assert base.word_boxes == ()
    assert base.line_boxes == ()
    assert base.paragraph_boxes == ()

    stage = _stage("skew", probability=1.0, angle_deg=4.0)
    out = apply_degradation(base, [stage], rng=Random(0))

    assert out.word_boxes == ()
    assert out.line_boxes == ()
    assert out.paragraph_boxes == ()
    # And the existing populated collections still got rotated.
    assert out.bbox != base.bbox
    assert out.glyph_runs != base.glyph_runs
