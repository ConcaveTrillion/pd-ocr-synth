"""Built-in degradation stages.

Every M06 stage is implemented in pure Pillow + NumPy. No OpenCV
dependency — the M06 minimum set is fast enough on Pillow alone, and
the roadmap explicitly flags OpenCV as a possible-future-add (per
``docs/roadmap/06-degradation.md`` "Risks / open items").

Sampling helpers from ``pd_ocr_synth.render.sampling`` are reused
verbatim — degradation options (sigma, factor, opacity, etc.) follow
the same scalar / range / weighted-choice contract as rendering.

All draws flow through the ``rng`` argument so determinism contracts
upstream are preserved end-to-end.
"""

from __future__ import annotations

import io
from random import Random
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from pd_ocr_synth.degradation.pipeline import (
    register_geometry_stage,
    register_pixel_stage,
)

if TYPE_CHECKING:
    from pd_ocr_synth.render.sample import RenderedSample


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _draw(value: Any, rng: Random) -> float:
    """Draw a numeric value supporting scalar / range-dict / weighted-choice list.

    Degradation options come straight from the recipe YAML, so they
    are still raw Python primitives at this point — no Pydantic
    coercion has happened (the M02 ``DegradationStage`` model uses
    ``extra="allow"`` and does not type-check stage-internal fields).
    We accept the three forms by their runtime shape:

    - plain ``int`` / ``float`` returned as-is.
    - ``{"min": x, "max": y}`` mapping → uniform draw.
    - ``[{"value": v, "weight": w}, ...]`` list → weighted choice.
    """

    if isinstance(value, dict) and "min" in value and "max" in value:
        lo = value["min"]
        hi = value["max"]
        if isinstance(lo, int) and isinstance(hi, int):
            return float(rng.randint(lo, hi))
        return rng.uniform(float(lo), float(hi))
    if isinstance(value, list) and value and isinstance(value[0], dict) and "value" in value[0]:
        weights = [max(float(c.get("weight", 1.0)), 0.0) for c in value]
        total = sum(weights)
        if total <= 0:
            return float(rng.choice(value)["value"])
        pick = rng.uniform(0.0, total)
        acc = 0.0
        for choice, weight in zip(value, weights, strict=True):
            acc += weight
            if pick <= acc:
                return float(choice["value"])
        return float(value[-1]["value"])
    return float(value)


def _draw_int(value: Any, rng: Random) -> int:
    return int(round(_draw(value, rng)))


def _to_rgb(image: Image.Image) -> Image.Image:
    """Force RGB mode. Stages that round-trip through NumPy expect three channels."""

    if image.mode != "RGB":
        return image.convert("RGB")
    return image


# ---------------------------------------------------------------------------
# Optical
# ---------------------------------------------------------------------------


def _blur(image: Image.Image, options: dict[str, Any], rng: Random) -> Image.Image:
    filter_kind = str(options.get("filter", "gaussian"))
    if filter_kind == "gaussian":
        sigma = _draw(options.get("sigma", 0.0), rng)
        if sigma <= 0:
            return image
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))
    if filter_kind == "motion":
        # Pillow has no built-in motion blur; approximate with a
        # 1-D box blur in the requested direction.
        length = _draw_int(options.get("motion_length_px", 0), rng)
        if length <= 0:
            return image
        angle = _draw(options.get("motion_angle_deg", 0.0), rng)
        # Build a 1-D kernel of ``length`` ones (normalized) rotated to
        # ``angle``. For small lengths the rotation is approximate
        # enough; we skip the bilinear resample and just use Pillow's
        # ``Kernel`` filter on a square.
        ksize = max(3, (length // 2) * 2 + 1)  # odd, >= 3
        kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2
        rad = np.deg2rad(angle)
        dx = float(np.cos(rad))
        dy = float(np.sin(rad))
        for t in range(-center, center + 1):
            x = int(round(center + t * dx))
            y = int(round(center + t * dy))
            if 0 <= x < ksize and 0 <= y < ksize:
                kernel[y, x] = 1.0
        if kernel.sum() == 0:
            return image
        kernel /= kernel.sum()
        flat = tuple(float(v) for v in kernel.ravel())
        return image.filter(ImageFilter.Kernel(size=(ksize, ksize), kernel=flat, scale=1.0))
    if filter_kind == "defocus":
        # Approximate disk defocus with BoxBlur (uniform low-pass).
        radius = _draw(options.get("sigma", 0.0), rng)
        if radius <= 0:
            return image
        return image.filter(ImageFilter.BoxBlur(radius=radius))
    raise ValueError(f"blur: unknown filter {filter_kind!r}")


def _noise(image: Image.Image, options: dict[str, Any], rng: Random) -> Image.Image:
    kind = str(options.get("noise_kind", "gaussian"))
    img = _to_rgb(image)
    arr = np.asarray(img, dtype=np.int16)

    # NumPy's per-stage RNG: seeded from the per-sample stdlib ``rng``
    # so the noise pattern is reproducible without leaking arbitrary
    # state between stages.
    np_rng = np.random.default_rng(rng.getrandbits(64))

    if kind == "gaussian":
        stddev = _draw(options.get("stddev", 0.0), rng)
        if stddev <= 0:
            return image
        noise = np_rng.normal(0.0, stddev, size=arr.shape)
        out = np.clip(arr + noise, 0, 255).astype(np.uint8)
    elif kind == "salt_pepper":
        amount = _draw(options.get("amount", 0.0), rng)
        if amount <= 0:
            return image
        out = arr.copy().astype(np.uint8)
        mask = np_rng.random(out.shape[:2])
        salt = mask < amount / 2
        pepper = (mask >= amount / 2) & (mask < amount)
        out[salt] = 255
        out[pepper] = 0
    elif kind == "poisson":
        # Shot noise scaled so it's visible at typical 8-bit ranges.
        # Lambda of pixel value reproduces the underlying photon
        # statistics without amplifying the noise to the point where
        # text becomes unreadable.
        out = np_rng.poisson(np.clip(arr, 0, 255).astype(np.float32)).clip(0, 255).astype(np.uint8)
    elif kind == "speckle":
        stddev = _draw(options.get("stddev", 0.0), rng)
        if stddev <= 0:
            return image
        speckle = np_rng.normal(0.0, stddev / 255.0, size=arr.shape)
        out = np.clip(arr + arr * speckle, 0, 255).astype(np.uint8)
    else:
        raise ValueError(f"noise: unknown noise_kind {kind!r}")

    return Image.fromarray(out, mode="RGB")


def _brightness(image: Image.Image, options: dict[str, Any], rng: Random) -> Image.Image:
    factor = _draw(options.get("factor", 1.0), rng)
    if factor == 1.0:
        return image
    return ImageEnhance.Brightness(_to_rgb(image)).enhance(factor)


def _contrast(image: Image.Image, options: dict[str, Any], rng: Random) -> Image.Image:
    factor = _draw(options.get("factor", 1.0), rng)
    if factor == 1.0:
        return image
    return ImageEnhance.Contrast(_to_rgb(image)).enhance(factor)


def _gamma(image: Image.Image, options: dict[str, Any], rng: Random) -> Image.Image:
    gamma = _draw(options.get("gamma", 1.0), rng)
    if gamma <= 0 or gamma == 1.0:
        return image
    img = _to_rgb(image)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    out = np.power(arr, 1.0 / gamma)
    return Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8), mode="RGB")


# ---------------------------------------------------------------------------
# Print / paper (ink only — paper_texture + foxing land alongside the
# bundled CC0 textures in commit C)
# ---------------------------------------------------------------------------


def _ink_bleed(image: Image.Image, options: dict[str, Any], rng: Random) -> Image.Image:
    """Dilate dark ink: simulate spread.

    Implemented as a ``MinFilter`` on the RGB image. Because a ``MinFilter``
    over RGB takes the per-channel minimum in the neighborhood, ink
    pixels (low values) grow outward into background — which is exactly
    what physical ink bleed looks like.
    """

    iterations = _draw_int(options.get("iterations", 1), rng)
    kernel_size = _draw_int(options.get("kernel_size_px", 1), rng)
    if iterations <= 0 or kernel_size <= 0:
        return image
    size = max(1, 2 * kernel_size + 1)  # MinFilter wants odd
    out = _to_rgb(image)
    for _ in range(iterations):
        out = out.filter(ImageFilter.MinFilter(size))
    return out


def _ink_thin(image: Image.Image, options: dict[str, Any], rng: Random) -> Image.Image:
    """Erode dark ink: simulate under-inking."""

    iterations = _draw_int(options.get("iterations", 1), rng)
    kernel_size = _draw_int(options.get("kernel_size_px", 1), rng)
    if iterations <= 0 or kernel_size <= 0:
        return image
    size = max(1, 2 * kernel_size + 1)
    out = _to_rgb(image)
    for _ in range(iterations):
        out = out.filter(ImageFilter.MaxFilter(size))
    return out


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


def _jpeg(image: Image.Image, options: dict[str, Any], rng: Random) -> Image.Image:
    quality = max(1, min(100, _draw_int(options.get("quality", 85), rng)))
    chroma_opt = options.get("chroma_subsampling", "random")
    if chroma_opt == "random":
        chroma = rng.choice([0, 2])  # 4:4:4 (0) or 4:2:0 (2)
    elif chroma_opt == "4:4:4":
        chroma = 0
    elif chroma_opt == "4:2:0":
        chroma = 2
    else:
        raise ValueError(f"jpeg: unknown chroma_subsampling {chroma_opt!r}")

    buf = io.BytesIO()
    _to_rgb(image).save(buf, format="JPEG", quality=quality, subsampling=chroma)
    buf.seek(0)
    return Image.open(buf).copy()


def _webp(image: Image.Image, options: dict[str, Any], rng: Random) -> Image.Image:
    quality = max(1, min(100, _draw_int(options.get("quality", 85), rng)))
    buf = io.BytesIO()
    _to_rgb(image).save(buf, format="WEBP", quality=quality)
    buf.seek(0)
    return Image.open(buf).copy()


# ---------------------------------------------------------------------------
# Color space
# ---------------------------------------------------------------------------


def _grayscale(image: Image.Image, options: dict[str, Any], rng: Random) -> Image.Image:
    """Project to grayscale, return as RGB.

    The image stays an RGB-mode PIL image throughout the pipeline so
    later stages can keep doing channel arithmetic. We just collapse
    R / G / B to identical values via the requested method.
    """

    del rng  # deterministic
    method = str(options.get("method", "luminosity"))
    img = _to_rgb(image)
    arr = np.asarray(img, dtype=np.float32)
    if method == "luminosity":
        gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    elif method == "average":
        gray = arr.mean(axis=-1)
    elif method == "red":
        gray = arr[..., 0]
    elif method == "green":
        gray = arr[..., 1]
    elif method == "blue":
        gray = arr[..., 2]
    else:
        raise ValueError(f"grayscale: unknown method {method!r}")
    out = np.stack([gray] * 3, axis=-1).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


# ---------------------------------------------------------------------------
# Geometry-aware (skew)
# ---------------------------------------------------------------------------


def _skew(sample: RenderedSample, options: dict[str, Any], rng: Random) -> RenderedSample:
    """Affine rotation around the image center; updates bbox / glyph_runs.

    The new image is sized to fit the rotated content (PIL's
    ``expand=True``), so the bbox is translated by the same offset the
    rotation applied. Glyph runs follow the same rotation matrix so
    detection-mode bbox tracking still works downstream.
    """

    from dataclasses import replace as dc_replace

    angle = _draw(options.get("angle_deg", 0.0), rng)
    if angle == 0:
        return sample

    fill = _resolve_fill(options.get("fill", "background"), sample)
    img = _to_rgb(sample.image)
    new_image = img.rotate(
        angle,
        resample=Image.Resampling.BILINEAR,
        expand=True,
        fillcolor=fill,
    )

    # PIL rotates around the image center and expands the canvas; the
    # new center is shifted by ``((new_w - old_w) / 2, (new_h - old_h) / 2)``.
    old_w, old_h = img.size
    new_w, new_h = new_image.size
    dx = (new_w - old_w) / 2.0
    dy = (new_h - old_h) / 2.0
    cx, cy = old_w / 2.0, old_h / 2.0

    rad = np.deg2rad(-angle)  # PIL rotates CCW for positive angle; image-coord rotation is CW.
    cos_a = float(np.cos(rad))
    sin_a = float(np.sin(rad))

    def _rotate_box(box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        # Rotate four corners around the old center, then translate by
        # the expand offset to land in the new canvas.
        x0, y0, x1, y1 = box
        corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        rotated_xs: list[float] = []
        rotated_ys: list[float] = []
        for x, y in corners:
            rx = cos_a * (x - cx) - sin_a * (y - cy) + cx + dx
            ry = sin_a * (x - cx) + cos_a * (y - cy) + cy + dy
            rotated_xs.append(rx)
            rotated_ys.append(ry)
        # Axis-aligned bbox of the rotated quad.
        return (
            int(round(min(rotated_xs))),
            int(round(min(rotated_ys))),
            int(round(max(rotated_xs))),
            int(round(max(rotated_ys))),
        )

    new_bbox = _rotate_box(sample.bbox)
    new_runs = tuple(dc_replace(g, bbox=_rotate_box(g.bbox)) for g in sample.glyph_runs)
    return dc_replace(sample, image=new_image, bbox=new_bbox, glyph_runs=new_runs)


def _resolve_fill(fill: Any, sample: RenderedSample) -> tuple[int, int, int]:
    if fill == "background":
        return tuple(int(c) for c in sample.background_color)  # type: ignore[return-value]
    if fill == "white":
        return (255, 255, 255)
    if fill == "black":
        return (0, 0, 0)
    if fill == "transparent":  # we render RGB; treat as white to avoid mode flip
        return (255, 255, 255)
    if isinstance(fill, list | tuple) and len(fill) == 3:
        return (int(fill[0]), int(fill[1]), int(fill[2]))
    raise ValueError(f"skew: unknown fill {fill!r}")


# ---------------------------------------------------------------------------
# registration
# ---------------------------------------------------------------------------

register_pixel_stage("blur", _blur)
register_pixel_stage("noise", _noise)
register_pixel_stage("brightness", _brightness)
register_pixel_stage("contrast", _contrast)
register_pixel_stage("gamma", _gamma)
register_pixel_stage("ink_bleed", _ink_bleed)
register_pixel_stage("ink_thin", _ink_thin)
register_pixel_stage("jpeg", _jpeg)
register_pixel_stage("webp", _webp)
register_pixel_stage("grayscale", _grayscale)

register_geometry_stage("skew", _skew)
