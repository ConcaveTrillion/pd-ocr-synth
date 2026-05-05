#!/usr/bin/env python3
"""Generate the bundled CC0 paper textures used by ``recipes/gaelic.yaml``.

These are 100% procedurally generated — no scanned material, no
upstream IP. The output is dedicated to the public domain (CC0).

Re-running this script with the same defaults produces byte-identical
PNGs (np.random with a fixed seed). If you change the algorithm or
seeds, regenerate the bundled set in
``recipes/gaelic/textures/aged-paper/`` and update its ``LICENSES.md``
if attribution shifts.

Usage:
    uv run python scripts/generate-paper-textures.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _fbm(rng: np.random.Generator, shape: tuple[int, int], octaves: int = 5) -> np.ndarray:
    """Cheap value-noise fractal Brownian motion.

    Each octave samples a coarse uniform-random grid and bilinearly
    upsamples to ``shape``; octaves are summed with halving amplitude.
    Result is normalized to [0, 1].
    """

    h, w = shape
    out = np.zeros(shape, dtype=np.float32)
    amplitude = 1.0
    for octave in range(octaves):
        scale = 2 ** (octave + 1)
        coarse_h = max(2, h // scale)
        coarse_w = max(2, w // scale)
        coarse = rng.random((coarse_h, coarse_w), dtype=np.float32)
        # Upsample bilinearly via Pillow.
        layer = (
            np.asarray(
                Image.fromarray((coarse * 255).astype(np.uint8)).resize(
                    (w, h), resample=Image.Resampling.BILINEAR
                ),
                dtype=np.float32,
            )
            / 255.0
        )
        out += amplitude * layer
        amplitude *= 0.5
    out -= out.min()
    out /= max(1e-6, out.max())
    return out


def _aged_paper(seed: int, *, size: tuple[int, int] = (1024, 1024)) -> Image.Image:
    """One aged-paper texture.

    Composition:
    - Warm cream base color with slight per-pixel variation.
    - Two octaves of fBm tinting the brightness (mottled paper).
    - A faint vignette darkening the corners.
    - A sparse sprinkle of darker speckles (paper fiber).
    """

    rng = np.random.default_rng(seed)
    h, w = size

    base = np.tile(np.array([243, 232, 205], dtype=np.float32), (h, w, 1))
    fbm = _fbm(rng, (h, w), octaves=5)
    brightness = 0.78 + 0.22 * fbm  # range [0.78, 1.0]
    out = base * brightness[..., None]

    # Vignette.
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    radius = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    vignette = np.clip(1.0 - 0.18 * radius, 0.7, 1.0)
    out *= vignette[..., None]

    # Fiber speckles: a small fraction of pixels nudged darker.
    speckle_mask = rng.random((h, w)) > 0.997
    speckle_strength = rng.uniform(0.6, 0.85, size=(h, w)).astype(np.float32)
    out[speckle_mask] *= speckle_strength[speckle_mask, None]

    out = np.clip(out, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "recipes"
        / "gaelic"
        / "textures"
        / "aged-paper",
        help="Output directory.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=4,
        help="Number of textures to generate.",
    )
    parser.add_argument("--size", type=int, default=384, help="Output side length (square).")
    args = parser.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.count):
        img = _aged_paper(seed=1000 + i, size=(args.size, args.size))
        out_path = args.out_dir / f"aged-paper-{i + 1:02d}.png"
        img.save(out_path, format="PNG", optimize=True)
        print(f"wrote {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
