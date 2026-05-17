"""Per-render-run shared state: font cache + RNG.

A ``RenderContext`` is built once per render run (by the dataset
loop) and threaded into every per-sample call. Heavy state — opened
freetype faces, uharfbuzz faces — lives here so we don't reopen
the font for every sample.

Determinism contract (per ``docs/roadmap/05-rendering.md``): all
randomness flows from the recipe's ``seed`` through a single
``Random`` per render run, **branched per sample by sample index**.
Same recipe + seed + sample index → identical output bytes. The
loop is expected to call ``ctx.reseed_for_sample(i)`` before each
``render_word_crop`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from random import Random

import freetype
import uharfbuzz as hb


@dataclass(slots=True)
class _FontHandles:
    ft_face: freetype.Face  # type: ignore[attr-defined]
    hb_face: hb.Face  # type: ignore[attr-defined]


def branched_seed(base_seed: int, sample_index: int) -> int:
    """Derive a stable per-sample seed from the recipe seed + index.

    Same constants as the text-transform pipeline's ``_step_seed`` —
    a Knuth/SplitMix-style mix that makes adjacent indices produce
    well-separated streams.
    """

    return (base_seed + 1) * 0x9E3779B1 ^ (sample_index * 0x85EBCA77)


@dataclass
class RenderContext:
    """Shared per-run rendering state."""

    rng: Random
    seed: int = 0
    _font_cache: dict[str, _FontHandles] = field(default_factory=dict)

    def font_handles(self, path: str | Path) -> _FontHandles:
        key = str(Path(path).resolve())
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        ft_face = freetype.Face(key)
        with open(key, "rb") as fh:
            font_bytes = fh.read()
        hb_face = hb.Face(font_bytes)  # type: ignore[attr-defined]
        handles = _FontHandles(ft_face=ft_face, hb_face=hb_face)
        self._font_cache[key] = handles
        return handles

    def reseed_for_sample(self, sample_index: int) -> None:
        """Replace ``self.rng`` with a per-sample-index branched stream.

        Called by the dataset loop before each render. The font cache
        is preserved across calls; only the RNG changes.
        """

        self.rng = Random(branched_seed(self.seed, sample_index))

    @classmethod
    def for_seed(cls, seed: int) -> RenderContext:
        return cls(rng=Random(seed), seed=seed)
