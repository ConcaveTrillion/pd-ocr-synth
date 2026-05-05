"""Per-render-run shared state: font cache + RNG.

A ``RenderContext`` is built once per render run (by the dataset
loop) and threaded into every per-sample call. Heavy state — opened
freetype faces, uharfbuzz faces — lives here so we don't reopen
the font for every sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from random import Random

import freetype
import uharfbuzz as hb


@dataclass(slots=True)
class _FontHandles:
    ft_face: freetype.Face
    hb_face: hb.Face


@dataclass
class RenderContext:
    """Shared per-run rendering state."""

    rng: Random
    _font_cache: dict[str, _FontHandles] = field(default_factory=dict)

    def font_handles(self, path: str | Path) -> _FontHandles:
        key = str(Path(path).resolve())
        cached = self._font_cache.get(key)
        if cached is not None:
            return cached
        ft_face = freetype.Face(key)
        with open(key, "rb") as fh:
            font_bytes = fh.read()
        hb_face = hb.Face(font_bytes)
        handles = _FontHandles(ft_face=ft_face, hb_face=hb_face)
        self._font_cache[key] = handles
        return handles

    @classmethod
    def for_seed(cls, seed: int) -> RenderContext:
        return cls(rng=Random(seed))
