"""Ground-truth payload returned per rendered sample."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True, slots=True)
class GlyphRun:
    """One shaped cluster within a sample."""

    cluster: int
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class RenderedSample:
    """Per-sample render output.

    ``image`` is a PIL Image (RGB). ``bbox`` is the tight inked box
    in pixel coordinates. ``glyph_runs`` carry per-cluster bounding
    boxes for downstream detection-mode use (M09).
    """

    text: str
    image: Image
    bbox: tuple[int, int, int, int]
    font_path: Path
    font_size_pt: float
    dpi: int
    ink_color: tuple[int, int, int]
    background_color: tuple[int, int, int]
    glyph_runs: tuple[GlyphRun, ...] = field(default_factory=tuple)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size  # type: ignore[no-any-return]
