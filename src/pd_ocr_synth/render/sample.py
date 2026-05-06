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
class WordBox:
    """One word's text + tight pixel bbox within the sample image.

    Populated by multi-word layout modes (``lines``, ``paragraphs``,
    ``pages``). Empty for ``word_crops`` since the whole sample *is*
    the word.

    ``bbox`` is ``(x0, y0, x1, y1)`` in image-pixel coordinates,
    matching ``RenderedSample.bbox`` and ``GlyphRun.bbox``.
    """

    text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class LineBox:
    """One line's text + tight pixel bbox within the sample image.

    Populated by multi-line layout modes (``paragraphs``, ``pages``).
    Empty for ``word_crops`` and ``lines`` (a ``lines`` sample *is*
    a single line, so per-line ground truth would be redundant with
    ``RenderedSample.bbox``).

    ``bbox`` is ``(x0, y0, x1, y1)`` in image-pixel coordinates,
    matching ``RenderedSample.bbox``, ``WordBox.bbox`` and
    ``GlyphRun.bbox``.
    """

    text: str
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class RenderedSample:
    """Per-sample render output.

    ``image`` is a PIL Image (RGB). ``bbox`` is the tight inked box
    in pixel coordinates. ``glyph_runs`` carry per-cluster bounding
    boxes for downstream detection-mode use (M09).

    ``word_boxes`` carries per-word ground-truth (text + tight pixel
    bbox) for multi-word layout modes (``lines``, ``paragraphs``,
    ``pages``). It's empty for the ``word_crops`` layout where each
    sample is a single word.

    ``line_boxes`` carries per-line ground-truth for multi-line layout
    modes (``paragraphs``, ``pages``). It's empty for ``word_crops``
    and ``lines`` (those samples are a single line by construction;
    ``RenderedSample.bbox`` already covers the line).
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
    word_boxes: tuple[WordBox, ...] = field(default_factory=tuple)
    line_boxes: tuple[LineBox, ...] = field(default_factory=tuple)

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size  # type: ignore[no-any-return]
