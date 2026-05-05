"""Font inspection — codepoint coverage and metadata.

Wraps ``freetype-py`` so the rest of the package never imports it
directly. Today we only need read-only inspection (open, list
covered codepoints, check OpenType feature presence). Rendering /
shaping happens in M05 with HarfBuzz on top of the same Face.

Note: ``Face.family_name`` and ``style_name`` come back as ``bytes``
from freetype-py. We decode to str at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import freetype


@dataclass(frozen=True, slots=True)
class FontInfo:
    """Read-only summary of one font face."""

    path: Path
    family: str
    style: str
    num_glyphs: int
    codepoints: frozenset[int] = field(default_factory=frozenset)

    def covers(self, codepoint: int | str) -> bool:
        cp = ord(codepoint) if isinstance(codepoint, str) else int(codepoint)
        return cp in self.codepoints

    def missing(self, text: str) -> set[int]:
        """Codepoints from ``text`` that this font does not cover."""
        return {ord(ch) for ch in text if ord(ch) not in self.codepoints}

    def coverage(self, text: str) -> tuple[int, int]:
        """Return (covered, total_unique) for ``text``."""
        unique = {ord(ch) for ch in text}
        if not unique:
            return 0, 0
        covered = sum(1 for cp in unique if cp in self.codepoints)
        return covered, len(unique)


class FontOpenError(Exception):
    """Raised when freetype-py cannot open the font file."""


def open_font(path: str | Path) -> FontInfo:
    """Open a font for inspection.

    Reads every cmap entry into a frozenset so callers can do cheap
    repeated coverage checks across many tokens / lines without
    paying the freetype cost each time.
    """

    p = Path(path)
    try:
        face = freetype.Face(str(p))
    except freetype.FT_Exception as exc:  # pragma: no cover - environment-specific
        raise FontOpenError(f"could not open font {p}: {exc}") from exc
    except OSError as exc:
        raise FontOpenError(f"could not open font {p}: {exc}") from exc

    return FontInfo(
        path=p,
        family=_decode(face.family_name),
        style=_decode(face.style_name),
        num_glyphs=face.num_glyphs,
        codepoints=frozenset(_iter_codepoints(face)),
    )


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin-1", errors="replace")
    return value


def _iter_codepoints(face: freetype.Face):
    """Yield every codepoint in the font's cmap.

    freetype-py's ``get_first_char`` / ``get_next_char`` walk the
    selected charmap. We use the default (Unicode if present).
    """

    charcode, glyph_index = face.get_first_char()
    while glyph_index:
        yield charcode
        charcode, glyph_index = face.get_next_char(charcode, glyph_index)
