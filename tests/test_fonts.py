"""Tests for ``pd_ocr_synth.fonts``.

Uses the real Gaelic fonts that the bundled fetch script downloads
when available. Tests skip gracefully when the fonts have not been
fetched (e.g. CI / fresh clones), so the suite stays portable.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.fonts import FontInfo, FontOpenError, open_font

FONT_DIR = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not FONT_DIR.exists():
        pytest.skip("Gaelic font not fetched. Run ./scripts/fetch-fonts-gaelic.sh to enable.")
    return FONT_DIR


def test_open_real_font_returns_metadata() -> None:
    info = open_font(_require_font())
    assert isinstance(info, FontInfo)
    assert info.num_glyphs > 0
    assert info.family
    assert info.style


def test_real_font_covers_gaelic_glyphs() -> None:
    info = open_font(_require_font())
    for ch in "ḃċċḋḟġṁṗṡṫ⁊":
        assert info.covers(ch), f"font missing required glyph: U+{ord(ch):04X}"


def test_covers_accepts_int_or_str() -> None:
    info = open_font(_require_font())
    assert info.covers("a") == info.covers(ord("a"))


def test_missing_returns_codepoints_not_in_font() -> None:
    info = open_font(_require_font())
    absent = next(cp for cp in range(0x10F000, 0x10FFFD) if cp not in info.codepoints)
    missing = info.missing(chr(absent))
    assert absent in missing
    assert ord("a") not in info.missing("a")


def test_coverage_returns_two_tuple() -> None:
    info = open_font(_require_font())
    covered, total = info.coverage("abc")
    assert covered == total == 3
    covered, total = info.coverage("")
    assert (covered, total) == (0, 0)


def test_open_nonexistent_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FontOpenError):
        open_font(tmp_path / "no-such-font.otf")


def test_open_garbage_file_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.otf"
    bad.write_bytes(b"not a font, just some bytes\n" * 16)
    with pytest.raises(FontOpenError):
        open_font(bad)
