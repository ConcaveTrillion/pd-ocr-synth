"""Tests for ``pd_ocr_synth.fonts``.

Uses the real Gaelic fonts that the bundled fetch script downloads
when available. Tests skip gracefully when the fonts have not been
fetched (e.g. CI / fresh clones), so the suite stays portable.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from pd_ocr_synth.fonts import FontInfo, FontOpenError, _decode, open_font

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


# ---------------------------------------------------------------------------
# Pure-Python invariants — no real font required
# ---------------------------------------------------------------------------
#
# These lock behaviour of the small helpers + ``FontInfo`` dataclass so
# refactors of ``fonts.py`` (e.g. swapping the cmap walker, adding a
# face cache, or porting the metadata decoder) cannot regress:
#
# * ``FontInfo`` is frozen — accidental mutation by a downstream consumer
#   (e.g. a renderer trying to "patch in" extra codepoints) raises
#   instead of silently de-syncing the cmap snapshot from the on-disk
#   font.
# * ``covers`` accepts both ``int`` and single-character ``str`` with
#   identical semantics; ``int`` codepoint 0 is honoured.
# * ``missing`` and ``coverage`` collapse duplicates so a paragraph of
#   ``"aaaa"`` reports the same coverage as ``"a"``.
# * ``_decode`` round-trips ``str`` unchanged, falls back to latin-1
#   with replacement on non-UTF-8 bytes, and treats ``None`` as the
#   empty string.


def _make_info(codepoints: frozenset[int]) -> FontInfo:
    return FontInfo(
        path=Path("/synthetic"),
        family="Synthetic",
        style="Regular",
        num_glyphs=len(codepoints),
        codepoints=codepoints,
    )


def test_font_info_is_frozen() -> None:
    info = _make_info(frozenset({0x41}))
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.family = "Mutated"  # type: ignore[misc]


def test_covers_handles_int_zero() -> None:
    info = _make_info(frozenset({0, 0x41}))
    assert info.covers(0) is True
    # And a codepoint not in the set.
    assert info.covers(0xFFFD) is False


def test_covers_str_and_int_agree_for_high_codepoints() -> None:
    cp = 0x1F600  # outside BMP
    info = _make_info(frozenset({cp}))
    assert info.covers(chr(cp)) is True
    assert info.covers(cp) is True


def test_covers_multichar_string_raises_typeerror() -> None:
    """``covers`` is documented as ``int | str`` (single codepoint).

    Multi-character strings raise ``TypeError`` from ``ord()``. We lock
    this so callers can't quietly "succeed" on the first char of a
    longer string — they must use ``missing`` / ``coverage`` for that.
    """

    info = _make_info(frozenset({0x41}))
    with pytest.raises(TypeError):
        info.covers("ab")


def test_missing_collapses_duplicates() -> None:
    info = _make_info(frozenset({0x41}))  # only 'A'
    # Three 'b's, all missing — should report a single codepoint.
    assert info.missing("bbb") == {ord("b")}


def test_missing_empty_text_returns_empty_set() -> None:
    info = _make_info(frozenset({0x41}))
    assert info.missing("") == set()


def test_missing_all_covered_returns_empty_set() -> None:
    info = _make_info(frozenset({ord("a"), ord("b"), ord("c")}))
    assert info.missing("abcabc") == set()


def test_coverage_collapses_duplicates_to_unique_count() -> None:
    info = _make_info(frozenset({ord("a")}))
    covered, total = info.coverage("aaaa")
    assert (covered, total) == (1, 1)


def test_coverage_partial_match() -> None:
    info = _make_info(frozenset({ord("a"), ord("b")}))
    covered, total = info.coverage("abcd")
    assert (covered, total) == (2, 4)


def test_coverage_empty_text_returns_zero_zero() -> None:
    info = _make_info(frozenset({ord("a")}))
    assert info.coverage("") == (0, 0)


def test_coverage_no_overlap_returns_zero_with_unique_count() -> None:
    info = _make_info(frozenset({ord("a")}))
    covered, total = info.coverage("xyz")
    assert (covered, total) == (0, 3)


def test_decode_passes_through_str() -> None:
    assert _decode("Already a str") == "Already a str"


def test_decode_returns_empty_for_none() -> None:
    assert _decode(None) == ""


def test_decode_uses_utf8_for_valid_bytes() -> None:
    # "Caoláin" — a name with U+00E1 á encoded as UTF-8.
    assert _decode("Caoláin".encode()) == "Caoláin"


def test_decode_falls_back_to_latin1_for_invalid_utf8() -> None:
    # 0xE1 alone is invalid UTF-8 but valid latin-1 ("á").
    decoded = _decode(b"Caol\xe1in")
    # latin-1 decode is total — no substitution should land on the
    # primary path. (We use ``errors="replace"`` defensively in case a
    # future Python tightens latin-1 decoding, so just check the
    # round-trip preserves the printable ASCII bookends.)
    assert decoded.startswith("Caol")
    assert decoded.endswith("in")


def test_decode_handles_empty_bytes() -> None:
    assert _decode(b"") == ""
