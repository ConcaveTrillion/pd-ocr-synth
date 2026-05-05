"""Tests for ``pd_ocr_synth.tokenization``."""

from __future__ import annotations

import pytest

from pd_ocr_synth.tokenization import tokenize

# ---------------------------------------------------------------------------
# word_crops
# ---------------------------------------------------------------------------


def test_word_crops_splits_on_whitespace() -> None:
    assert tokenize("the quick brown fox", mode="word_crops") == [
        "the",
        "quick",
        "brown",
        "fox",
    ]


def test_word_crops_strips_edge_punctuation() -> None:
    assert tokenize('"hello," she said.', mode="word_crops") == ["hello", "she", "said"]


def test_word_crops_preserves_internal_punctuation() -> None:
    out = tokenize("don't twenty-one Sgéal Féin", mode="word_crops")
    assert "don't" in out
    assert "twenty-one" in out
    assert "Sgéal" in out


def test_word_crops_drops_pure_punctuation_tokens() -> None:
    assert tokenize("--- ... !!", mode="word_crops") == []


def test_word_crops_handles_unicode_apostrophe() -> None:
    out = tokenize("d’fhág na fir", mode="word_crops")
    # The fancy apostrophe is internal and stays attached to its word.
    assert "d’fhág" in out


def test_word_crops_skips_empty_input() -> None:
    assert tokenize("   \n\n  ", mode="word_crops") == []


# ---------------------------------------------------------------------------
# lines
# ---------------------------------------------------------------------------


def test_lines_split_on_newline() -> None:
    out = tokenize("first line\nsecond line\nthird", mode="lines")
    assert out == ["first line", "second line", "third"]


def test_lines_drops_blank_lines() -> None:
    out = tokenize("a\n\nb\n   \nc\n", mode="lines")
    assert out == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# paragraphs / pages
# ---------------------------------------------------------------------------


def test_paragraphs_split_on_blank_line() -> None:
    text = "para 1 line a\npara 1 line b\n\npara 2\n\n\npara 3\n"
    assert tokenize(text, mode="paragraphs") == [
        "para 1 line a\npara 1 line b",
        "para 2",
        "para 3",
    ]


def test_paragraphs_drops_empty_after_strip() -> None:
    text = "  \n\nmiddle\n\n   \n\n"
    assert tokenize(text, mode="paragraphs") == ["middle"]


def test_pages_mode_currently_aliases_paragraphs() -> None:
    text = "a\n\nb"
    assert tokenize(text, mode="pages") == tokenize(text, mode="paragraphs")


# ---------------------------------------------------------------------------
# error path
# ---------------------------------------------------------------------------


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown layout"):
        tokenize("x", mode="ranchwords")  # type: ignore[arg-type]
