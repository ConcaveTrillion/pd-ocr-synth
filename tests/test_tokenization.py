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


def test_pages_mode_keeps_paragraphs_glued_until_triple_newline() -> None:
    """Pages mode splits on a triple-blank-line boundary, not the
    paragraph boundary used by ``paragraphs`` mode.

    A page chunk preserves its inner ``\\n\\n``-separated paragraphs
    so the renderer can compose multi-paragraph pages. The triple
    newline is the only signal that demotes us to a new page-sized
    sample.
    """
    text = "para 1 line a\npara 1 line b\n\npara 2\n\n\npage 2 para 1\n\npage 2 para 2\n"
    assert tokenize(text, mode="pages") == [
        "para 1 line a\npara 1 line b\n\npara 2",
        "page 2 para 1\n\npage 2 para 2",
    ]


def test_pages_mode_falls_back_to_whole_corpus_without_triple_newline() -> None:
    """A corpus with only paragraph-level breaks yields a single
    page-sized token covering the whole corpus.

    This is intentional: a recipe author who hasn't marked page
    boundaries still gets a usable pages-mode render, just with one
    page replicated across the requested ``count``.
    """
    text = "para 1\n\npara 2\n\npara 3"
    assert tokenize(text, mode="pages") == ["para 1\n\npara 2\n\npara 3"]


def test_pages_mode_drops_empty_pages_after_strip() -> None:
    """Whitespace-only pages between triple-newline breaks are dropped.

    Mirrors the ``paragraphs`` empty-strip rule so a corpus with
    stray multi-blank-line runs at the head/tail doesn't produce
    empty page tokens that would crash the renderer.
    """
    text = "  \n\n\nmiddle\n\n\n   \n\n\n"
    assert tokenize(text, mode="pages") == ["middle"]


# ---------------------------------------------------------------------------
# edge cases — line endings, empty input, exotic whitespace
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty_list_for_every_mode() -> None:
    """An empty corpus must produce an empty token list, not raise.

    Downstream renderers handle an empty token list with a clean
    ``no tokens after tokenization`` error; an unexpected exception
    here would surface as a stack trace instead, hiding the cause.
    """
    for mode in ("word_crops", "lines", "paragraphs", "pages"):
        assert tokenize("", mode=mode) == []  # type: ignore[arg-type]


def test_only_whitespace_or_newlines_returns_empty_list() -> None:
    """Whitespace-only / newline-only corpora must not produce empty
    tokens that would crash the renderer downstream."""
    for raw in ("   ", "\n", "\n\n", "\n\n\n", "  \n\n\n  ", "\t\t\t"):
        for mode in ("word_crops", "lines", "paragraphs", "pages"):
            assert tokenize(raw, mode=mode) == [], (raw, mode)  # type: ignore[arg-type]


def test_crlf_line_endings_work_for_lines_mode() -> None:
    """Windows-style ``\\r\\n`` line endings tokenize the same as
    plain ``\\n`` for lines mode (``str.splitlines`` handles both)."""
    text = "line1\r\nline2\r\nline3"
    assert tokenize(text, mode="lines") == ["line1", "line2", "line3"]


def test_crlf_line_endings_work_for_paragraphs_mode() -> None:
    """A ``\\r\\n\\r\\n`` paragraph break must split paragraphs the
    same as ``\\n\\n`` does. The regex's ``\\s*`` slot consumes the
    intervening ``\\r``s so the boundary still fires on two ``\\n``s.
    """
    text = "para1\r\n\r\npara2\r\n\r\npara3"
    assert tokenize(text, mode="paragraphs") == ["para1", "para2", "para3"]


def test_crlf_line_endings_work_for_pages_mode() -> None:
    """A ``\\r\\n\\r\\n\\r\\n`` page break must split pages the same
    as ``\\n\\n\\n`` does."""
    text = "page1\r\n\r\n\r\npage2"
    assert tokenize(text, mode="pages") == ["page1", "page2"]


def test_pages_mode_collapses_runs_of_4_or_more_blank_lines() -> None:
    """The page-split regex is greedy on ``\\n+`` after the third
    newline, so any run of >=3 newlines acts as a single page break —
    we don't synthesise empty pages for arbitrarily long blank runs.
    """
    text = "a\n\n\n\n\nb"
    assert tokenize(text, mode="pages") == ["a", "b"]


def test_paragraphs_mode_treats_triple_newline_as_single_break() -> None:
    """``paragraphs`` mode collapses any run of 2+ consecutive newlines
    into a single paragraph break — a triple newline does **not**
    produce an empty paragraph between two real ones.

    This is the canonical guarantee that pages-mode's stronger
    triple-newline boundary doesn't leak into paragraphs-mode
    callers."""
    text = "a\n\n\nb\n\n\n\nc"
    assert tokenize(text, mode="paragraphs") == ["a", "b", "c"]


def test_unicode_line_separators_split_lines_but_not_paragraphs() -> None:
    """``str.splitlines`` splits on U+2028 / U+2029 / U+0085 / U+000B
    / U+000C in addition to ``\\n``, but the paragraph and page
    regexes only fire on ``\\n``-anchored runs. Documenting this
    asymmetry as a regression guard so a future migration to a
    Unicode-aware paragraph splitter is a deliberate change.
    """
    # Two U+2028 LINE SEPARATORs do split lines.
    text = "p1  p2"
    assert tokenize(text, mode="lines") == ["p1", "p2"]
    # ...but they do NOT count as a paragraph boundary.
    assert tokenize(text, mode="paragraphs") == ["p1  p2"]


def test_word_crops_treats_nbsp_as_separator() -> None:
    """Non-breaking space (U+00A0) is a Unicode whitespace char, so
    ``\\s`` (and therefore ``[^\\s]+``) treats it as a word boundary.
    Recipe authors who want NBSPs glued to their tokens must strip
    them in a text transform first.
    """
    text = "word1 word2 word3"
    assert tokenize(text, mode="word_crops") == ["word1", "word2", "word3"]


def test_pages_mode_drops_whitespace_only_middle_pages() -> None:
    """A page chunk whose body is whitespace-only after stripping is
    dropped, even when bracketed on both sides by real pages. Without
    this guarantee the renderer would receive a 0-paragraph token
    and die in :func:`_split_page_into_paragraphs`'s caller."""
    text = "alpha\n\n\n   \t   \n\n\nbeta"
    assert tokenize(text, mode="pages") == ["alpha", "beta"]


def test_pages_mode_falls_back_with_only_paragraph_breaks_and_crlf() -> None:
    """The whole-corpus fallback for pages mode must trigger even when
    the corpus uses CRLF paragraph breaks (no triple-newline)."""
    text = "p1\r\n\r\np2\r\n\r\np3"
    out = tokenize(text, mode="pages")
    assert len(out) == 1
    # Body is preserved verbatim (modulo strip), so the renderer's
    # page-paragraph re-split sees the original paragraph structure.
    assert "p1" in out[0] and "p2" in out[0] and "p3" in out[0]


# ---------------------------------------------------------------------------
# error path
# ---------------------------------------------------------------------------


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown layout"):
        tokenize("x", mode="ranchwords")  # type: ignore[arg-type]
