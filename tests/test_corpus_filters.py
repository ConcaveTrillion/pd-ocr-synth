"""Tests for ``pd_ocr_synth.corpus.filters``."""

from __future__ import annotations

import pytest

from pd_ocr_synth.corpus import CorpusFilter, apply_filter


def test_no_options_is_passthrough() -> None:
    assert apply_filter("a\nb\n", None) == "a\nb\n"
    assert apply_filter("a\nb\n", {}) == "a\nb\n"


def test_drop_lines_matching_drops_matches() -> None:
    out = apply_filter("alpha\n123\nbeta\n", {"drop_lines_matching": r"^\d+$"})
    assert out == "alpha\nbeta\n"


def test_keep_only_lines_matching_filters_inverse() -> None:
    out = apply_filter(
        "the quick\n12345\nfox jumps\n",
        {"keep_only_lines_matching": r"[A-Za-z]"},
    )
    assert out == "the quick\nfox jumps\n"


def test_min_line_chars_filters_short_lines() -> None:
    out = apply_filter("hello\nx\n  \nworld\n", {"min_line_chars": 3})
    # 'x' (1 char) and '  ' (0 after strip) drop; 'hello' and 'world' stay.
    assert out == "hello\nworld\n"


def test_combined_rules() -> None:
    options = {
        "drop_lines_matching": r"^#",
        "keep_only_lines_matching": r"\w",
        "min_line_chars": 2,
    }
    out = apply_filter("# comment\nok\nz\n  \nvalid\n", options)
    assert out == "ok\nvalid\n"


def test_no_lines_pass_returns_empty_string() -> None:
    out = apply_filter("a\nb\n", {"min_line_chars": 100})
    assert out == ""


def test_corpus_filter_from_options_returns_none_when_inert() -> None:
    assert CorpusFilter.from_options(None) is None
    assert CorpusFilter.from_options({}) is None
    # All-default options also short-circuit.
    assert CorpusFilter.from_options({"min_line_chars": 0}) is None


def test_corpus_filter_rejects_non_string_pattern() -> None:
    with pytest.raises(TypeError, match="must be a string"):
        CorpusFilter.from_options({"drop_lines_matching": 123})
