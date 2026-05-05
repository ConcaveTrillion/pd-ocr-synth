"""Tests for parser functions in ``pd_ocr_synth.corpus.parsers``."""

from __future__ import annotations

import pytest

from pd_ocr_synth.corpus.exceptions import ProviderError
from pd_ocr_synth.corpus.parsers import (
    get_parser,
    parse_html_text,
    parse_json,
    parse_plain,
    parse_tei_text,
)


def test_plain_passthrough() -> None:
    assert parse_plain("hello\nworld\n") == "hello\nworld\n"


def test_html_text_drops_script_and_style() -> None:
    body = """\
<html><head><style>body{color:red}</style></head>
<body>
  <p>Hello, <em>world</em>!</p>
  <script>alert('boom')</script>
  <p>Second paragraph.</p>
</body></html>
"""
    out = parse_html_text(body)
    assert "alert" not in out
    assert "color:red" not in out
    assert "Hello" in out
    assert "Second paragraph" in out


def test_html_text_collapses_blank_runs() -> None:
    body = "<p>a</p>\n\n\n\n<p>b</p>"
    out = parse_html_text(body)
    # No more than one consecutive blank line internally.
    blocks = out.strip().split("\n")
    runs = [(blocks[i] == "", blocks[i - 1] == "") for i in range(1, len(blocks))]
    assert not any(prev and curr for curr, prev in runs)


def test_tei_text_extracts_text_block() -> None:
    body = """<?xml version="1.0"?>
<TEI><teiHeader><title>Drop me</title></teiHeader>
  <text><body>
    <p>Keep this</p>
  </body></text>
</TEI>
"""
    out = parse_tei_text(body)
    assert "Keep this" in out
    assert "Drop me" not in out


def test_json_parser_path_extracts_array_field() -> None:
    body = '{"entries": [{"body": "alpha"}, {"body": "beta"}]}'
    out = parse_json(body, field_path="$.entries[*].body")
    assert out.strip().splitlines() == ["alpha", "beta"]


def test_json_parser_root_returns_serialized() -> None:
    body = '{"a": 1}'
    out = parse_json(body)
    assert '"a": 1' in out


def test_json_parser_invalid_path_raises() -> None:
    with pytest.raises(ProviderError, match=r"\$"):
        parse_json("{}", field_path="entries.body")  # missing leading $


def test_get_parser_unknown_raises() -> None:
    with pytest.raises(ProviderError, match="unknown parser"):
        get_parser("nope")
