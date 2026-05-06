"""Split post-transform corpus text into samples for the renderer.

The result is a list of strings — one entry per sample the render
loop will emit. The string contents depend on ``layout.mode``:

- ``word_crops``: one word per sample (whitespace + punctuation split,
  internal apostrophes / hyphens preserved).
- ``lines``: one line per sample (split on newlines).
- ``paragraphs``: one paragraph per sample (split on blank lines —
  a run of 2+ consecutive newlines separates paragraphs).
- ``pages``: one **page-sized chunk** per sample, where paragraphs
  inside a page stay glued together. Pages are separated by a
  **triple**-blank-line boundary (a run of 3+ consecutive newlines);
  paragraphs inside a page are separated by the regular blank-line
  boundary used by ``paragraphs`` mode. This lets a corpus author
  express both boundary kinds in a single text stream — the
  renderer's pages-mode dispatch re-splits each token on the
  paragraph boundary to produce the multi-paragraph shape
  ``render_page`` expects.

Token sampling/weighting (uniform vs unique-weighted vs frequency)
is the renderer's responsibility — this layer just enumerates.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

LayoutMode = Literal["word_crops", "lines", "paragraphs", "pages"]

# Non-letter, non-mark, non-internal-word-punctuation characters that
# act as word separators. Apostrophes (Po + Pf + Pi) and hyphens (Pd)
# are *not* separators when they sit between letters; we strip them
# only at token edges.
_WORD_TOKEN_RE = re.compile(
    r"[^\s]+",
    flags=re.UNICODE,
)


def tokenize(text: str, *, mode: LayoutMode) -> list[str]:
    """Split ``text`` into samples for the given layout mode.

    Empty results (after stripping) are dropped. Order is preserved.
    """

    if mode == "word_crops":
        return _word_crops(text)
    if mode == "lines":
        return _lines(text)
    if mode == "paragraphs":
        return _paragraphs(text)
    if mode == "pages":
        return _pages(text)
    raise ValueError(f"unknown layout.mode: {mode!r}")


def _word_crops(text: str) -> list[str]:
    """Whitespace-then-edge-punctuation split.

    Internal punctuation (``it's``, ``twenty-one``, ``Mo Sgéal Féin``)
    stays with the word. Edge punctuation (``"hello,``) is trimmed.
    Tokens that strip to empty are dropped.
    """

    out: list[str] = []
    for raw in _WORD_TOKEN_RE.findall(text):
        cleaned = _strip_edge_punctuation(raw)
        if cleaned:
            out.append(cleaned)
    return out


def _lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n+", flags=re.UNICODE)
# A page boundary is a run of 3+ newlines (with optional intra-run
# whitespace). The triple-newline convention lets a single corpus
# express both paragraph and page boundaries in plain text — the
# renderer's pages-mode dispatch re-splits each page token on the
# paragraph boundary to recover the inner shape.
_PAGE_SPLIT_RE = re.compile(r"\n\s*\n\s*\n+", flags=re.UNICODE)


def _paragraphs(text: str) -> list[str]:
    chunks = _PARAGRAPH_SPLIT_RE.split(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _pages(text: str) -> list[str]:
    """Split ``text`` into pages on the triple-blank-line boundary.

    A corpus with no triple-blank-line breaks yields a single
    page-sized token (the whole corpus). That's still useful — the
    pages-mode dispatch will sample the same page index repeatedly,
    which exercises the multi-paragraph composition path even on a
    one-page corpus. Recipe authors who want page variety should
    insert triple-blank-line breaks between page-sized passages.
    """
    chunks = _PAGE_SPLIT_RE.split(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _strip_edge_punctuation(token: str) -> str:
    """Drop leading/trailing characters whose Unicode category starts with P.

    Internal punctuation is preserved so ``don't`` stays as ``don't``
    and ``Mo Sgéal`` would tokenize per spec into ``Mo`` and ``Sgéal``
    (the comma after ``Sgéal,`` would be stripped on the right).
    """

    start = 0
    end = len(token)
    while start < end and unicodedata.category(token[start]).startswith("P"):
        start += 1
    while end > start and unicodedata.category(token[end - 1]).startswith("P"):
        end -= 1
    return token[start:end]
