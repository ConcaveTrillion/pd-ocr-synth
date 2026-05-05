"""Provider-level line filters.

Spec: ``docs/specs/04-corpus-providers.md`` ("Provider-level filters").
Operates on the post-fetch text of one provider before that text joins
the recipe-wide corpus pool. Independent of recipe-level text
transforms.

The filter is applied line-by-line. Regex patterns use Python's ``re``
module (multiline flag is *not* set; each line is matched on its own).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CorpusFilter:
    """Compiled, immutable filter applied to one provider's output."""

    drop_lines_matching: re.Pattern[str] | None = None
    keep_only_lines_matching: re.Pattern[str] | None = None
    min_line_chars: int = 0

    @classmethod
    def from_options(cls, options: Mapping[str, object] | None) -> CorpusFilter | None:
        """Build a filter from the recipe's ``filter:`` mapping.

        Returns ``None`` if no filter options were specified — that
        skips the apply pass entirely so callers don't pay for it.
        """

        if not options:
            return None
        drop = _compile(options.get("drop_lines_matching"))
        keep = _compile(options.get("keep_only_lines_matching"))
        min_chars = int(options.get("min_line_chars", 0) or 0)
        if drop is None and keep is None and min_chars <= 0:
            return None
        return cls(
            drop_lines_matching=drop,
            keep_only_lines_matching=keep,
            min_line_chars=min_chars,
        )

    def apply(self, text: str) -> str:
        out: list[str] = []
        for line in text.splitlines():
            if self.drop_lines_matching is not None and self.drop_lines_matching.search(line):
                continue
            if (
                self.keep_only_lines_matching is not None
                and not self.keep_only_lines_matching.search(line)
            ):
                continue
            if self.min_line_chars > 0 and len(line.strip()) < self.min_line_chars:
                continue
            out.append(line)
        if not out:
            return ""
        return "\n".join(out) + "\n"


def _compile(pattern: object) -> re.Pattern[str] | None:
    if pattern is None:
        return None
    if not isinstance(pattern, str):
        raise TypeError(f"filter pattern must be a string, got {type(pattern).__name__}")
    return re.compile(pattern)


def apply_filter(text: str, options: Mapping[str, object] | None) -> str:
    """Convenience entry point: build a filter from options and apply it."""
    f = CorpusFilter.from_options(options)
    return text if f is None else f.apply(text)
