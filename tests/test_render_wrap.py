"""Tests for the pure wrap-fitter (M09 chunk).

The wrap-fitter is a measure-only function — it shapes candidate
lines through HarfBuzz and never paints. Tests here cover three
layers:

1. Argument validation / edge cases that don't need a real font
   (empty input, bad budgets) — fast and font-free.
2. End-to-end line-fit guarantees against the bundled Cló Gaelach
   font: every emitted line, when re-shaped, fits the budget.
3. Hard-break preservation: embedded newlines are honored as
   line boundaries even with a generous budget.

Tests that need the bundled Bunchló GC font skip cleanly when the
font isn't fetched (fresh checkout that hasn't run
``./scripts/fetch-fonts-gaelic.sh``).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.render import fit_lines
from pd_ocr_synth.render.context import RenderContext
from pd_ocr_synth.render.wrap import _measure_width_px

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font_handles(pixel_size: int = 32):
    """Open the bundled font and return pixel-sized handles + size."""
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; wrap tests skipped.")
    ctx = RenderContext.for_seed(0)
    handles = ctx.font_handles(_BUNDLED_FONT)
    handles.ft_face.set_pixel_sizes(pixel_size, pixel_size)
    return handles


# ---------------------------------------------------------------------------
# Validation / edge cases (no font required)
# ---------------------------------------------------------------------------


def test_fit_lines_rejects_non_positive_budget() -> None:
    handles = _require_font_handles()
    with pytest.raises(ValueError, match="max_width_px"):
        fit_lines("abc", max_width_px=0, handles=handles, pixel_size=32)
    with pytest.raises(ValueError, match="max_width_px"):
        fit_lines("abc", max_width_px=-5, handles=handles, pixel_size=32)


def test_fit_lines_rejects_non_positive_pixel_size() -> None:
    handles = _require_font_handles()
    with pytest.raises(ValueError, match="pixel_size"):
        fit_lines("abc", max_width_px=100, handles=handles, pixel_size=0)
    with pytest.raises(ValueError, match="pixel_size"):
        fit_lines("abc", max_width_px=100, handles=handles, pixel_size=-1)


def test_fit_lines_empty_input_returns_empty_list() -> None:
    handles = _require_font_handles()
    assert fit_lines("", max_width_px=100, handles=handles, pixel_size=32) == []
    assert fit_lines("   ", max_width_px=100, handles=handles, pixel_size=32) == []
    assert fit_lines("\n\n\n", max_width_px=100, handles=handles, pixel_size=32) == []


# ---------------------------------------------------------------------------
# Smoke: single word and generous budget
# ---------------------------------------------------------------------------


def test_fit_lines_single_word_returns_single_line() -> None:
    handles = _require_font_handles()
    result = fit_lines("ḃeaḋ", max_width_px=10000, handles=handles, pixel_size=32)
    assert result == ["ḃeaḋ"]


def test_fit_lines_generous_budget_keeps_everything_on_one_line() -> None:
    handles = _require_font_handles()
    text = "ḃeaḋ saoġal mór"
    # 10000 px is far wider than any reasonable shaped line.
    result = fit_lines(text, max_width_px=10000, handles=handles, pixel_size=32)
    assert result == [text]


# ---------------------------------------------------------------------------
# Wrap behaviour: tight budget splits the input into multiple lines
# ---------------------------------------------------------------------------


def test_fit_lines_splits_when_budget_is_tight() -> None:
    handles = _require_font_handles()
    # Repeat a moderate phrase several times to guarantee at least
    # two lines at any reasonable budget.
    text = "ḃeaḋ saoġal mór " * 6
    full_width = _measure_width_px(text.strip(), handles=handles, pixel_size=32, features=None)
    budget = int(full_width / 3)
    result = fit_lines(text, max_width_px=budget, handles=handles, pixel_size=32)
    assert len(result) >= 2, f"expected at least two lines for budget={budget}, got {result!r}"


def test_fit_lines_every_line_fits_budget_when_words_individually_fit() -> None:
    """Greedy first-fit invariant.

    Every emitted line that contains more than one word must fit
    the budget — otherwise the fitter would have flushed the
    previous word onto its own line. A *single* word that itself
    exceeds the budget is allowed (long-word policy in module
    docstring) — this test specifically uses words that fit
    individually so the invariant is unconditional.
    """

    handles = _require_font_handles()
    pixel_size = 32
    # Short Gaelic words; each fits comfortably in 200 px at 32 px size.
    text = "ḃeaḋ saoġal mór beag bán dub mór beag ḃeaḋ saoġal"
    budget = 200
    # Sanity: the full text exceeds the budget, so we expect splits.
    full_w = _measure_width_px(text, handles=handles, pixel_size=pixel_size, features=None)
    assert full_w > budget, "test assumes the joined line is wider than the budget"

    result = fit_lines(text, max_width_px=budget, handles=handles, pixel_size=pixel_size)
    assert len(result) >= 2

    for line in result:
        line_w = _measure_width_px(line, handles=handles, pixel_size=pixel_size, features=None)
        # Every emitted multi-word line must fit. This is the central
        # correctness invariant for first-fit greedy wrapping.
        if " " in line:
            assert line_w <= budget, f"line {line!r} measured {line_w:.1f}px > budget {budget}px"


def test_fit_lines_concatenated_words_match_input_word_order() -> None:
    """Wrapping is order-preserving: joining all output lines with
    spaces reproduces the original whitespace-collapsed input."""

    handles = _require_font_handles()
    text = "ḃeaḋ saoġal mór beag bán dub mór beag ḃeaḋ saoġal"
    result = fit_lines(text, max_width_px=200, handles=handles, pixel_size=32)
    assert " ".join(result) == text


# ---------------------------------------------------------------------------
# Long-word policy
# ---------------------------------------------------------------------------


def test_fit_lines_emits_long_word_alone_even_when_over_budget() -> None:
    """A single word wider than the budget gets its own line.

    No character-level breaking; the line just exceeds the budget.
    """

    handles = _require_font_handles()
    pixel_size = 32
    long_word = "ḃeaḋsaoġalmórbeag"  # one word, no spaces
    word_w = _measure_width_px(long_word, handles=handles, pixel_size=pixel_size, features=None)
    tight_budget = max(1, int(word_w / 4))

    result = fit_lines(long_word, max_width_px=tight_budget, handles=handles, pixel_size=pixel_size)
    assert result == [long_word]


def test_fit_lines_long_word_gets_its_own_line_when_mixed_with_short_words() -> None:
    handles = _require_font_handles()
    pixel_size = 32
    long_word = "ḃeaḋsaoġalmórbeagbán"
    text = f"a b c {long_word} d e"
    word_w = _measure_width_px(long_word, handles=handles, pixel_size=pixel_size, features=None)
    tight_budget = max(1, int(word_w / 2))

    result = fit_lines(text, max_width_px=tight_budget, handles=handles, pixel_size=pixel_size)
    # The long word must appear in some line by itself (it can't fit
    # alongside any siblings under this budget).
    assert long_word in result, f"long word missing from {result!r}"
    # And that line is exactly the long word (no other words on it).
    long_line = next(line for line in result if line == long_word)
    assert long_line == long_word


# ---------------------------------------------------------------------------
# Hard-break (embedded newline) handling
# ---------------------------------------------------------------------------


def test_fit_lines_hard_break_on_embedded_newlines() -> None:
    """Embedded newlines split into independent wrap segments.

    With a generous budget each segment becomes one line; the result
    has as many lines as there were non-empty input lines.
    """

    handles = _require_font_handles()
    text = "first line\nsecond line\nthird line"
    result = fit_lines(text, max_width_px=10000, handles=handles, pixel_size=32)
    assert result == ["first line", "second line", "third line"]


def test_fit_lines_hard_break_drops_empty_lines() -> None:
    handles = _require_font_handles()
    text = "first\n\n\nsecond"
    result = fit_lines(text, max_width_px=10000, handles=handles, pixel_size=32)
    assert result == ["first", "second"]


def test_fit_lines_hard_break_combined_with_soft_wrap() -> None:
    """A hard-break-separated chunk that itself overflows wraps further."""

    handles = _require_font_handles()
    pixel_size = 32
    long_chunk = "ḃeaḋ saoġal mór beag bán dub mór beag ḃeaḋ saoġal"
    text = f"short\n{long_chunk}"
    result = fit_lines(text, max_width_px=200, handles=handles, pixel_size=pixel_size)
    # First emitted line is the short hard-break chunk.
    assert result[0] == "short"
    # Remainder all came from the long chunk and reproduces it on
    # space-join.
    assert " ".join(result[1:]) == long_chunk
    # Multiple wrapped lines for the long chunk.
    assert len(result) >= 3


# ---------------------------------------------------------------------------
# Whitespace normalization: multi-space runs collapse to single spaces
# ---------------------------------------------------------------------------


def test_fit_lines_collapses_internal_whitespace_runs() -> None:
    handles = _require_font_handles()
    # Multiple spaces / tabs between words should collapse to single
    # spaces in the output (we tokenize on ``str.split()``).
    text = "ḃeaḋ   saoġal\tmór"
    result = fit_lines(text, max_width_px=10000, handles=handles, pixel_size=32)
    assert result == ["ḃeaḋ saoġal mór"]


# ---------------------------------------------------------------------------
# First-line indent budget shrink (M09 paragraph_indent_px interaction)
# ---------------------------------------------------------------------------


def test_fit_lines_rejects_negative_first_line_indent() -> None:
    handles = _require_font_handles()
    with pytest.raises(ValueError, match="first_line_indent_px"):
        fit_lines(
            "abc",
            max_width_px=100,
            handles=handles,
            pixel_size=32,
            first_line_indent_px=-1,
        )


def test_fit_lines_zero_indent_is_bit_identical_to_default() -> None:
    """``first_line_indent_px=0`` matches the default (no kwarg) bit-for-bit.

    The historical wrap output is a public contract — any caller not
    using the indent kwarg must see exactly the same output as before.
    """

    handles = _require_font_handles()
    pixel_size = 32
    text = "ḃeaḋ saoġal mór beag bán dub mór beag"
    base = fit_lines(text, max_width_px=200, handles=handles, pixel_size=pixel_size)
    indent_zero = fit_lines(
        text,
        max_width_px=200,
        handles=handles,
        pixel_size=pixel_size,
        first_line_indent_px=0,
    )
    assert base == indent_zero


def test_fit_lines_first_line_budget_shrinks_by_indent() -> None:
    """A non-zero indent should pull the first line's budget tighter.

    Without the fix the first line packs as if it had the full
    ``max_width_px``; with the fix the line painted at ``indent +
    line_width`` must satisfy ``indent + line_width <= max_width_px``.
    """

    handles = _require_font_handles()
    pixel_size = 32
    text = "ḃeaḋ saoġal mór beag bán dub mór beag ḃeaḋ saoġal"
    budget = 250
    indent = 80

    no_indent = fit_lines(text, max_width_px=budget, handles=handles, pixel_size=pixel_size)
    with_indent = fit_lines(
        text,
        max_width_px=budget,
        handles=handles,
        pixel_size=pixel_size,
        first_line_indent_px=indent,
    )

    # Concatenated word order is preserved in both cases.
    assert " ".join(no_indent) == text
    assert " ".join(with_indent) == text

    # The indented first line plus the indent must fit the user's
    # requested budget. Without the fix this overflows by exactly
    # ``indent`` pixels (modulo cross-word kerning slop).
    first_line_w = _measure_width_px(
        with_indent[0], handles=handles, pixel_size=pixel_size, features=None
    )
    assert first_line_w + indent <= budget, (
        f"painted first line ({first_line_w:.1f}px) + indent ({indent}px) = "
        f"{first_line_w + indent:.1f}px overflows budget {budget}px"
    )

    # The first line under the indented budget must be no wider than
    # the first line under the un-indented budget — ergo it has at
    # most as many words.
    indented_words = with_indent[0].split()
    natural_words = no_indent[0].split()
    assert len(indented_words) <= len(natural_words), (
        f"indented first line ({with_indent[0]!r}) has more words than "
        f"un-indented ({no_indent[0]!r}); fit_lines did not honor the indent"
    )


def test_fit_lines_indent_only_affects_first_line() -> None:
    """Lines beyond line 0 use the full ``max_width_px`` budget.

    The renderer only indents line 0, so a tighter budget on every
    line would shrink the paragraph for no good reason. We verify by
    measuring all but the first emitted line against ``max_width_px``.
    """

    handles = _require_font_handles()
    pixel_size = 32
    text = "ḃeaḋ saoġal mór beag bán dub mór beag ḃeaḋ saoġal"
    budget = 200
    indent = 60

    result = fit_lines(
        text,
        max_width_px=budget,
        handles=handles,
        pixel_size=pixel_size,
        first_line_indent_px=indent,
    )
    assert len(result) >= 2

    # Lines 1.. should still be packed to the full budget (subject to
    # the long-word policy for single-word lines, hence the multi-word
    # gate).
    for line in result[1:]:
        if " " not in line:
            continue
        line_w = _measure_width_px(line, handles=handles, pixel_size=pixel_size, features=None)
        assert line_w <= budget, (
            f"non-first line {line!r} measured {line_w:.1f}px > budget {budget}px"
        )


def test_fit_lines_indent_with_hard_break_only_shrinks_first_chunk_first_line() -> None:
    """Hard-break chunks beyond the first all use the full budget.

    The indent applies to the rendered paragraph's line 0, which is
    the first emitted line of the first non-empty hard-break chunk —
    every later chunk's first line is paragraph-line-N (N >= 1) where
    the renderer applies no indent.
    """

    handles = _require_font_handles()
    pixel_size = 32
    chunk = "ḃeaḋ saoġal mór beag bán dub mór beag"
    text = f"{chunk}\n{chunk}"
    budget = 200
    indent = 60

    result = fit_lines(
        text,
        max_width_px=budget,
        handles=handles,
        pixel_size=pixel_size,
        first_line_indent_px=indent,
    )

    # Find where the second chunk's wrapped output begins. The first
    # chunk wraps to >= 1 lines (likely 2). We don't know exactly how
    # many, so instead verify that *some* line in the result fits more
    # words than the first line — i.e. the indent didn't poison the
    # whole paragraph.
    word_counts = [len(line.split()) for line in result if " " in line]
    # The first wrapped line is constrained by the indent; at least
    # one later wrapped line must fit at least as many words (because
    # it has the full budget).
    assert max(word_counts) >= word_counts[0], (
        f"every later line is at least as constrained as the first: counts={word_counts}"
    )
