"""Pure-function line wrap-fitter for ``paragraphs`` / ``pages`` modes.

Spec 06 says:

    layout:
      mode: paragraphs
      max_width_px: 800
      ...

The wrap-fitter turns a free-form word stream + a pixel-width budget
into a ``list[str]`` of lines that each fit within the budget when
shaped end-to-end through HarfBuzz. Pure function over font metrics —
no rasterization, no canvas, no RNG.

Why measure-by-shaping (instead of e.g. summing per-glyph advances of
each word in isolation): cross-word context — kerning, contextual
alternates, even simple ASCII pairs like "Te" — can shrink or grow a
joined line's width vs. the sum of its parts. We shape the *candidate
line* (`" ".join(words_so_far + [word])`) on every trial so the budget
check matches what ``render_line`` will later actually paint.

Greedy first-fit. We don't do Knuth-Plass; that would give prettier
ragged-right margins but is overkill for synthetic OCR training data,
where any reasonable wrap that produces left-aligned line bboxes is
fine.

Long-word policy: if a single word's shaped width exceeds
``max_width_px``, the word is emitted alone on its own line — no
character-level breaking, no glyph splitting. The caller's downstream
``render_line`` / ``render_paragraph`` will still happily render an
oversized line; the only consequence is that the resulting line bbox
will exceed the recipe's wrap budget. Recipes with huge tokens and
tight budgets are a recipe-author bug, not a renderer concern.

Empty-input policy: empty / whitespace-only ``text`` returns ``[]``
rather than raising. Callers that need at least one line should check
for the empty result themselves — same convention as
``str.splitlines``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pd_ocr_synth.render.word_crop import _shape

if TYPE_CHECKING:
    from pd_ocr_synth.render.context import _FontHandles


def fit_lines(
    text: str,
    *,
    max_width_px: int,
    handles: _FontHandles,
    pixel_size: int,
    features: dict[str, Any] | None = None,
    first_line_indent_px: int = 0,
) -> list[str]:
    """Greedy wrap ``text`` into lines that each fit ``max_width_px``.

    Args:
        text: Free-form text. Words are split on contiguous-whitespace
            runs (same convention as ``render_line``'s per-word
            grouping). Embedded newlines are honored as **hard** line
            breaks: each newline-separated chunk wraps independently
            and the results are concatenated. This lets a paragraph
            corpus token that already carries line structure (e.g.
            poetry) keep its hard breaks while still wrapping any
            still-too-long chunks.
        max_width_px: Pixel budget per line. Must be positive.
        handles: Already-opened font handles for the chosen font, with
            ``ft_face`` already pixel-sized (caller responsibility).
            Only ``hb_face`` is used here — measurement runs through
            HarfBuzz alone.
        pixel_size: The pixel size HarfBuzz scales the font to. Same
            value the eventual ``render_line`` will use.
        features: Optional OpenType feature overrides, identical shape
            to what ``_shape`` accepts.
        first_line_indent_px: Pixels by which the **first** emitted
            line will be indented at paint time (see
            :func:`render_paragraph`'s ``first_line_indent_px``). The
            wrap-fitter shrinks the first line's budget to
            ``max_width_px - first_line_indent_px`` so the painted
            line — image strip + indent — still fits the user's
            requested wrap budget. Defaults to ``0`` (no shrink), which
            preserves the historical wrap output bit-for-bit. Must be
            non-negative; values that would zero or negate the first
            line's budget fall back to ``max_width_px`` so a single
            word still fits on line 0 (the renderer's long-word policy
            is to emit an over-budget word alone rather than refuse).

    Returns:
        A list of lines, each a non-empty string with no embedded
        whitespace runs collapsed (we preserve the original word
        spelling). Words within a line are joined by single ASCII
        spaces. Empty / whitespace-only input returns ``[]``.

    Raises:
        ValueError: if ``max_width_px <= 0``, ``pixel_size <= 0``, or
            ``first_line_indent_px < 0``.
    """

    if max_width_px <= 0:
        raise ValueError(f"max_width_px must be positive, got {max_width_px!r}")
    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size!r}")
    if first_line_indent_px < 0:
        raise ValueError(f"first_line_indent_px must be >= 0, got {first_line_indent_px!r}")

    if not text or not text.strip():
        return []

    # Effective first-line budget: trim the indent off ``max_width_px``.
    # Clamp at 1 so a pathological ``indent >= max_width_px`` still
    # produces a usable line-0 budget (one word will go on it under
    # the long-word policy regardless, but a 0/negative budget would
    # be a malformed input to the trial-width compare below).
    if first_line_indent_px > 0:
        first_line_budget = max(1, max_width_px - first_line_indent_px)
    else:
        first_line_budget = max_width_px

    out: list[str] = []
    # Hard-break on existing newlines. ``splitlines`` strips the
    # delimiters and leaves us free to re-join with single spaces.
    # Only the very first emitted line of the **whole paragraph**
    # carries the indent — subsequent lines (whether from continued
    # wrap or from a hard break) all use the full budget. We track
    # ``first_line_used`` across hard-break chunks to honor that.
    first_line_used = False
    for chunk in text.splitlines():
        chunk_stripped = chunk.strip()
        if not chunk_stripped:
            continue
        words = chunk_stripped.split()
        chunk_first_budget = max_width_px if first_line_used else first_line_budget
        chunk_lines = _greedy_pack(
            words,
            max_width_px=max_width_px,
            handles=handles,
            pixel_size=pixel_size,
            features=features,
            first_line_budget_px=chunk_first_budget,
        )
        if chunk_lines:
            first_line_used = True
        out.extend(chunk_lines)
    return out


def _greedy_pack(
    words: list[str],
    *,
    max_width_px: int,
    handles: _FontHandles,
    pixel_size: int,
    features: dict[str, Any] | None,
    first_line_budget_px: int | None = None,
) -> list[str]:
    """Pack ``words`` left-to-right into lines that each fit the budget.

    ``first_line_budget_px`` (when provided and < ``max_width_px``)
    applies a tighter budget to the first emitted line only — used to
    reserve space for ``layout.paragraph_indent_px`` so the painted
    line + indent still fits the recipe's wrap budget. ``None`` (the
    default) means "use ``max_width_px`` for every line", preserving
    bit-for-bit the historical wrap output for callers that don't
    indent.
    """

    if not words:
        return []

    line_budget_px = first_line_budget_px if first_line_budget_px is not None else max_width_px

    lines: list[str] = []
    current: list[str] = []

    for word in words:
        trial = " ".join([*current, word])
        trial_width = _measure_width_px(
            trial,
            handles=handles,
            pixel_size=pixel_size,
            features=features,
        )
        if trial_width <= line_budget_px or not current:
            # Either it fits, or ``current`` is empty (so even an
            # over-budget single word goes onto its own line — we don't
            # do character-level breaking).
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
            # Once we've emitted line 0, the remaining lines use the
            # full ``max_width_px`` budget — the indent only ever
            # applies to line 0.
            line_budget_px = max_width_px

    if current:
        lines.append(" ".join(current))
    return lines


def _measure_width_px(
    text: str,
    *,
    handles: _FontHandles,
    pixel_size: int,
    features: dict[str, Any] | None,
) -> float:
    """Sum a shaped line's per-glyph ``x_advance`` in pixels.

    Mirrors the pen-x summation that ``render_line`` performs at paint
    time so the wrap budget reflects the true painted advance, not a
    naive char-count or per-word measurement that misses cross-word
    shaping. We deliberately use the full HarfBuzz total advance —
    including any trailing whitespace's advance — so a line that fits
    here also fits when ``render_line`` paints it.

    The empty / whitespace-only case is unreachable from
    :func:`fit_lines` (we filter empties up front), but defensively
    return 0.0 rather than crashing on an empty buffer.
    """

    if not text:
        return 0.0
    _glyphs, positions = _shape(handles.hb_face, text, pixel_size, features)
    if not positions:
        return 0.0
    return sum(pos.x_advance for pos in positions) / 64.0
