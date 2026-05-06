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

from typing import TYPE_CHECKING

from pd_ocr_synth.render.word_crop import _shape

if TYPE_CHECKING:
    from pd_ocr_synth.render.context import _FontHandles


def fit_lines(
    text: str,
    *,
    max_width_px: int,
    handles: _FontHandles,
    pixel_size: int,
    features: dict | None = None,
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

    Returns:
        A list of lines, each a non-empty string with no embedded
        whitespace runs collapsed (we preserve the original word
        spelling). Words within a line are joined by single ASCII
        spaces. Empty / whitespace-only input returns ``[]``.

    Raises:
        ValueError: if ``max_width_px <= 0`` or ``pixel_size <= 0``.
    """

    if max_width_px <= 0:
        raise ValueError(f"max_width_px must be positive, got {max_width_px!r}")
    if pixel_size <= 0:
        raise ValueError(f"pixel_size must be positive, got {pixel_size!r}")

    if not text or not text.strip():
        return []

    out: list[str] = []
    # Hard-break on existing newlines. ``splitlines`` strips the
    # delimiters and leaves us free to re-join with single spaces.
    for chunk in text.splitlines():
        chunk_stripped = chunk.strip()
        if not chunk_stripped:
            continue
        words = chunk_stripped.split()
        out.extend(
            _greedy_pack(
                words,
                max_width_px=max_width_px,
                handles=handles,
                pixel_size=pixel_size,
                features=features,
            )
        )
    return out


def _greedy_pack(
    words: list[str],
    *,
    max_width_px: int,
    handles: _FontHandles,
    pixel_size: int,
    features: dict | None,
) -> list[str]:
    """Pack ``words`` left-to-right into lines that each fit the budget."""

    if not words:
        return []

    lines: list[str] = []
    current: list[str] = []

    for word in words:
        trial = " ".join(current + [word])
        trial_width = _measure_width_px(
            trial,
            handles=handles,
            pixel_size=pixel_size,
            features=features,
        )
        if trial_width <= max_width_px or not current:
            # Either it fits, or ``current`` is empty (so even an
            # over-budget single word goes onto its own line — we don't
            # do character-level breaking).
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]

    if current:
        lines.append(" ".join(current))
    return lines


def _measure_width_px(
    text: str,
    *,
    handles: _FontHandles,
    pixel_size: int,
    features: dict | None,
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
