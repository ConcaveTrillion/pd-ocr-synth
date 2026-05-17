"""Built-in text transforms.

All callables here implement the ``Transform`` protocol:
``(text, options, rng) -> str``. They are registered into the default
registry by :func:`register_builtins`.

References:

- Spec ``docs/specs/05-text-transforms.md``
- The Gaelic recipe (``recipes/gaelic.yaml``) is the worked example
  exercising ``normalize_whitespace``, ``keep_only``,
  ``apply_lenition_dots``, ``tironian_et``, and ``long_s_medial``.
"""

from __future__ import annotations

import re
import unicodedata
from random import Random
from typing import Any

# ---------------------------------------------------------------------------
# Generic transforms
# ---------------------------------------------------------------------------


_WS_INTRA_PARA_RE = re.compile(r"[ \t]+")
_BLANK_LINE_RUN_RE = re.compile(r"\n{3,}")
_TRAILING_WS_RE = re.compile(r"[ \t]+$", re.MULTILINE)


def normalize_whitespace(text: str, options: dict[str, Any], rng: Random) -> str:
    """Collapse whitespace while preserving paragraph breaks.

    Steps:
    1. Trim trailing spaces/tabs on each line.
    2. Replace runs of ``[ \\t]`` with a single space.
    3. Collapse 3+ consecutive newlines to exactly two (paragraph break).
    """
    _ = options, rng
    out = _TRAILING_WS_RE.sub("", text)
    out = _WS_INTRA_PARA_RE.sub(" ", out)
    out = _BLANK_LINE_RUN_RE.sub("\n\n", out)
    return out.strip("\n") + "\n" if out else ""


def lowercase(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = options, rng
    return text.lower()


def uppercase(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = options, rng
    return text.upper()


def strip_punctuation(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = options, rng
    return "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))


def nfc(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = options, rng
    return unicodedata.normalize("NFC", text)


def nfd(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = options, rng
    return unicodedata.normalize("NFD", text)


def nfkc(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = options, rng
    return unicodedata.normalize("NFKC", text)


def nfkd(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = options, rng
    return unicodedata.normalize("NFKD", text)


_REGEX_FLAG_MAP = {
    "i": re.IGNORECASE,
    "m": re.MULTILINE,
    "s": re.DOTALL,
    "x": re.VERBOSE,
    "a": re.ASCII,
    "u": re.UNICODE,
}


def regex_replace(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = rng
    pattern = options.get("pattern")
    replacement = options.get("replacement", "")
    flags_str = str(options.get("flags") or "")
    if pattern is None:
        raise ValueError("regex_replace requires 'pattern'")
    flags = 0
    for ch in flags_str:
        if ch in _REGEX_FLAG_MAP:
            flags |= _REGEX_FLAG_MAP[ch]
    return re.sub(pattern, replacement, text, flags=flags)


def keep_only(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = rng
    chars = options.get("chars")
    if not chars:
        raise ValueError("keep_only requires 'chars'")
    allowed = set(chars)
    return "".join(ch for ch in text if ch in allowed)


def min_token_length(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = rng
    min_len = int(options.get("min", options.get("length", 0)))
    if min_len <= 0:
        return text
    return _filter_tokens(text, lambda token: len(token) >= min_len)


def max_token_length(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = rng
    max_len = int(options.get("max", options.get("length", 0)))
    if max_len <= 0:
        return text
    return _filter_tokens(text, lambda token: len(token) <= max_len)


_TOKEN_RE = re.compile(r"\S+")


def _filter_tokens(text: str, predicate) -> str:
    """Drop whitespace-delimited tokens that fail ``predicate``.

    Whitespace runs between kept tokens collapse to a single space;
    paragraph breaks (blank lines) are preserved.
    """
    paragraphs = text.split("\n\n")
    out_paras: list[str] = []
    for paragraph in paragraphs:
        kept = [tok for tok in _TOKEN_RE.findall(paragraph) if predicate(tok)]
        out_paras.append(" ".join(kept))
    return "\n\n".join(out_paras)


# ---------------------------------------------------------------------------
# Gaelic / pre-reform Irish transforms
# ---------------------------------------------------------------------------

# Digraph → dotted-consonant table per spec 05.
_LENITION_MAP = {
    "b": "ḃ",
    "c": "ċ",
    "d": "ḋ",
    "f": "ḟ",
    "g": "ġ",
    "m": "ṁ",
    "p": "ṗ",
    "s": "ṡ",
    "t": "ṫ",
    "B": "Ḃ",
    "C": "Ċ",
    "D": "Ḋ",
    "F": "Ḟ",
    "G": "Ġ",
    "M": "Ṁ",
    "P": "Ṗ",
    "S": "Ṡ",
    "T": "Ṫ",
}
_LENITION_PATTERN = re.compile(r"([bcdfgmpstBCDFGMPST])h")
_VOWELS = set("aeiouáéíóúAEIOUÁÉÍÓÚ")


def apply_lenition_dots(text: str, options: dict[str, Any], rng: Random) -> str:
    """Convert ``Xh`` digraphs to dotted ``Ẋ`` consonants.

    options:
        ``mode``: ``aggressive`` (default) replaces every match;
        ``conservative`` requires the next character (after the
        ``h``) to be a vowel.
        ``probability``: per-match keep rate (default 1.0).
    """

    mode = options.get("mode", "aggressive")
    if mode not in {"aggressive", "conservative"}:
        raise ValueError(f"apply_lenition_dots: unknown mode {mode!r}")
    probability = float(options.get("probability", 1.0))

    def repl(match: re.Match[str]) -> str:
        prefix = match.group(1)
        if mode == "conservative":
            end = match.end()
            following = text[end] if end < len(text) else ""
            if following not in _VOWELS:
                return match.group(0)
        if probability < 1.0 and rng.random() > probability:
            return match.group(0)
        return _LENITION_MAP[prefix]

    return _LENITION_PATTERN.sub(repl, text)


# Aliases per spec 05: seimhiu_to_dot is the canonical alias of
# apply_lenition_dots; dot_to_seimhiu reverses the mapping.
seimhiu_to_dot = apply_lenition_dots


_DOTTED_TO_DIGRAPH = {dotted: prefix + "h" for prefix, dotted in _LENITION_MAP.items()}
_DOTTED_PATTERN = re.compile("|".join(re.escape(c) for c in _DOTTED_TO_DIGRAPH))


def dot_to_seimhiu(text: str, options: dict[str, Any], rng: Random) -> str:
    _ = options, rng
    return _DOTTED_PATTERN.sub(lambda m: _DOTTED_TO_DIGRAPH[m.group(0)], text)


def tironian_et(text: str, options: dict[str, Any], rng: Random) -> str:
    """Replace selected words with the Tironian sign ``⁊``.

    options:
        ``replace_words``: list of words to swap (default
        ``['agus', 'and', 'et']``).
        ``probability``: per-occurrence keep rate (default 1.0).
        ``case_sensitive``: default False.
    """

    words = options.get("replace_words") or ["agus", "and", "et"]
    probability = float(options.get("probability", 1.0))
    case_sensitive = bool(options.get("case_sensitive", False))
    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(r"\b(" + "|".join(re.escape(w) for w in words) + r")\b", flags=flags)

    def repl(match: re.Match[str]) -> str:
        if probability < 1.0 and rng.random() > probability:
            return match.group(0)
        return "⁊"

    return pattern.sub(repl, text)


_LONG_S_TOKEN_RE = re.compile(r"\S+", flags=re.UNICODE)


def long_s_medial(text: str, options: dict[str, Any], rng: Random) -> str:
    """Replace ``s`` with ``\u017f`` in non-final positions, never before s/h.

    Per spec: word-medial only (i.e. not the last character of a
    word) and never immediately before another ``s`` or ``h``.
    Word-initial ``s`` is allowed — early-modern typography did use
    long-s word-initially. ``S`` (uppercase) is left as-is to match
    historical practice.

    options:
        ``probability``: per-eligible-``s`` keep rate (default 1.0).
    """

    probability = float(options.get("probability", 1.0))

    def replace_in_token(token: str) -> str:
        chars = list(token)
        for i, ch in enumerate(chars):
            if ch != "s":
                continue
            # Word-final → skip.
            if i == len(chars) - 1:
                continue
            nxt = chars[i + 1]
            if nxt in {"s", "h"}:
                continue
            if probability < 1.0 and rng.random() > probability:
                continue
            chars[i] = "\u017f"  # U+017F LATIN SMALL LETTER LONG S
        return "".join(chars)

    return _LONG_S_TOKEN_RE.sub(lambda m: replace_in_token(m.group(0)), text)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_builtins(registry) -> None:
    """Register every built-in transform on ``registry``."""

    registry.register("normalize_whitespace", normalize_whitespace)
    registry.register("lowercase", lowercase)
    registry.register("uppercase", uppercase)
    registry.register("strip_punctuation", strip_punctuation)
    registry.register("nfc", nfc)
    registry.register("nfd", nfd)
    registry.register("nfkc", nfkc)
    registry.register("nfkd", nfkd)
    registry.register("regex_replace", regex_replace)
    registry.register("keep_only", keep_only)
    registry.register("min_token_length", min_token_length)
    registry.register("max_token_length", max_token_length)
    registry.register("apply_lenition_dots", apply_lenition_dots)
    registry.register("seimhiu_to_dot", seimhiu_to_dot)
    registry.register("dot_to_seimhiu", dot_to_seimhiu)
    registry.register("tironian_et", tironian_et)
    registry.register("long_s_medial", long_s_medial)
