"""Body parsers shared by web/local/web_list providers.

The parser is selected by the ``parser:`` recipe key. Today's parsers:

- ``plain`` — body returned as-is.
- ``html-text`` — BeautifulSoup → drop ``<script>`` / ``<style>`` →
  collapsed text content.
- ``tei-text`` — extract content under the ``<text>`` element of a TEI
  XML document (CELT, OpenGreekAndLatin layout).
- ``json`` — apply a simple ``$.dotted.path[*]`` extraction; v1
  intentionally minimal.

Each parser is a pure function ``parse(body: str) -> str``.
"""

from __future__ import annotations

import json
from collections.abc import Callable

from pd_ocr_synth.corpus.exceptions import ProviderError

ParserFn = Callable[[str], str]


def parse_plain(body: str) -> str:
    return body


def parse_html_text(body: str) -> str:
    # Imported inside the function so the module is importable on
    # systems where bs4 has not been installed yet (e.g. partial
    # editable installs); the actual parser still requires it.
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise ProviderError(
            "the 'html-text' parser requires beautifulsoup4 — install it via "
            "the project's runtime dependencies"
        ) from exc

    soup = BeautifulSoup(body, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Collapse runs of blank lines so downstream tokenization sees
    # meaningful paragraph boundaries.
    lines = [line.strip() for line in text.splitlines()]
    cleaned: list[str] = []
    blank = 0
    for line in lines:
        if line:
            cleaned.append(line)
            blank = 0
        else:
            blank += 1
            if blank == 1:
                cleaned.append("")
    return "\n".join(cleaned).strip() + "\n"


def parse_tei_text(body: str) -> str:
    """Extract the body of a TEI XML document.

    Uses BeautifulSoup's ``html.parser`` rather than its ``xml`` mode
    so we don't pull in lxml as a hard dependency. TEI documents are
    forgiving enough that this works for CELT and similar archives;
    if a producer ships strict-XML-only documents we can revisit.
    """

    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:  # pragma: no cover
        raise ProviderError("the 'tei-text' parser requires beautifulsoup4") from exc

    soup = BeautifulSoup(body, "html.parser")
    # Drop <teiheader> contents — front-matter, not corpus material.
    for tag in soup.find_all("teiheader"):
        tag.decompose()
    text_block = soup.find("text") or soup
    return text_block.get_text(separator="\n").strip() + "\n"


def parse_json(body: str, *, field_path: str | None = None) -> str:
    payload = json.loads(body)
    if field_path is None or field_path in {"$", "$."}:
        return json.dumps(payload, ensure_ascii=False)
    # Minimal path support: '$.a.b[*]' walks dict keys and flattens
    # arrays. Anything fancier waits until a recipe demands it.
    parts = _split_json_path(field_path)
    items = _walk_json(payload, parts)
    return "\n".join(_stringify(item) for item in items) + "\n"


def _split_json_path(path: str) -> list[str]:
    if not path.startswith("$"):
        raise ProviderError(f"json field_path must start with '$', got {path!r}")
    rest = path[1:].lstrip(".")
    if not rest:
        return []
    return [seg for seg in rest.replace("[*]", ".*").split(".") if seg]


def _walk_json(node: object, parts: list[str]) -> list[object]:
    current: list[object] = [node]
    for part in parts:
        nxt: list[object] = []
        for item in current:
            if part == "*":
                if isinstance(item, list):
                    nxt.extend(item)
                elif isinstance(item, dict):
                    nxt.extend(item.values())
                else:
                    raise ProviderError(f"json field_path '*' on non-collection: {type(item)}")
            else:
                if isinstance(item, dict) and part in item:
                    nxt.append(item[part])
                else:
                    raise ProviderError(f"json field_path segment {part!r} missing")
        current = nxt
    return current


def _stringify(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


PARSERS: dict[str, ParserFn] = {
    "plain": parse_plain,
    "html-text": parse_html_text,
    "tei-text": parse_tei_text,
}


def get_parser(name: str) -> ParserFn:
    """Look up a parser by name. ``json`` is special-cased by callers
    because it accepts an additional ``field_path`` option."""
    try:
        return PARSERS[name]
    except KeyError as exc:
        raise ProviderError(
            f"unknown parser '{name}'. Known: {sorted(PARSERS) + ['json']}"
        ) from exc
