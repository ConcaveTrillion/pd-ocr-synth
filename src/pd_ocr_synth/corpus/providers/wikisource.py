"""Wikisource corpus provider via the MediaWiki API.

For each title in ``options['titles']``, GET the wiki's ``api.php``
with ``action=parse&prop=text``, then strip HTML to plain text. One
yielded chunk per title.

Category-based lookup (``options['category']``) is intentionally not
yet implemented — it requires a second API roundtrip and the M03
goal is title-based. The next milestone (or a follow-up commit in
this one) can add it.
"""

from __future__ import annotations

import hashlib
import json
import urllib.parse
from collections.abc import Iterable
from typing import ClassVar

import httpx

from pd_ocr_synth.corpus.context import ProviderContext
from pd_ocr_synth.corpus.exceptions import OfflineCacheMissError, ProviderError
from pd_ocr_synth.corpus.http import (
    DEFAULT_RETRIES,
    HostRateLimiter,
    build_client,
    get_with_retries,
)
from pd_ocr_synth.corpus.parsers import parse_html_text


class WikisourceProvider:
    """Pull pages from a Wikisource language edition by title."""

    type_name: ClassVar[str] = "wikisource"
    schema_version: ClassVar[int] = 1

    def cache_key(self, options: dict) -> str:
        language = options["language"]
        titles = options.get("titles") or []
        category = options.get("category")
        material = json.dumps(
            {"language": language, "titles": list(titles), "category": category},
            sort_keys=True,
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]
        return f"wikisource-{language}-{digest}"

    def fetch(self, ctx: ProviderContext, options: dict) -> Iterable[str]:
        if options.get("category"):
            raise ProviderError(
                "wikisource: category-based lookup is not yet implemented; use 'titles:' instead."
            )
        titles = options.get("titles") or []
        if not titles:
            raise ProviderError("wikisource: 'titles' must be a non-empty list")

        cache_enabled = options.get("cache", True)
        key = self.cache_key(options)

        if cache_enabled and ctx.cache.has(self.type_name, key):
            yield ctx.cache.read_text(self.type_name, key)
            return

        if ctx.offline:
            raise OfflineCacheMissError(
                f"offline=True and cache miss for wikisource {options['language']}/"
                f"{titles[0]}…. Run `pd-ocr-synth fetch <recipe>` first."
            )

        client, owns_client = _resolve_client(ctx)
        rate_limiter: HostRateLimiter | None = options.get("_rate_limiter")
        retries = int(options.get("retries", DEFAULT_RETRIES))
        try:
            chunks = [
                _fetch_title(
                    client,
                    language=options["language"],
                    title=title,
                    retries=retries,
                    rate_limiter=rate_limiter,
                )
                for title in titles
            ]
        finally:
            if owns_client:
                client.close()

        text = "\n\n".join(chunks).rstrip() + "\n"
        if cache_enabled:
            ctx.cache.write_text(
                self.type_name,
                key,
                text,
                source=f"wikisource:{options['language']}",
                extras={"titles": ",".join(titles)},
            )
        yield text


def _resolve_client(ctx: ProviderContext) -> tuple[httpx.Client, bool]:
    if isinstance(ctx.http, httpx.Client):
        return ctx.http, False
    return build_client(), True


def _fetch_title(
    client: httpx.Client,
    *,
    language: str,
    title: str,
    retries: int,
    rate_limiter: HostRateLimiter | None,
) -> str:
    # The multilingual Wikisource lives at wikisource.org (no
    # subdomain). All other language editions are at <lang>.wikisource.org.
    host = "wikisource.org" if language == "mul" else f"{language}.wikisource.org"
    api_url = f"https://{host}/w/api.php"
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
        "formatversion": "2",
        "redirects": "1",
    }
    full_url = f"{api_url}?{urllib.parse.urlencode(params)}"
    try:
        response = get_with_retries(
            client,
            full_url,
            retries=retries,
            rate_limiter=rate_limiter,
        )
    except httpx.HTTPError as exc:
        raise ProviderError(f"wikisource fetch failed for {language}/{title}: {exc}") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        snippet = response.text[:120].replace("\n", " ")
        raise ProviderError(
            f"wikisource: non-JSON response for {language}/{title}: {snippet!r}"
        ) from exc
    if "error" in payload:
        info = payload["error"].get("info", "unknown error")
        raise ProviderError(f"wikisource API error for {language}/{title}: {info}")

    parse_block = payload.get("parse")
    if not parse_block:
        raise ProviderError(
            f"wikisource: unexpected response for {language}/{title}: missing 'parse'"
        )
    html_body = parse_block.get("text")
    if not isinstance(html_body, str):
        raise ProviderError(f"wikisource: unexpected 'parse.text' shape for {language}/{title}")
    return parse_html_text(html_body)
