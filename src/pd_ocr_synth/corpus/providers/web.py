"""Web corpus provider: HTTP GET → parse → cache."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any, ClassVar

import httpx

from pd_ocr_synth.corpus.context import ProviderContext
from pd_ocr_synth.corpus.exceptions import OfflineCacheMissError, ProviderError
from pd_ocr_synth.corpus.http import (
    DEFAULT_RETRIES,
    HostRateLimiter,
    build_client,
    get_with_retries,
)
from pd_ocr_synth.corpus.parsers import get_parser, parse_json


class WebProvider:
    """Fetch a single URL and parse it according to ``options['parser']``."""

    type_name: ClassVar[str] = "web"
    schema_version: ClassVar[int] = 1

    def cache_key(self, options: dict[str, Any]) -> str:
        url = str(options["url"])
        parser = options.get("parser") or "plain"
        # ``field_path`` selects which JSON sub-tree the parser returns,
        # so it changes the cached output and must be part of the key.
        # Other options like ``retries``/``user_agent`` only affect the
        # transport, not the content, so they are intentionally omitted.
        field_path = options.get("field_path") or ""
        material = f"{parser}|{field_path}|{url}"
        digest = hashlib.sha256(material.encode()).hexdigest()[:16]
        return f"web-{digest}"

    def fetch(self, ctx: ProviderContext, options: dict[str, Any]) -> Iterable[str]:
        url = str(options["url"])
        parser_name = options.get("parser") or "plain"
        cache_enabled = options.get("cache", True)
        key = self.cache_key(options)

        if cache_enabled and ctx.cache.has(self.type_name, key):
            yield ctx.cache.read_text(self.type_name, key)
            return

        if ctx.offline:
            raise OfflineCacheMissError(
                f"offline=True and cache miss for {url} (provider=web). "
                f"Run `pd-ocr-synth fetch <recipe>` first."
            )

        body = _http_get(ctx, url, options)
        try:
            text = _apply_parser(body, parser_name, options)
        except ProviderError:
            raise
        except (ValueError, LookupError) as exc:
            # ``parse_json`` can raise ``json.JSONDecodeError`` (a
            # ``ValueError``) on malformed responses; wrap so callers
            # see a uniform error surface across providers.
            raise ProviderError(f"web parse failed for {url}: {exc}") from exc

        if cache_enabled:
            ctx.cache.write_text(
                self.type_name,
                key,
                text,
                source=url,
                extras={"parser": parser_name},
            )

        yield text


def _http_get(ctx: ProviderContext, url: str, options: dict[str, Any]) -> str:
    """Issue an HTTP GET via the context client (or build one)."""

    retries = int(options.get("retries", DEFAULT_RETRIES))
    rate_limiter: HostRateLimiter | None = options.get("_rate_limiter")
    client = ctx.http if isinstance(ctx.http, httpx.Client) else None
    owns_client = client is None
    if client is None:
        client = build_client()
    try:
        response = get_with_retries(
            client,
            url,
            retries=retries,
            rate_limiter=rate_limiter,
        )
        return response.text
    except httpx.HTTPError as exc:
        raise ProviderError(f"web fetch failed for {url}: {exc}") from exc
    finally:
        if owns_client:
            client.close()


def _apply_parser(body: str, parser_name: str, options: dict[str, Any]) -> str:
    if parser_name == "json":
        field_path = options.get("field_path")
        return parse_json(body, field_path=field_path)
    parser = get_parser(parser_name)
    return parser(body)
