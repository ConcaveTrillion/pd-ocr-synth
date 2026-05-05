"""Tests for ``pd_ocr_synth.corpus.providers.web.WebProvider``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from pd_ocr_synth.corpus import (
    CacheStore,
    OfflineCacheMissError,
    ProviderContext,
    ProviderError,
    WebProvider,
)
from pd_ocr_synth.corpus.http import build_client


def _ctx(
    tmp_path: Path,
    handler: Callable[[httpx.Request], httpx.Response] | None = None,
    *,
    offline: bool = False,
) -> ProviderContext:
    cache = CacheStore(root=tmp_path / "cache")
    client = build_client(transport=httpx.MockTransport(handler)) if handler is not None else None
    return ProviderContext(recipe_dir=tmp_path, cache=cache, offline=offline, http=client)


def test_fetch_plain_caches(tmp_path: Path) -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text="payload")

    ctx = _ctx(tmp_path, handler)
    options = {"url": "https://example.com/p", "parser": "plain"}
    chunks = list(WebProvider().fetch(ctx, options))
    assert chunks == ["payload"]
    assert ctx.cache.has("web", WebProvider().cache_key(options))
    # Second fetch: served from cache, no extra http call.
    chunks2 = list(WebProvider().fetch(ctx, options))
    assert chunks2 == ["payload"]
    assert len(calls) == 1


def test_fetch_html_text_strips_scripts(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text="<html><body><p>Hi <em>there</em></p><script>x</script></body></html>",
        )

    ctx = _ctx(tmp_path, handler)
    chunks = list(
        WebProvider().fetch(
            ctx,
            {"url": "https://example.com/h", "parser": "html-text"},
        )
    )
    assert "Hi" in chunks[0]
    assert "x" not in chunks[0]


def test_fetch_offline_with_no_cache_raises(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, offline=True)  # no handler — would fail if reached
    with pytest.raises(OfflineCacheMissError, match="cache miss"):
        list(WebProvider().fetch(ctx, {"url": "https://example.com/q"}))


def test_fetch_offline_with_cache_serves_from_disk(tmp_path: Path) -> None:
    ctx_offline = _ctx(tmp_path, offline=True)
    options = {"url": "https://example.com/cached", "parser": "plain"}
    key = WebProvider().cache_key(options)
    ctx_offline.cache.write_text("web", key, "from-cache", source=options["url"])
    chunks = list(WebProvider().fetch(ctx_offline, options))
    assert chunks == ["from-cache"]


def test_cache_disabled_skips_write(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="never-cached")

    ctx = _ctx(tmp_path, handler)
    options = {"url": "https://example.com/n", "parser": "plain", "cache": False}
    chunks = list(WebProvider().fetch(ctx, options))
    assert chunks == ["never-cached"]
    key = WebProvider().cache_key(options)
    assert not ctx.cache.has("web", key)


def test_404_surfaces_as_provider_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="missing")

    ctx = _ctx(tmp_path, handler)
    with pytest.raises(ProviderError, match="web fetch failed"):
        list(WebProvider().fetch(ctx, {"url": "https://example.com/m"}))


def test_cache_key_differs_by_parser(tmp_path: Path) -> None:
    p = WebProvider()
    a = p.cache_key({"url": "https://x", "parser": "plain"})
    b = p.cache_key({"url": "https://x", "parser": "html-text"})
    assert a != b
