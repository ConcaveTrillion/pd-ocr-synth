"""Tests for ``pd_ocr_synth.corpus.providers.local``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.corpus import (
    CacheStore,
    LocalProvider,
    ProviderContext,
    ProviderError,
    default_registry,
)


@pytest.fixture
def ctx(tmp_path: Path) -> ProviderContext:
    return ProviderContext(
        recipe_dir=tmp_path,
        cache=CacheStore(root=tmp_path / "cache"),
    )


def test_local_provider_registered_by_default() -> None:
    p = default_registry().get("local")
    assert p.type_name == "local"


def test_fetch_single_file(tmp_path: Path, ctx: ProviderContext) -> None:
    target = tmp_path / "seed.txt"
    target.write_text("alpha\nbeta\n", encoding="utf-8")
    chunks = list(LocalProvider().fetch(ctx, {"path": "./seed.txt"}))
    assert chunks == ["alpha\nbeta\n"]


def test_fetch_absolute_path(tmp_path: Path, ctx: ProviderContext) -> None:
    target = tmp_path / "abs.txt"
    target.write_text("x", encoding="utf-8")
    chunks = list(LocalProvider().fetch(ctx, {"path": str(target)}))
    assert chunks == ["x"]


def test_fetch_directory_walks_recursively(tmp_path: Path, ctx: ProviderContext) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "1.txt").write_text("one", encoding="utf-8")
    (tmp_path / "a" / "2.txt").write_text("two", encoding="utf-8")
    (tmp_path / "a" / "nested").mkdir()
    (tmp_path / "a" / "nested" / "3.txt").write_text("three", encoding="utf-8")
    chunks = list(LocalProvider().fetch(ctx, {"path": "./a"}))
    assert chunks == ["one", "two", "three"]


def test_fetch_glob_pattern(tmp_path: Path, ctx: ProviderContext) -> None:
    (tmp_path / "p1.txt").write_text("first", encoding="utf-8")
    (tmp_path / "p2.txt").write_text("second", encoding="utf-8")
    (tmp_path / "ignore.md").write_text("skipped", encoding="utf-8")
    chunks = list(LocalProvider().fetch(ctx, {"path": "*.txt"}))
    assert chunks == ["first", "second"]


def test_fetch_glob_no_matches_raises(tmp_path: Path, ctx: ProviderContext) -> None:
    with pytest.raises(ProviderError, match="glob matched no files"):
        list(LocalProvider().fetch(ctx, {"path": "./*.never"}))


def test_fetch_missing_path_raises(tmp_path: Path, ctx: ProviderContext) -> None:
    with pytest.raises(ProviderError, match="does not exist"):
        list(LocalProvider().fetch(ctx, {"path": "./no-such-file.txt"}))


def test_fetch_empty_directory_raises(tmp_path: Path, ctx: ProviderContext) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ProviderError, match="empty"):
        list(LocalProvider().fetch(ctx, {"path": "./empty"}))


def test_explicit_html_parser_raises_until_web_provider(
    tmp_path: Path, ctx: ProviderContext
) -> None:
    (tmp_path / "page.txt").write_text("ok", encoding="utf-8")
    with pytest.raises(ProviderError, match="html-text"):
        list(
            LocalProvider().fetch(
                ctx,
                {"path": "./page.txt", "parser": "html-text"},
            )
        )


def test_cache_key_stable_for_same_options() -> None:
    p = LocalProvider()
    k1 = p.cache_key({"path": "/a/b.txt"})
    k2 = p.cache_key({"path": "/a/b.txt"})
    assert k1 == k2
    # Different parser → different key.
    k3 = p.cache_key({"path": "/a/b.txt", "parser": "html-text"})
    assert k1 != k3


def test_cache_key_differs_per_path() -> None:
    p = LocalProvider()
    assert p.cache_key({"path": "/a"}) != p.cache_key({"path": "/b"})
