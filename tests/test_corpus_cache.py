"""Tests for ``pd_ocr_synth.corpus.cache``."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pd_ocr_synth.corpus import (
    CacheMeta,
    CacheMissError,
    CacheStore,
    default_cache_root,
)


def test_default_cache_root_uses_env_var(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(tmp_path / "explicit"))
    assert default_cache_root() == tmp_path / "explicit"


def test_default_cache_root_falls_back_to_home(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PD_OCR_SYNTH_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_cache_root() == tmp_path / ".cache" / "pd-ocr-synth"


def test_write_then_read_roundtrip(tmp_path: Path) -> None:
    cache = CacheStore(root=tmp_path)
    meta = cache.write_text(
        "local",
        "k1",
        "hello world\n",
        source=str(tmp_path / "src.txt"),
        extras={"note": "test"},
    )
    assert meta.size_bytes == len(b"hello world\n")
    assert meta.sha256 != ""
    assert cache.has("local", "k1")
    assert cache.read_text("local", "k1") == "hello world\n"
    assert cache.read_meta("local", "k1").extras == {"note": "test"}


def test_meta_sidecar_is_valid_json(tmp_path: Path) -> None:
    cache = CacheStore(root=tmp_path)
    cache.write_text("web", "alpha", "x", source="https://example.com/")
    meta_path = tmp_path / "web" / "alpha.meta.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    assert payload["provider"] == "web"
    assert payload["key"] == "alpha"
    assert payload["source"] == "https://example.com/"


def test_read_text_misses_raise(tmp_path: Path) -> None:
    cache = CacheStore(root=tmp_path)
    with pytest.raises(CacheMissError):
        cache.read_text("local", "ghost")


def test_remove_clears_both_files(tmp_path: Path) -> None:
    cache = CacheStore(root=tmp_path)
    cache.write_text("local", "k", "x", source="src")
    assert cache.remove("local", "k") is True
    assert not cache.has("local", "k")
    # Idempotent.
    assert cache.remove("local", "k") is False


def test_iter_keys_lists_entries(tmp_path: Path) -> None:
    cache = CacheStore(root=tmp_path)
    cache.write_text("web", "k1", "a", source="s")
    cache.write_text("web", "k2", "b", source="s")
    assert cache.iter_keys("web") == ["k1", "k2"]
    assert cache.iter_keys("local") == []  # provider has no entries


def test_long_keys_get_hashed(tmp_path: Path) -> None:
    cache = CacheStore(root=tmp_path)
    long_key = "this-is-a-very-long-key-that-exceeds-the-eighty-character-limit-and-then-some-more"
    meta = cache.write_text("web", long_key, "x", source="s")
    written_files = list((tmp_path / "web").glob("*.txt"))
    assert len(written_files) == 1
    # Stored filename does not equal the raw key.
    assert written_files[0].stem != long_key
    # Round-trip still works using the original key.
    assert cache.read_text("web", long_key) == "x"
    assert meta.size_bytes == 1


def test_safe_name_is_injective_for_short_keys_with_forbidden_chars(
    tmp_path: Path,
) -> None:
    """Short keys differing only in forbidden characters must not collide.

    Regression: ``_safe_name`` previously replaced forbidden characters
    with ``_`` and returned the cleaned form unchanged for inputs
    <=80 chars. That meant ``"a/b"`` and ``"a_b"`` (and ``"a:b"``)
    all hashed to the same on-disk filename, so distinct cache entries
    silently shared a slot — the second writer's text overwrote the
    first, and reads returned the wrong content.
    """

    cache = CacheStore(root=tmp_path)
    cache.write_text("p", "a/b", "text-from-slash", source="s1")
    cache.write_text("p", "a_b", "text-from-underscore", source="s2")
    cache.write_text("p", "a:b", "text-from-colon", source="s3")
    assert cache.read_text("p", "a/b") == "text-from-slash"
    assert cache.read_text("p", "a_b") == "text-from-underscore"
    assert cache.read_text("p", "a:b") == "text-from-colon"
    # Three distinct on-disk files — no collision.
    written = sorted(p.name for p in (tmp_path / "p").glob("*.txt"))
    assert len(written) == 3, written


def test_safe_name_passes_through_common_provider_keys(tmp_path: Path) -> None:
    """Provider-generated keys (``<provider>-<digest>``) must remain
    human-readable on disk — i.e. unchanged after sanitization.

    The injective-fix above must not regress this: only inputs that
    actually require sanitization or truncation should pick up a
    digest suffix.
    """

    cache = CacheStore(root=tmp_path)
    cache.write_text("web", "web-1234567890abcdef", "x", source="s")
    written = list((tmp_path / "web").glob("*.txt"))
    assert [p.stem for p in written] == ["web-1234567890abcdef"]


def test_cache_meta_for_text_computes_hash() -> None:
    meta = CacheMeta.for_text(
        provider="x",
        key="k",
        source="s",
        text="hello",
    )
    assert meta.size_bytes == 5
    assert len(meta.sha256) == 64  # sha256 hex length


def test_write_text_is_atomic_across_text_and_meta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between the text rename and the meta rename must not
    leave a readable orphan.

    Regression: ``write_text`` previously wrote ``<key>.txt`` then
    ``<key>.meta.json`` directly. A crash between those calls (process
    kill, OOM, exception in a wrapper, etc.) left a text file on disk
    with no meta sidecar. ``has()`` returned False, but a crafted
    caller still got the orphan via ``read_text()``. The atomic-write
    refactor uses meta-as-sentinel: an orphan text file is treated as
    a cache miss by both ``has()`` and ``read_text()``.
    """

    cache = CacheStore(root=tmp_path)
    text_path, meta_path = tmp_path / "p" / "k.txt", tmp_path / "p" / "k.meta.json"
    text_path.parent.mkdir(parents=True, exist_ok=True)
    # Simulate a crash mid-write by leaving an orphan text file
    # without its meta sidecar.
    text_path.write_text("orphan-from-crash", encoding="utf-8")
    assert text_path.exists()
    assert not meta_path.exists()

    assert cache.has("p", "k") is False
    with pytest.raises(CacheMissError):
        cache.read_text("p", "k")


def test_write_text_meta_rename_failure_does_not_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the meta-rename step fails, the entry is not published.

    The text file may remain on disk after the text-rename succeeded,
    but neither ``has()`` nor ``read_text()`` should report it as a
    valid cache hit.
    """

    cache = CacheStore(root=tmp_path)
    real_replace = os.replace
    calls: list[str] = []

    def flaky_replace(src, dst):  # type: ignore[no-untyped-def]
        calls.append(str(dst))
        # Let the text rename succeed; fail the meta rename.
        if str(dst).endswith(".meta.json"):
            raise OSError("simulated crash before meta publish")
        return real_replace(src, dst)

    monkeypatch.setattr(os, "replace", flaky_replace)

    with pytest.raises(OSError, match="simulated crash"):
        cache.write_text("p", "k", "x", source="s")

    # No meta sidecar means no cache hit.
    assert cache.has("p", "k") is False
    with pytest.raises(CacheMissError):
        cache.read_text("p", "k")
    # And no leftover ``.tmp`` files remain on disk.
    leftovers = sorted(p.name for p in (tmp_path / "p").glob("*.tmp"))
    assert leftovers == [], leftovers


def test_has_returns_false_for_corrupt_meta_json(tmp_path: Path) -> None:
    """A garbled meta sidecar must be treated as a cache miss.

    Regression: previously ``has()`` only checked file presence, so a
    truncated or non-JSON sidecar (filesystem corruption, partial
    write before the atomic-write fix, an editor-saved scratch file)
    returned True — and the next ``read_meta()`` call surprised the
    caller with an unhandled ``JSONDecodeError``.
    """

    cache = CacheStore(root=tmp_path)
    cache.write_text("p", "k", "good text", source="s")
    meta_path = tmp_path / "p" / "k.meta.json"
    meta_path.write_text("{not-json", encoding="utf-8")  # corrupt sidecar
    assert cache.has("p", "k") is False


def test_read_meta_raises_cache_miss_for_corrupt_json(tmp_path: Path) -> None:
    """``read_meta`` reports a corrupt sidecar as a typed CacheMiss
    rather than leaking ``json.JSONDecodeError`` to callers."""

    cache = CacheStore(root=tmp_path)
    cache.write_text("p", "k", "good text", source="s")
    meta_path = tmp_path / "p" / "k.meta.json"
    meta_path.write_text("not-valid-json", encoding="utf-8")
    with pytest.raises(CacheMissError, match="corrupt"):
        cache.read_meta("p", "k")


def test_has_returns_false_for_meta_with_wrong_schema(tmp_path: Path) -> None:
    """A sidecar that parses as JSON but lacks required CacheMeta
    fields must also be reported as absent.

    Defensive: if a future version adds a required field, an old
    sidecar should look like a miss (forcing a refetch) rather than
    raising ``TypeError`` from inside ``CacheMeta(**data)``.
    """

    cache = CacheStore(root=tmp_path)
    cache.write_text("p", "k", "good text", source="s")
    meta_path = tmp_path / "p" / "k.meta.json"
    meta_path.write_text('{"provider": "p"}', encoding="utf-8")  # missing fields
    assert cache.has("p", "k") is False
    with pytest.raises(CacheMissError):
        cache.read_meta("p", "k")
