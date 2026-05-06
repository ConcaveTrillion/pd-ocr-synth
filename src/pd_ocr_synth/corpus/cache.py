"""On-disk cache for corpus provider output.

Layout under the cache root::

    <root>/<provider>/<key>.txt          # the parsed text
    <root>/<provider>/<key>.meta.json    # CacheMeta (sidecar)

The cache is content-addressable: the ``key`` is computed by each
provider's ``cache_key(options)`` method, so two corpus entries that
fetch the same data share a single cache slot.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from pd_ocr_synth.corpus.exceptions import CorpusError

CACHE_ENV_VAR = "PD_OCR_SYNTH_CACHE"


def default_cache_root() -> Path:
    """Resolve the default cache root.

    Honors ``$PD_OCR_SYNTH_CACHE`` if set, otherwise
    ``~/.cache/pd-ocr-synth/``. Path is not created — the caller does
    that lazily on first write.
    """

    env = os.environ.get(CACHE_ENV_VAR)
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "pd-ocr-synth"


class CacheMissError(CorpusError):
    """Raised by ``CacheStore.read_text`` when a key has no entry."""


@dataclass(frozen=True, slots=True)
class CacheMeta:
    """Sidecar metadata for a cache entry.

    ``extras`` carries provider-specific fields (URL, item identifier,
    etc.) so cache forensics (`pd-ocr-synth describe`, manual `cat
    *.meta.json`) don't lose context.
    """

    provider: str
    key: str
    source: str
    fetched_at: str
    sha256: str
    size_bytes: int
    extras: dict[str, str] = field(default_factory=dict)

    @classmethod
    def for_text(
        cls,
        *,
        provider: str,
        key: str,
        source: str,
        text: str,
        extras: dict[str, str] | None = None,
    ) -> CacheMeta:
        encoded = text.encode("utf-8")
        return cls(
            provider=provider,
            key=key,
            source=source,
            fetched_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            sha256=hashlib.sha256(encoded).hexdigest(),
            size_bytes=len(encoded),
            extras=dict(extras or {}),
        )


@dataclass(frozen=True, slots=True)
class CacheStore:
    """File-backed cache store.

    Construct with a root path; the directory tree is created lazily.
    Keys are sanitized to a safe filename when they hit disk.
    """

    root: Path

    # ----- path helpers -----

    def _provider_dir(self, provider: str) -> Path:
        return self.root / _safe_name(provider)

    def _entry_paths(self, provider: str, key: str) -> tuple[Path, Path]:
        d = self._provider_dir(provider)
        safe_key = _safe_name(key)
        return d / f"{safe_key}.txt", d / f"{safe_key}.meta.json"

    # ----- read / write -----

    def has(self, provider: str, key: str) -> bool:
        text_path, meta_path = self._entry_paths(provider, key)
        return text_path.exists() and meta_path.exists()

    def read_text(self, provider: str, key: str) -> str:
        text_path, _ = self._entry_paths(provider, key)
        if not text_path.exists():
            raise CacheMissError(f"cache miss: {provider}/{key}")
        return text_path.read_text(encoding="utf-8")

    def read_meta(self, provider: str, key: str) -> CacheMeta:
        _, meta_path = self._entry_paths(provider, key)
        if not meta_path.exists():
            raise CacheMissError(f"cache meta missing: {provider}/{key}")
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        return CacheMeta(**data)

    def write_text(
        self,
        provider: str,
        key: str,
        text: str,
        *,
        source: str,
        extras: dict[str, str] | None = None,
    ) -> CacheMeta:
        text_path, meta_path = self._entry_paths(provider, key)
        text_path.parent.mkdir(parents=True, exist_ok=True)
        text_path.write_text(text, encoding="utf-8")
        meta = CacheMeta.for_text(
            provider=provider,
            key=key,
            source=source,
            text=text,
            extras=extras,
        )
        meta_path.write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")
        return meta

    # ----- delete / list -----

    def remove(self, provider: str, key: str) -> bool:
        """Remove a single entry. Returns True if anything was deleted."""
        text_path, meta_path = self._entry_paths(provider, key)
        existed = False
        for p in (text_path, meta_path):
            if p.exists():
                p.unlink()
                existed = True
        return existed

    def iter_keys(self, provider: str) -> list[str]:
        """List sanitized keys present on disk for the provider."""
        d = self._provider_dir(provider)
        if not d.exists():
            return []
        return sorted(p.stem for p in d.glob("*.txt"))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


_FORBIDDEN = set('<>:"/\\|?*\x00')


def _safe_name(value: str) -> str:
    """Sanitize a path component to safe filename characters.

    The cache is content-addressable, so the sanitizer must be
    *injective*: distinct inputs MUST produce distinct outputs.

    - Long strings (URLs, etc.) are truncated and a 16-char sha256
      digest is appended.
    - Short strings whose cleaned form equals the raw input pass
      through unchanged (the common provider case ``"web-<digest>"``).
    - Short strings that contain forbidden characters get a digest
      suffix too, so e.g. ``"a/b"`` and ``"a_b"`` cannot share a slot.
    """

    cleaned = "".join("_" if c in _FORBIDDEN else c for c in value)
    if cleaned == value and len(cleaned) <= 80:
        return cleaned
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    head = cleaned[:60].rstrip("._-")
    return f"{head}-{digest}"
