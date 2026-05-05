"""Corpus provider framework.

A *provider* turns a YAML ``corpus:`` entry into an iterable of UTF-8
strings (one chunk per yield — typically a document or a page). The
framework concatenates them, applies recipe-level text transforms, and
tokenizes for the chosen layout mode.

Public surface (M03):

- ``ProviderContext`` — the per-recipe environment passed to providers
  (recipe directory, cache root, http client, offline flag, logger).
- ``Provider`` — the runtime protocol providers implement.
- ``Registry`` + ``default_registry()`` — dispatches a corpus entry's
  ``type:`` field to the right provider class.
- ``CacheStore`` + ``CacheMeta`` — on-disk cache layer used by every
  provider that opts in with ``cache: true``.
- ``CorpusError`` — base exception for fetch / cache failures.

Web, Wikisource, HF, Internet Archive, and Gutenberg providers ship in
later commits in this milestone.
"""

from __future__ import annotations

from pd_ocr_synth.corpus.cache import CacheMeta, CacheMissError, CacheStore, default_cache_root
from pd_ocr_synth.corpus.context import ProviderContext
from pd_ocr_synth.corpus.exceptions import CorpusError, OfflineCacheMissError, ProviderError
from pd_ocr_synth.corpus.filters import CorpusFilter, apply_filter
from pd_ocr_synth.corpus.providers.local import LocalProvider
from pd_ocr_synth.corpus.providers.web import WebProvider
from pd_ocr_synth.corpus.providers.wikisource import WikisourceProvider
from pd_ocr_synth.corpus.registry import Provider, Registry, default_registry

__all__ = [
    "CacheMeta",
    "CacheMissError",
    "CacheStore",
    "CorpusError",
    "CorpusFilter",
    "LocalProvider",
    "OfflineCacheMissError",
    "Provider",
    "ProviderContext",
    "ProviderError",
    "Registry",
    "WebProvider",
    "WikisourceProvider",
    "apply_filter",
    "default_cache_root",
    "default_registry",
]
