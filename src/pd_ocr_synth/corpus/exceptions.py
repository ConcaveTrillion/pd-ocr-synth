"""Corpus-layer exceptions.

Kept in their own module so that ``cache.py`` and ``registry.py`` can
import them without pulling each other in.
"""

from __future__ import annotations


class CorpusError(Exception):
    """Base class for any corpus-layer failure."""


class ProviderError(CorpusError):
    """Raised by a provider when fetch fails for non-transient reasons."""


class OfflineCacheMissError(CorpusError):
    """Raised when the cache lacks a key and ``offline=True`` forbids fetching."""
