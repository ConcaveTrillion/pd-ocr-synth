"""Per-recipe environment passed to corpus providers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pd_ocr_synth.corpus.cache import CacheStore


@dataclass(frozen=True, slots=True)
class ProviderContext:
    """Read-only environment a provider sees during fetch.

    ``http`` is typed loosely as ``object`` here so tests + the web
    provider don't both have to import httpx; the actual httpx.Client
    is wired up in the M03 web provider commit.
    """

    recipe_dir: Path
    cache: CacheStore
    offline: bool = False
    logger: logging.Logger = logging.getLogger("pd_ocr_synth.corpus")
    http: object | None = None

    @property
    def cache_dir(self) -> Path:
        """Convenience accessor for ``ctx.cache.root`` — providers
        commonly want just the path."""
        return self.cache.root
