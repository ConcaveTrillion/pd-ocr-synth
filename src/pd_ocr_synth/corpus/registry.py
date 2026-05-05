"""Corpus provider registry + Provider protocol."""

from __future__ import annotations

from collections.abc import Iterable
from typing import ClassVar, Protocol, runtime_checkable

from pd_ocr_synth.corpus.context import ProviderContext
from pd_ocr_synth.corpus.exceptions import ProviderError


@runtime_checkable
class Provider(Protocol):
    """Runtime protocol every corpus provider satisfies.

    Mirrors ``docs/specs/09-extending.md``. Providers are stateless
    (one shared instance per ``type_name``); per-call configuration
    arrives via the ``options`` dict.
    """

    type_name: ClassVar[str]
    schema_version: ClassVar[int]

    def fetch(self, ctx: ProviderContext, options: dict) -> Iterable[str]: ...

    def cache_key(self, options: dict) -> str: ...


class Registry:
    """Maps ``type_name`` to a ``Provider`` instance.

    Built-in providers are registered eagerly; entry-point-based
    third-party providers (``pd_ocr_synth.corpus_providers`` group)
    are loaded lazily on the first lookup so import cost is paid only
    when the CLI actually needs them.
    """

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self._loaded_entry_points = False

    def register(self, provider: Provider) -> None:
        existing = self._providers.get(provider.type_name)
        if existing is not None and existing is not provider:
            raise ProviderError(
                f"provider type '{provider.type_name}' is already registered: "
                f"{existing!r}; refusing to replace with {provider!r}"
            )
        self._providers[provider.type_name] = provider

    def get(self, type_name: str) -> Provider:
        if type_name not in self._providers and not self._loaded_entry_points:
            self._load_entry_points()
        try:
            return self._providers[type_name]
        except KeyError as exc:
            raise ProviderError(
                f"unknown corpus provider '{type_name}'. Registered: {sorted(self._providers)}"
            ) from exc

    def types(self) -> list[str]:
        if not self._loaded_entry_points:
            self._load_entry_points()
        return sorted(self._providers)

    def _load_entry_points(self) -> None:
        # Entry-point loading is opt-in for the test environment but
        # always available at runtime. Keep failures non-fatal: a
        # broken third-party provider must not stop the CLI from
        # validating recipes that don't use it.
        self._loaded_entry_points = True
        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover
            return
        try:
            eps = entry_points(group="pd_ocr_synth.corpus_providers")
        except Exception:  # pragma: no cover
            return
        for ep in eps:
            try:
                provider_cls = ep.load()
                provider = provider_cls() if isinstance(provider_cls, type) else provider_cls
            except Exception:  # pragma: no cover - third-party failure
                continue
            self.register(provider)


_DEFAULT_REGISTRY: Registry | None = None


def default_registry() -> Registry:
    """Return the process-wide default registry, building it lazily."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = Registry()
        # Built-ins. Imported here to avoid an import cycle with the
        # package __init__.
        from pd_ocr_synth.corpus.providers.local import LocalProvider

        registry.register(LocalProvider())
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY
