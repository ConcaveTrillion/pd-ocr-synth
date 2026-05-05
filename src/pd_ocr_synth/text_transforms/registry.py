"""Text-transform registry + Transform protocol."""

from __future__ import annotations

from random import Random
from typing import Protocol, runtime_checkable


@runtime_checkable
class Transform(Protocol):
    """Runtime protocol every text transform satisfies.

    A transform is a pure function: ``(text, options, rng) -> str``.
    It must be deterministic given the rng — no os-level randomness,
    no time-based behavior. Anything stochastic should pull from
    ``rng`` so the recipe seed propagates.
    """

    def __call__(self, text: str, options: dict, rng: Random) -> str: ...


class UnknownTransformError(KeyError):
    """Raised when a recipe names a transform the registry doesn't know."""


class Registry:
    """Maps a transform ``name`` to a callable.

    Built-ins are registered eagerly via ``register_builtins``;
    third-party transforms register through the
    ``pd_ocr_synth.text_transforms`` entry-point group, loaded lazily
    on first miss.
    """

    def __init__(self) -> None:
        self._transforms: dict[str, Transform] = {}
        self._loaded_entry_points = False

    def register(self, name: str, fn: Transform) -> None:
        if name in self._transforms and self._transforms[name] is not fn:
            raise ValueError(f"transform '{name}' is already registered with a different callable")
        self._transforms[name] = fn

    def get(self, name: str) -> Transform:
        if name not in self._transforms and not self._loaded_entry_points:
            self._load_entry_points()
        try:
            return self._transforms[name]
        except KeyError as exc:
            raise UnknownTransformError(
                f"unknown text transform '{name}'. Registered: {sorted(self._transforms)}"
            ) from exc

    def names(self) -> list[str]:
        if not self._loaded_entry_points:
            self._load_entry_points()
        return sorted(self._transforms)

    def _load_entry_points(self) -> None:
        self._loaded_entry_points = True
        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover
            return
        try:
            eps = entry_points(group="pd_ocr_synth.text_transforms")
        except Exception:  # pragma: no cover
            return
        for ep in eps:
            try:
                fn = ep.load()
            except Exception:  # pragma: no cover - third-party failure
                continue
            self.register(ep.name, fn)


_DEFAULT_REGISTRY: Registry | None = None


def default_registry() -> Registry:
    """Return the process-wide default registry, building it lazily."""

    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        registry = Registry()
        from pd_ocr_synth.text_transforms.builtins import register_builtins

        register_builtins(registry)
        _DEFAULT_REGISTRY = registry
    return _DEFAULT_REGISTRY
