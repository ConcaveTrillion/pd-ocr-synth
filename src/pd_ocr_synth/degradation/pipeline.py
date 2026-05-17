"""Stage protocol and pipeline runner.

A degradation pipeline is an ordered list of stages. Each stage is
applied with its declared probability; if the dice roll says skip,
the sample passes through unmodified. Order matters:

    geometric → optical → paper → noise → JPEG

(per ``docs/specs/07-degradation.md``). The runner does not enforce
this order; the recipe author is responsible. Tests guard against
reorderings that would obviously break output (see ``test_degradation``).

There are two flavors of stage:

- **Pixel-only** (``Stage`` protocol). The vast majority. Takes a
  ``PIL.Image`` plus the stage's options dict and the per-sample RNG
  and returns a new image. The bbox / glyph_runs do not change.
- **Geometry-aware** (``GeometryStage`` protocol). Used by stages that
  rotate, warp, or otherwise move pixels relative to the bbox (e.g.
  ``skew``). Takes the whole ``RenderedSample`` so the stage can
  update bbox + glyph_runs alongside the image.

Both protocols are registered into the same ``REGISTRY`` dict by kind.
The runner inspects the registry entry's signature shape to decide
which call form to use; we tag each entry explicitly to keep the
dispatch trivial.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random
from typing import TYPE_CHECKING, Any, Literal, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image

    from pd_ocr_synth.recipe import DegradationStage
    from pd_ocr_synth.render.sample import RenderedSample


class DegradationError(Exception):
    """Raised when a stage cannot be applied."""


class Stage(Protocol):
    """Pixel-only stage. Image in, image out."""

    def __call__(self, image: Image, options: dict[str, Any], rng: Random) -> Image: ...


class GeometryStage(Protocol):
    """Geometry-aware stage. Sample in, sample out (image + bbox updated together)."""

    def __call__(
        self, sample: RenderedSample, options: dict[str, Any], rng: Random
    ) -> RenderedSample: ...


@dataclass(frozen=True, slots=True)
class _RegistryEntry:
    """One entry in :data:`REGISTRY`.

    ``shape`` distinguishes the call form. Pixel stages get a much
    cheaper path (no bbox plumbing); geometric stages see the whole
    sample so they can update bbox/glyph_runs in lockstep.
    """

    fn: Stage | GeometryStage
    shape: Literal["pixel", "geometry"]


# Built-ins register themselves into this dict at module import time.
# Populated by :mod:`pd_ocr_synth.degradation.builtins` (and similar
# for geometry stages). M06 keeps this closed-over; the plugin
# extension point arrives with M09 and is documented in spec 09.
REGISTRY: dict[str, _RegistryEntry] = {}


def register_pixel_stage(kind: str, fn: Stage) -> None:
    """Register a pixel-only stage by ``kind``."""

    REGISTRY[kind] = _RegistryEntry(fn=fn, shape="pixel")


def register_geometry_stage(kind: str, fn: GeometryStage) -> None:
    """Register a geometry-aware stage by ``kind``."""

    REGISTRY[kind] = _RegistryEntry(fn=fn, shape="geometry")


def apply_degradation(
    sample: RenderedSample,
    stages: list[DegradationStage],
    *,
    rng: Random,
) -> RenderedSample:
    """Run ``stages`` over ``sample`` in order.

    Per-stage probability is drawn from ``rng`` first; if the stage
    runs, all its inner sampling also draws from ``rng``. This means
    a stage that's skipped by the probability gate consumes exactly
    one ``rng.random()`` call — which keeps the per-sample RNG state
    well-defined regardless of which stages activate, and stable
    against future stage-internal-sampling changes (the gate does not
    depend on the internals).
    """

    # Lazy import keeps a render-only path (preview without
    # degradation) from paying the builtins import cost.
    _ensure_builtins_registered()

    current = sample
    for stage_cfg in stages:
        if rng.random() >= stage_cfg.probability:
            continue
        entry = REGISTRY.get(stage_cfg.kind)
        if entry is None:
            raise DegradationError(
                f"unknown degradation kind {stage_cfg.kind!r}; known: {sorted(REGISTRY)}"
            )
        # ``model_extra`` carries every key beyond ``kind`` /
        # ``probability``. That's the stage's options dict.
        options: dict[str, Any] = dict(stage_cfg.model_extra or {})
        if entry.shape == "pixel":
            new_image = entry.fn(current.image, options, rng)  # pyright: ignore[reportArgumentType,reportAttributeAccessIssue]
            if new_image is current.image:  # pyright: ignore[reportAttributeAccessIssue]
                continue
            current = replace(current, image=new_image)  # pyright: ignore[reportArgumentType]
        else:
            current = entry.fn(current, options, rng)  # pyright: ignore[reportArgumentType,reportReturnType]
    return current  # pyright: ignore[reportReturnType]


_BUILTINS_REGISTERED = False


def _ensure_builtins_registered() -> None:
    """Idempotent registration trigger.

    Importing ``pd_ocr_synth.degradation.builtins`` performs the
    ``register_*_stage`` calls as a side effect of module import. The
    flag prevents re-running the import dance on every sample.
    """

    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return

    from pd_ocr_synth.degradation import builtins  # noqa: F401

    _BUILTINS_REGISTERED = True  # pyright: ignore[reportConstantRedefinition]
