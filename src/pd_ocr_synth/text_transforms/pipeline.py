"""Apply a list of transforms in order.

A ``PipelineStep`` is the user-facing description of one step:
``{name, options}``. ``apply_pipeline`` resolves names through the
registry, derives a per-step RNG from the seed, and threads the text
through them.
"""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass, field

from pd_ocr_synth.text_transforms.registry import Registry, default_registry


@dataclass(frozen=True, slots=True)
class PipelineStep:
    """One entry in a transforms pipeline."""

    name: str
    options: dict = field(default_factory=dict)


def _step_seed(base_seed: int, index: int) -> int:
    """Derive a stable per-step seed.

    Mixing the index into the seed keeps each step independent (one
    transform's RNG state cannot leak into the next), but keeps the
    whole pipeline reproducible from the recipe seed.
    """

    return (base_seed + 1) * 0x9E3779B1 ^ (index * 0x85EBCA77)


def apply_pipeline(
    text: str,
    steps: Iterable[PipelineStep | dict],
    *,
    seed: int = 0,
    registry: Registry | None = None,
) -> str:
    """Run ``text`` through ``steps`` in order and return the result.

    Each step may be a ``PipelineStep`` or a plain ``{name, options}``
    dict. The output of step *i* feeds step *i+1*. Determinism comes
    from a per-step RNG seeded by ``seed`` plus the step index.
    """

    reg = registry or default_registry()
    current = text
    for index, step in enumerate(steps):
        ps = step if isinstance(step, PipelineStep) else PipelineStep(**step)
        fn = reg.get(ps.name)
        rng = random.Random(_step_seed(seed, index))
        current = fn(current, dict(ps.options), rng)
    return current
