"""Draw values from the scalar / range / weighted-choice recipe fields.

The recipe schema (see ``docs/specs/02-recipe-format.md``) lets every
"varying value" be one of three forms. The render layer needs to pull
one concrete value per sample; this module is the single home for
that draw.
"""

from __future__ import annotations

from random import Random

from pd_ocr_synth.recipe.models import ColorSpec, Range, WeightedChoice


def sample_value(field, rng: Random):
    """Draw one scalar from a scalar / Range / WeightedChoice field.

    - ``int`` / ``float`` returned as-is (no draw).
    - ``Range[int]`` → ``rng.randint(min, max)``.
    - ``Range[float]`` → ``rng.uniform(min, max)``.
    - ``list[WeightedChoice[T]]`` → choose by normalized weights.
    """

    if isinstance(field, Range):
        return _draw_range(field, rng)
    if isinstance(field, list) and field and isinstance(field[0], WeightedChoice):
        return weighted_choice(field, rng)
    return field


def _draw_range[T: (int, float)](r: Range[T], rng: Random) -> T:
    if isinstance(r.min, int) and isinstance(r.max, int):
        return rng.randint(r.min, r.max)
    return rng.uniform(float(r.min), float(r.max))


def weighted_choice(choices: list[WeightedChoice], rng: Random):
    """Pick one ``WeightedChoice.value`` honoring the weights."""

    if not choices:
        raise ValueError("weighted_choice: empty choices list")
    weights = [max(c.weight, 0.0) for c in choices]
    total = sum(weights)
    if total <= 0:
        # All zero / negative → uniform fallback.
        return rng.choice(choices).value
    pick = rng.uniform(0.0, total)
    acc = 0.0
    for choice, weight in zip(choices, weights, strict=True):
        acc += weight
        if pick <= acc:
            return choice.value
    return choices[-1].value


def sample_color(spec: ColorSpec, rng: Random) -> tuple[int, int, int]:
    """Draw an (R, G, B) triple from a ``ColorSpec``."""

    return (
        int(sample_value(spec.r, rng)),  # pyright: ignore[reportArgumentType]
        int(sample_value(spec.g, rng)),  # pyright: ignore[reportArgumentType]
        int(sample_value(spec.b, rng)),  # pyright: ignore[reportArgumentType]
    )
