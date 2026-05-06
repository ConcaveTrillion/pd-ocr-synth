"""Recipe linter — heuristic checks beyond schema validation.

Per ``docs/roadmap/10-stretch.md`` § Quality-of-life follow-ups:

    Recipe linter. Beyond schema validation, flag suspicious patterns:
    degradation always at probability 1.0, single font, no text
    transforms when the corpus is in modern spelling.

These are *style* issues — a recipe that flunks every lint check still
renders correctly. Lint output is therefore purely advisory: every
issue surfaced here uses ``severity="warning"``. Hard errors live in
:mod:`pd_ocr_synth.validation`.

The CLI :func:`pd_ocr_synth.cli._cmd_lint` runs full schema validation
first and only then layers lint on top, so a malformed recipe never
reaches this module.
"""

from __future__ import annotations

from pd_ocr_synth.recipe import Recipe
from pd_ocr_synth.validation import ValidationIssue, ValidationReport

# Lower bound below which we warn that the dataset is probably too
# small to train against. Picked to roughly match the smallest
# experiments observed in the trainer's docs (~100 samples for a
# smoke run); anything below that almost certainly means a forgotten
# ``--count`` flag or a misconfigured recipe.
SMALL_SAMPLE_THRESHOLD = 100


def lint_recipe(recipe: Recipe) -> ValidationReport:
    """Return a :class:`ValidationReport` containing only lint warnings.

    All issues produced here have severity ``warning`` — a recipe that
    fails lint is still renderable. The CLI surfaces both validation
    errors and lint warnings; lint alone never exits non-zero.
    """

    issues: list[ValidationIssue] = []
    issues.extend(_lint_degradation_probabilities(recipe))
    issues.extend(_lint_font_diversity(recipe))
    issues.extend(_lint_text_transforms(recipe))
    issues.extend(_lint_sample_count(recipe))
    issues.extend(_lint_seed(recipe))
    return ValidationReport(issues=tuple(issues))


def _lint_degradation_probabilities(recipe: Recipe) -> list[ValidationIssue]:
    """Warn when every degradation stage runs unconditionally.

    A pipeline where every stage has ``probability=1.0`` produces the
    same degraded output for every sample — defeating the point of
    randomized augmentation. Recipes typically vary at least one
    stage's probability so the trainer sees a mix of clean and
    degraded inputs.
    """

    if not recipe.degradation:
        return []
    if all(stage.probability >= 1.0 for stage in recipe.degradation):
        return [
            ValidationIssue(
                severity="warning",
                code="lint_degradation_always_certain",
                message=(
                    f"all {len(recipe.degradation)} degradation stage(s) have "
                    "probability=1.0; every sample receives identical "
                    "augmentation. Consider lowering at least one "
                    "stage's probability so the model sees a mix of "
                    "clean and degraded inputs."
                ),
                location="degradation",
            )
        ]
    return []


def _lint_font_diversity(recipe: Recipe) -> list[ValidationIssue]:
    """Warn when only a single (mandatory) font is declared.

    Models trained on a single typeface tend to overfit to its
    rasterization quirks. Two or more fonts — or a primary plus
    optional fallbacks — almost always produces a more robust
    recogniser. Optional fonts count: a recipe that lists e.g. one
    mandatory + one optional font is fine (the loader will fall
    back to the mandatory one when the optional file is missing,
    but the recipe author has expressed an intent to diversify).
    """

    if len(recipe.fonts) <= 1:
        return [
            ValidationIssue(
                severity="warning",
                code="lint_single_font",
                message=(
                    f"recipe declares {len(recipe.fonts)} font(s); a "
                    "single typeface tends to overfit. Consider adding "
                    "additional fonts (mandatory or optional) to "
                    "diversify the rendering."
                ),
                location="fonts",
            )
        ]
    return []


def _lint_text_transforms(recipe: Recipe) -> list[ValidationIssue]:
    """Warn when no text transforms are declared.

    Most historical / specialty-typography recipes need at least one
    transform — e.g. ``tironian_et`` for Gaelic, long-s substitution
    for early-modern English, polytonic accent normalisation for
    Greek. A recipe with zero text transforms is suspicious unless
    the corpus is already pre-normalised to the target script.
    """

    if not recipe.text_transforms:
        return [
            ValidationIssue(
                severity="warning",
                code="lint_no_text_transforms",
                message=(
                    "recipe declares no text_transforms; if the corpus "
                    "is in modern spelling but the target script is "
                    "historical, the rendered text will not match the "
                    "expected typography. Add transforms or confirm "
                    "the corpus is already pre-normalised."
                ),
                location="text_transforms",
            )
        ]
    return []


def _lint_sample_count(recipe: Recipe) -> list[ValidationIssue]:
    """Warn when ``output.count`` is small enough to suggest a typo.

    Triggered below :data:`SMALL_SAMPLE_THRESHOLD`. Override via
    ``--count`` is fine for ad-hoc smoke runs; this lint catches the
    case where the *recipe itself* asks for a tiny dataset.
    """

    count = recipe.output.count
    if count < SMALL_SAMPLE_THRESHOLD:
        return [
            ValidationIssue(
                severity="warning",
                code="lint_low_sample_count",
                message=(
                    f"output.count={count} is below the suggested "
                    f"minimum of {SMALL_SAMPLE_THRESHOLD}. Use the "
                    "CLI ``--count`` flag for short smoke runs and "
                    "keep the recipe value at production scale."
                ),
                location="output.count",
            )
        ]
    return []


def _lint_seed(recipe: Recipe) -> list[ValidationIssue]:
    """Warn when ``seed`` is left at the default 0.

    Seed 0 is the schema default, so a recipe that omits ``seed``
    silently lands here. That's fine for a single-author project but
    means every render of every fork produces bit-identical samples,
    which is rarely what dataset publishers want. Surfaced as a
    gentle nudge to set an explicit seed.
    """

    if recipe.seed == 0:
        return [
            ValidationIssue(
                severity="warning",
                code="lint_seed_default",
                message=(
                    "seed=0 is the schema default; every render of "
                    "this recipe across forks will produce identical "
                    "samples. Consider setting an explicit seed in "
                    "the recipe."
                ),
                location="seed",
            )
        ]
    return []
