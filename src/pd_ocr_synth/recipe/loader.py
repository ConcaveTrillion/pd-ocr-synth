"""YAML loader for recipe files.

Steps performed by :func:`load_recipe`:

1. Read the YAML file with ``yaml.safe_load``.
2. Resolve path-bearing keys (``~``, ``${ENV_VAR}``, relative-to-recipe)
   via :func:`pd_ocr_synth.recipe.paths.resolve_paths`.
3. Validate against the pydantic models in
   :mod:`pd_ocr_synth.recipe.models` and return a frozen ``Recipe``.

Errors at steps 1-2 raise :class:`RecipeLoadError`; errors at step 3
propagate as :class:`pydantic.ValidationError` so callers can format the
location info themselves (the CLI ``validate`` subcommand does this).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from pd_ocr_synth.recipe.models import Recipe
from pd_ocr_synth.recipe.paths import resolve_paths


class RecipeLoadError(Exception):
    """Raised when a recipe file cannot be read or parsed as YAML."""


def load_recipe(path: str | Path) -> Recipe:
    """Load and validate a recipe from disk.

    The returned ``Recipe`` carries a ``source_path`` attribute pointing
    at the absolute resolved path of the YAML file. All path-like
    fields inside the recipe have been resolved to absolute paths.
    """

    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise RecipeLoadError(f"recipe file not found: {p}")
    if not p.is_file():
        raise RecipeLoadError(f"recipe path is not a file: {p}")

    try:
        raw_text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecipeLoadError(f"could not read recipe {p}: {exc}") from exc

    try:
        data: Any = yaml.safe_load(raw_text)
    except yaml.YAMLError as exc:
        raise RecipeLoadError(f"YAML parse error in {p}: {exc}") from exc

    if data is None:
        raise RecipeLoadError(f"recipe file is empty: {p}")
    if not isinstance(data, dict):
        raise RecipeLoadError(f"recipe root must be a mapping, got {type(data).__name__}: {p}")

    resolve_paths(data, base_dir=p.parent)

    recipe = Recipe.model_validate(data)
    return recipe.model_copy(update={"source_path": p})
