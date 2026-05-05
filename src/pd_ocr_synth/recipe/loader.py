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
    _expand_degradation_presets(data, source=p)

    recipe = Recipe.model_validate(data)
    return recipe.model_copy(update={"source_path": p})


def _expand_degradation_presets(data: dict[str, Any], *, source: Path) -> None:
    """Inline ``- preset: <name>`` entries inside ``degradation``.

    Per ``docs/specs/07-degradation.md``: a recipe may declare named
    groups under ``degradation_presets``, then reference them from the
    ordered ``degradation`` list. Resolution rules:

    - ``- preset: <name>`` expands to the list of stages declared
      under ``degradation_presets[<name>]``, in order, at the
      reference site.
    - Presets are local to one recipe (no cross-recipe lookup).
    - Presets cannot reference other presets — the expansion is a
      single pass, not recursive. This keeps the spec tractable and
      prevents accidental cycles.
    - Unknown preset names raise ``RecipeLoadError`` so the user gets
      a load-time failure instead of silently dropping stages.
    """

    degradation = data.get("degradation")
    presets = data.get("degradation_presets") or {}
    if not isinstance(degradation, list):
        return
    if not isinstance(presets, dict):
        raise RecipeLoadError(
            f"degradation_presets must be a mapping in {source}, got {type(presets).__name__}"
        )

    expanded: list[Any] = []
    for entry in degradation:
        if isinstance(entry, dict) and "preset" in entry and "kind" not in entry:
            name = entry["preset"]
            if not isinstance(name, str):
                raise RecipeLoadError(
                    f"degradation preset reference must be a string name; got {name!r} in {source}"
                )
            if name not in presets:
                raise RecipeLoadError(
                    f"degradation references unknown preset {name!r} in {source}; "
                    f"declared presets: {sorted(presets) or '[]'}"
                )
            stages = presets[name]
            if not isinstance(stages, list):
                raise RecipeLoadError(
                    f"degradation_presets[{name!r}] must be a list of stages in {source}"
                )
            expanded.extend(stages)
            continue
        expanded.append(entry)
    data["degradation"] = expanded
