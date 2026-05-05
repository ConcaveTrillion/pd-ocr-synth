"""Recipe search-path resolution.

A recipe argument on the CLI may be either a path or a name. Names are
resolved against the search path defined in ``docs/specs/01-cli.md``:

1. ``$PD_OCR_SYNTH_RECIPES`` (colon-separated)
2. ``./recipes/`` relative to CWD
3. ``<package>/recipes/`` shipped with the install

For each entry, both ``<dir>/<name>.yaml`` and ``<dir>/<name>/recipe.yaml``
are recognized — the latter mirrors the layout produced by
``pd-ocr-synth init``.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

ENV_VAR = "PD_OCR_SYNTH_RECIPES"


@dataclass(frozen=True, slots=True)
class RecipeEntry:
    """One discovered recipe."""

    name: str
    path: Path


class RecipeNotFoundError(Exception):
    """Raised when a recipe name does not resolve on the search path."""


def search_path(*, cwd: Path | None = None) -> list[Path]:
    """Return the search path in priority order.

    ``cwd`` is overridable for testability; defaults to the process CWD
    at call time. The packaged ``<pkg>/recipes/`` is appended last and
    silently dropped if it does not exist.
    """

    paths: list[Path] = []
    env_value = os.environ.get(ENV_VAR, "")
    if env_value:
        for entry in env_value.split(os.pathsep):
            entry = entry.strip()
            if entry:
                paths.append(Path(entry))

    paths.append((cwd if cwd is not None else Path.cwd()) / "recipes")

    pkg_recipes = Path(__file__).resolve().parent / "recipes"
    if pkg_recipes not in paths:
        paths.append(pkg_recipes)

    return paths


def _iter_recipes_in(directory: Path) -> Iterable[RecipeEntry]:
    """Yield recipes found directly in ``directory``.

    Two layouts recognized:
      - ``<dir>/<name>.yaml``                → name = ``<name>``
      - ``<dir>/<name>/recipe.yaml``         → name = ``<name>``
    """

    if not directory.exists() or not directory.is_dir():
        return
    seen: set[str] = set()
    # Flat .yaml / .yml files first (so they take precedence on
    # duplicate names).
    for child in sorted(directory.iterdir()):
        if child.is_file() and child.suffix in {".yaml", ".yml"}:
            name = child.stem
            if name in seen:
                continue
            seen.add(name)
            yield RecipeEntry(name=name, path=child)
    # Subdirectory layout (``<name>/recipe.yaml``).
    for child in sorted(directory.iterdir()):
        if not child.is_dir():
            continue
        candidate = child / "recipe.yaml"
        if candidate.exists():
            name = child.name
            if name in seen:
                continue
            seen.add(name)
            yield RecipeEntry(name=name, path=candidate)


def iter_recipes(*, cwd: Path | None = None) -> list[RecipeEntry]:
    """List recipes across the entire search path.

    Earlier entries on the search path shadow later ones with the same
    name. The list is sorted by name for stable CLI output.
    """

    seen: set[str] = set()
    found: list[RecipeEntry] = []
    for directory in search_path(cwd=cwd):
        for entry in _iter_recipes_in(directory):
            if entry.name in seen:
                continue
            seen.add(entry.name)
            found.append(entry)
    found.sort(key=lambda e: e.name)
    return found


def resolve_recipe(arg: str, *, cwd: Path | None = None) -> Path:
    """Resolve a recipe argument (name or path) to an absolute path.

    Resolution order:
      1. If ``arg`` looks like a path (contains a separator or has a
         YAML suffix) and that path exists, return its absolute form.
      2. Otherwise treat ``arg`` as a recipe name and look it up on
         the search path.

    Raises ``RecipeNotFoundError`` if neither succeeds.
    """

    # Path-like input takes precedence.
    if os.sep in arg or "/" in arg or arg.endswith((".yaml", ".yml")):
        candidate = Path(arg).expanduser()
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()

    # Search-path lookup.
    for entry in iter_recipes(cwd=cwd):
        if entry.name == arg:
            return entry.path.resolve()

    raise RecipeNotFoundError(
        f"recipe '{arg}' not found. Tried path lookup and search path: "
        + ", ".join(str(p) for p in search_path(cwd=cwd))
    )
