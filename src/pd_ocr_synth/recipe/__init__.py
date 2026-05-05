"""Recipe schema, loader, and supporting types.

Public surface (M02):

- ``Recipe`` and friends — pydantic v2 models for the YAML contract
  defined in ``docs/specs/02-recipe-format.md``.
- ``load_recipe(path)`` — parse a YAML file, expand ``~`` and
  ``${ENV_VAR}`` in path-like fields, resolve relative paths against the
  recipe's directory, and return a frozen ``Recipe``.
- ``RecipeLoadError`` — raised for syntactic / schema problems before
  semantic validation runs (see :mod:`pd_ocr_synth.validation`).
"""

from __future__ import annotations

from pd_ocr_synth.recipe.loader import RecipeLoadError, load_recipe
from pd_ocr_synth.recipe.models import (
    CorpusEntry,
    DegradationStage,
    Font,
    HFDatasetCorpus,
    Layout,
    LocalCorpus,
    OutputBlock,
    PublishBlock,
    Recipe,
    Rendering,
    TextTransform,
    WebCorpus,
    WikisourceCorpus,
)

__all__ = [
    "CorpusEntry",
    "DegradationStage",
    "Font",
    "HFDatasetCorpus",
    "Layout",
    "LocalCorpus",
    "OutputBlock",
    "PublishBlock",
    "Recipe",
    "RecipeLoadError",
    "Rendering",
    "TextTransform",
    "WebCorpus",
    "WikisourceCorpus",
    "load_recipe",
]
