"""Drift guard: ``docs/specs/recipe.schema.json`` ↔ pydantic ``Recipe`` model.

The JSON schema artifact at ``docs/specs/recipe.schema.json`` is generated
by ``make schema`` (which invokes ``pd-ocr-synth schema -o ...``) from
``Recipe.model_json_schema()``. There is no automatic hook that runs the
regen, so a contributor can land a Recipe-model field change without
remembering to refresh the artifact — leaving the published JSON-Schema
contract out of sync with the code.

This test fails CI when that happens. It re-runs the exact same regen
the ``schema`` CLI command does (in-process, no subprocess) and compares
byte-for-byte to the checked-in artifact. On drift it tells the
contributor exactly which command to run to fix it.

Iter 69 (commit ``c762238``) ad-hoc verified byte-identity once; this
test locks the contract permanently.
"""

from __future__ import annotations

import json
from pathlib import Path

from pd_ocr_synth.recipe.models import Recipe

# ---------------------------------------------------------------------------
# Locate the artifact relative to the repo root (this file lives in tests/).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_ARTIFACT = REPO_ROOT / "docs" / "specs" / "recipe.schema.json"


def _generate_schema_text() -> str:
    """Reproduce ``_cmd_schema``'s file-write content exactly.

    Mirrors ``src/pd_ocr_synth/cli.py::_cmd_schema``:
      - ``Recipe.model_json_schema()``
      - ``json.dumps(..., indent=2, sort_keys=False)``
      - trailing ``"\n"`` on file write
    """
    schema = Recipe.model_json_schema()
    return json.dumps(schema, indent=2, sort_keys=False) + "\n"


def test_schema_artifact_exists() -> None:
    """The checked-in artifact must exist at the spec'd path."""
    assert SCHEMA_ARTIFACT.is_file(), (
        f"Expected JSON-Schema artifact at {SCHEMA_ARTIFACT}; not found. "
        f"Run `make schema` to regenerate."
    )


def test_schema_artifact_matches_recipe_model() -> None:
    """Artifact must be byte-identical to a fresh regen from the model.

    If this fails, the pydantic ``Recipe`` model was changed without
    refreshing ``docs/specs/recipe.schema.json``. Run ``make schema``
    to regenerate the artifact, then commit the result.
    """
    actual = SCHEMA_ARTIFACT.read_text(encoding="utf-8")
    expected = _generate_schema_text()

    if actual != expected:
        # Provide a short, actionable diagnostic. Avoid dumping the full
        # ~1k-line schema in pytest output: surface the structural diff.
        actual_obj = json.loads(actual)
        expected_obj = json.loads(expected)

        actual_top = set(actual_obj.get("properties", {}).keys())
        expected_top = set(expected_obj.get("properties", {}).keys())
        added = sorted(expected_top - actual_top)
        removed = sorted(actual_top - expected_top)

        actual_defs = set(actual_obj.get("$defs", {}).keys())
        expected_defs = set(expected_obj.get("$defs", {}).keys())
        defs_added = sorted(expected_defs - actual_defs)
        defs_removed = sorted(actual_defs - expected_defs)

        diag_lines = [
            "docs/specs/recipe.schema.json is out of sync with pd_ocr_synth.recipe.models.Recipe.",
            "Run `make schema` to regenerate, then commit the result.",
        ]
        if added or removed:
            diag_lines.append(f"Top-level properties: +{added} -{removed}")
        if defs_added or defs_removed:
            diag_lines.append(f"$defs: +{defs_added} -{defs_removed}")
        if not (added or removed or defs_added or defs_removed):
            # Drift is structural-but-deeper (descriptions, types, enums,
            # constraints). Surface a hint without the full diff blob.
            diag_lines.append(
                "Drift is in nested fields (types/enums/descriptions/"
                "constraints), not top-level keys."
            )
        raise AssertionError("\n".join(diag_lines))


def test_schema_artifact_is_valid_json_with_recipe_title() -> None:
    """Cheap shape check so a corrupted artifact fails fast and clearly."""
    payload = json.loads(SCHEMA_ARTIFACT.read_text(encoding="utf-8"))
    assert payload.get("title") == "Recipe"
    assert "properties" in payload
    assert "$defs" in payload
