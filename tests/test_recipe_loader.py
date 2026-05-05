"""Tests for the YAML recipe loader and its supporting models."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from pd_ocr_synth.recipe import (
    LocalCorpus,
    Recipe,
    RecipeLoadError,
    WebCorpus,
    WikisourceCorpus,
    load_recipe,
)
from pd_ocr_synth.recipe.models import Range, WeightedChoice
from pd_ocr_synth.recipe.paths import expand_path

MINIMAL_RECIPE = """
schema_version: 1
name: minimal
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 100
corpus:
  - type: local
    path: ./seed.txt
fonts:
  - path: ./fake.otf
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color:
    r: 10
    g: 10
    b: 10
  background_color:
    r: 240
    g: 240
    b: 240
layout:
  mode: word_crops
  padding_px: 8
"""


@pytest.fixture
def write_recipe(tmp_path: Path):
    def _write(content: str, name: str = "recipe.yaml") -> Path:
        p = tmp_path / name
        p.write_text(content, encoding="utf-8")
        return p

    return _write


# ---------------------------------------------------------------------------
# end-to-end load
# ---------------------------------------------------------------------------


def test_load_gaelic_recipe(recipes_dir: str) -> None:
    recipe = load_recipe(Path(recipes_dir) / "gaelic.yaml")
    assert isinstance(recipe, Recipe)
    assert recipe.schema_version == 1
    assert recipe.name == "gaelic"
    assert recipe.output.mode == "recognition"
    assert recipe.layout.mode == "word_crops"
    assert recipe.fonts, "expected at least one font"
    assert any(isinstance(c, WebCorpus) for c in recipe.corpus)
    assert any(isinstance(c, LocalCorpus) for c in recipe.corpus)
    assert any(isinstance(c, WikisourceCorpus) for c in recipe.corpus)
    assert recipe.source_path is not None
    assert recipe.source_path.name == "gaelic.yaml"


def test_loaded_recipe_is_frozen(write_recipe) -> None:
    recipe = load_recipe(write_recipe(MINIMAL_RECIPE))
    with pytest.raises(ValidationError):
        # frozen models reject attribute assignment
        recipe.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# path resolution
# ---------------------------------------------------------------------------


def test_relative_paths_resolve_against_recipe_dir(write_recipe) -> None:
    recipe_path = write_recipe(MINIMAL_RECIPE)
    recipe = load_recipe(recipe_path)
    expected_dir = recipe_path.parent
    assert recipe.fonts[0].path == expected_dir / "fake.otf"
    assert recipe.output.destination == expected_dir / "out"
    assert isinstance(recipe.corpus[0], LocalCorpus)
    assert recipe.corpus[0].path == expected_dir / "seed.txt"


def test_absolute_paths_pass_through(tmp_path: Path, write_recipe) -> None:
    abs_font = tmp_path / "elsewhere" / "font.otf"
    yaml_text = MINIMAL_RECIPE.replace("./fake.otf", str(abs_font))
    recipe = load_recipe(write_recipe(yaml_text))
    assert recipe.fonts[0].path == abs_font


def test_env_var_expansion(monkeypatch, tmp_path: Path, write_recipe) -> None:
    monkeypatch.setenv("MY_OCR_OUT", str(tmp_path / "envroot"))
    yaml_text = MINIMAL_RECIPE.replace("./out", "${MY_OCR_OUT}/run1")
    recipe = load_recipe(write_recipe(yaml_text))
    assert recipe.output.destination == tmp_path / "envroot" / "run1"


def test_home_expansion(monkeypatch, tmp_path: Path, write_recipe) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    yaml_text = MINIMAL_RECIPE.replace("./fake.otf", "~/fonts/foo.otf")
    recipe = load_recipe(write_recipe(yaml_text))
    assert recipe.fonts[0].path == tmp_path / "home" / "fonts" / "foo.otf"


def test_expand_path_unsets_left_unmodified(tmp_path: Path) -> None:
    # An unset env var stays in place; the resolver does not raise.
    out = expand_path("${PROBABLY_UNSET_VAR_XYZ}/sub", tmp_path)
    assert out.endswith("${PROBABLY_UNSET_VAR_XYZ}/sub")


# ---------------------------------------------------------------------------
# scalar / range / weighted-choice forms
# ---------------------------------------------------------------------------


SCALAR_FORMS = pytest.mark.parametrize(
    ("yaml_value", "expected_kind", "check"),
    [
        ("14", "scalar", lambda v: v == 14),
        ("{ min: 10, max: 18 }", "range", lambda v: isinstance(v, Range) and v.max == 18),
        (
            "[{ value: 12, weight: 0.6 }, { value: 14, weight: 0.4 }]",
            "choice",
            lambda v: (
                isinstance(v, list)
                and all(isinstance(x, WeightedChoice) for x in v)
                and len(v) == 2
            ),
        ),
    ],
)


@SCALAR_FORMS
def test_font_size_supports_three_forms(
    write_recipe, yaml_value: str, expected_kind: str, check
) -> None:
    yaml_text = MINIMAL_RECIPE.replace("font_size_pt: 14", f"font_size_pt: {yaml_value}")
    recipe = load_recipe(write_recipe(yaml_text))
    assert check(recipe.rendering.font_size_pt), f"failed for {expected_kind}"


# ---------------------------------------------------------------------------
# text transforms — bare-name vs single-key-mapping forms
# ---------------------------------------------------------------------------


def test_text_transform_bare_name_form(write_recipe) -> None:
    yaml_text = MINIMAL_RECIPE + "\ntext_transforms:\n  - normalize_whitespace\n"
    recipe = load_recipe(write_recipe(yaml_text))
    assert len(recipe.text_transforms) == 1
    assert recipe.text_transforms[0].name == "normalize_whitespace"
    assert recipe.text_transforms[0].options == {}


def test_text_transform_options_form(write_recipe) -> None:
    yaml_text = (
        MINIMAL_RECIPE
        + "\ntext_transforms:\n"
        + "  - tironian_et:\n"
        + "      replace_words: ['agus', 'and']\n"
        + "      probability: 0.7\n"
    )
    recipe = load_recipe(write_recipe(yaml_text))
    t = recipe.text_transforms[0]
    assert t.name == "tironian_et"
    assert t.options == {"replace_words": ["agus", "and"], "probability": 0.7}


# ---------------------------------------------------------------------------
# corpus discriminator
# ---------------------------------------------------------------------------


def test_corpus_discriminator_rejects_unknown_type(write_recipe) -> None:
    yaml_text = MINIMAL_RECIPE.replace("type: local", "type: definitely-not-a-provider")
    with pytest.raises(ValidationError):
        load_recipe(write_recipe(yaml_text))


# ---------------------------------------------------------------------------
# loader error paths
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(RecipeLoadError, match="not found"):
        load_recipe(tmp_path / "no-such-file.yaml")


def test_invalid_yaml_raises(write_recipe) -> None:
    with pytest.raises(RecipeLoadError, match="YAML parse error"):
        load_recipe(write_recipe("schema_version: 1\nname: x\n  bad: indent\n"))


def test_empty_yaml_raises(write_recipe) -> None:
    with pytest.raises(RecipeLoadError, match="empty"):
        load_recipe(write_recipe(""))


def test_root_must_be_mapping_raises(write_recipe) -> None:
    with pytest.raises(RecipeLoadError, match="must be a mapping"):
        load_recipe(write_recipe("- a\n- b\n"))


def test_unsupported_schema_version_raises(write_recipe) -> None:
    yaml_text = MINIMAL_RECIPE.replace("schema_version: 1", "schema_version: 99")
    with pytest.raises(Exception, match="schema_version"):
        load_recipe(write_recipe(yaml_text))


def test_inverted_range_raises(write_recipe) -> None:
    yaml_text = MINIMAL_RECIPE.replace("font_size_pt: 14", "font_size_pt: { min: 30, max: 10 }")
    with pytest.raises(Exception, match=">= min"):
        load_recipe(write_recipe(yaml_text))
