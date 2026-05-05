"""Tests for ``pd_ocr_synth.validation``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.recipe import load_recipe
from pd_ocr_synth.validation import (
    KNOWN_DEGRADATION_KINDS,
    ValidationReport,
    validate_recipe,
)


# Reused minimal recipe — kept fully in-memory so each test can mutate
# only what it needs and fix up paths against tmp_path.
def _minimal_yaml(*, font: str, dest: str, corpus: str) -> str:
    return f"""
schema_version: 1
name: minimal
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {dest}
  count: 100
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {font}
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
def good_recipe(tmp_path: Path):
    """Build an in-memory recipe whose paths all exist + dest is writable."""
    font = tmp_path / "fake.otf"
    font.write_bytes(b"")
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hello world\n", encoding="utf-8")
    dest = tmp_path / "out"
    yaml_text = _minimal_yaml(font=str(font), dest=str(dest), corpus=str(corpus))
    recipe_path = tmp_path / "recipe.yaml"
    recipe_path.write_text(yaml_text, encoding="utf-8")
    return load_recipe(recipe_path)


def test_minimal_recipe_validates_clean(good_recipe) -> None:
    report = validate_recipe(good_recipe)
    assert isinstance(report, ValidationReport)
    assert report.is_ok, [i.format() for i in report.issues]
    assert report.errors == ()


def test_missing_font_is_error(tmp_path: Path) -> None:
    yaml_text = _minimal_yaml(
        font=str(tmp_path / "ghost.otf"),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "font_missing" in codes


def test_missing_optional_font_is_warning(tmp_path: Path) -> None:
    seed = _make_file(tmp_path / "seed.txt")
    yaml_text = _minimal_yaml(
        font=str(tmp_path / "ghost.otf"),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    # Mark the font optional.
    yaml_text = yaml_text.replace(
        f"  - path: {tmp_path}/ghost.otf",
        f"  - path: {tmp_path}/ghost.otf\n    optional: true",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    assert report.errors == ()
    codes = [i.code for i in report.warnings]
    assert "optional_font_missing" in codes


def test_missing_local_corpus_is_error(tmp_path: Path) -> None:
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(tmp_path / "no-seed.txt"),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "local_corpus_missing" in codes


def test_unresolved_env_var_in_destination_is_error(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("DEFINITELY_UNSET_VAR", raising=False)
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest="${DEFINITELY_UNSET_VAR}/out",
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "output_destination_unresolved" in codes


def test_unwritable_destination_is_error(tmp_path: Path) -> None:
    # /proc/1 has no writable ancestor for our user.
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest="/proc/1/no-write/here",
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "output_destination_unwritable" in codes


def test_unknown_degradation_kind_is_error(tmp_path: Path) -> None:
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += "degradation:\n  - kind: not_a_real_stage\n    probability: 0.5\n"
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "degradation_kind_unknown" in codes


def test_paper_texture_missing_directory_key_is_error(tmp_path: Path) -> None:
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += "degradation:\n  - kind: paper_texture\n    probability: 0.5\n"
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "paper_texture_missing_directory" in codes


def test_paper_texture_directory_missing_is_error(tmp_path: Path) -> None:
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += (
        "degradation:\n"
        "  - kind: paper_texture\n"
        "    probability: 0.5\n"
        f"    directory: {tmp_path / 'no-textures'}\n"
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "paper_texture_directory_missing" in codes


def test_layout_mode_warns_on_unused_keys(tmp_path: Path) -> None:
    # word_crops + line_spacing → key is set but mode does not use it.
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        "layout:\n  mode: word_crops\n  padding_px: 8\n  line_spacing: 1.2\n",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.warnings]
    assert "layout_key_unused" in codes


def test_known_degradation_set_includes_canonical_kinds() -> None:
    # Spot-check the catalog matches docs/specs/07-degradation.md.
    for k in ("skew", "blur", "paper_texture", "jpeg", "noise", "ink_bleed"):
        assert k in KNOWN_DEGRADATION_KINDS


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_file(p: Path, content: str = "") -> Path:
    p.write_text(content, encoding="utf-8")
    return p


def _write(dirpath: Path, yaml_text: str, name: str = "recipe.yaml") -> Path:
    rp = dirpath / name
    rp.write_text(yaml_text, encoding="utf-8")
    return rp
