"""CLI tests for ``pd-ocr-synth lint`` (M10 stretch).

Exit-code matrix per the M10 contract:

- ``0`` for clean recipe (no warnings, no errors).
- ``0`` for warnings-only output (lint warnings never fail).
- ``2`` for pydantic structural failure (missing required keys).
- ``3`` for ``validate_recipe`` errors (e.g. font path missing).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.cli import main


# Production-shaped recipe — every lint heuristic passes. Mirrors
# ``tests/test_lint.py`` ``_full_yaml`` but inlined here so changes to
# one fixture don't subtly break the other.
def _clean_yaml(*, font: str, font2: str, dest: str, corpus: str) -> str:
    return f"""\
schema_version: 1
name: cli-lint-clean
seed: 42
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {dest}
  count: 10000
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {font}
  - path: {font2}
text_transforms:
  - normalize_whitespace
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color: {{r: 0, g: 0, b: 0}}
  background_color: {{r: 255, g: 255, b: 255}}
layout:
  mode: word_crops
  padding_px: 4
degradation:
  - kind: blur
    probability: 0.3
"""


def _write_clean(tmp_path: Path, font_bytes: bytes) -> Path:
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hello world\n", encoding="utf-8")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(
        _clean_yaml(font=str(f1), font2=str(f2), dest=str(tmp_path / "out"), corpus=str(corpus)),
        encoding="utf-8",
    )
    return rp


def test_lint_subcommand_help_exits_zero(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["lint", "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "lint" in out.lower()


def test_lint_clean_recipe_exits_zero_no_warnings(
    tmp_path: Path,
    writable_font_bytes: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _write_clean(tmp_path, writable_font_bytes)
    rc = main(["lint", str(rp)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "no warnings" in captured.out


def test_lint_surfaces_validate_warnings(
    tmp_path: Path,
    writable_font_bytes: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`paragraph_alignment` on `lines` mode → existing validate
    warning surfaced through lint at exit 0."""
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = f"""\
schema_version: 1
name: cli-lint-paragraph-alignment-on-lines
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {tmp_path / "out"}
  count: 5000
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {f1}
  - path: {f2}
text_transforms:
  - normalize_whitespace
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color: {{r: 0, g: 0, b: 0}}
  background_color: {{r: 255, g: 255, b: 255}}
layout:
  mode: lines
  padding_px: 4
  paragraph_alignment: center
degradation:
  - kind: blur
    probability: 0.3
"""
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    rc = main(["lint", str(rp)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "layout_key_unused" in out
    assert "1 warning" in out


def test_lint_surfaces_lint_warnings(
    tmp_path: Path,
    writable_font_bytes: bytes,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A single-font recipe is otherwise valid → exit 0 with
    ``lint_single_font`` warning printed to stdout."""
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = f"""\
schema_version: 1
name: cli-lint-single-font
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {tmp_path / "out"}
  count: 5000
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {f1}
text_transforms:
  - normalize_whitespace
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color: {{r: 0, g: 0, b: 0}}
  background_color: {{r: 255, g: 255, b: 255}}
layout:
  mode: word_crops
  padding_px: 4
degradation:
  - kind: blur
    probability: 0.3
"""
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    rc = main(["lint", str(rp)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "lint_single_font" in out


def test_lint_validation_error_exits_three(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Recipe loads but validate finds a missing font → exit 3."""
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    # Two font paths declared but neither file exists on disk.
    yaml_text = f"""\
schema_version: 1
name: cli-lint-missing-fonts
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {tmp_path / "out"}
  count: 5000
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {tmp_path / "ghost1.otf"}
  - path: {tmp_path / "ghost2.otf"}
text_transforms:
  - normalize_whitespace
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color: {{r: 0, g: 0, b: 0}}
  background_color: {{r: 255, g: 255, b: 255}}
layout:
  mode: word_crops
  padding_px: 4
degradation:
  - kind: blur
    probability: 0.3
"""
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    rc = main(["lint", str(rp)])
    captured = capsys.readouterr()
    assert rc == 3
    assert "font_missing" in captured.err


def test_lint_pydantic_structural_failure_exits_two(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Recipe missing a required field (no fonts block) → exit 2."""
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = f"""\
schema_version: 1
name: cli-lint-no-fonts
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {tmp_path / "out"}
  count: 5000
corpus:
  - type: local
    path: {corpus}
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color: {{r: 0, g: 0, b: 0}}
  background_color: {{r: 255, g: 255, b: 255}}
layout:
  mode: word_crops
  padding_px: 4
"""
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    rc = main(["lint", str(rp)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "schema load failed" in captured.err


def test_lint_unknown_recipe_exits_three(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Unresolvable recipe name → exit 3 (same as ``validate``)."""
    monkeypatch.delenv("PD_OCR_SYNTH_RECIPES", raising=False)
    monkeypatch.chdir(tmp_path)
    rc = main(["lint", "definitely-not-a-recipe"])
    captured = capsys.readouterr()
    assert rc == 3
    assert "not found" in captured.err
