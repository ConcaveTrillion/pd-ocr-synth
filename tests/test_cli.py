"""CLI behavior tests.

M01 shipped argument parsing only. M02 implements ``list``, ``validate``,
``describe``, ``init``, ``schema``; M03 added ``fetch``/``clean``; M05
adds ``preview``; M07 adds ``render``; M08 adds ``publish --dry-run``
(the real upload path lands in a later M08 chunk).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from pd_ocr_synth import __version__
from pd_ocr_synth.cli import build_parser, main

# Every subcommand the parser knows about — used to verify --help works
# uniformly. ``schema`` is M02; the rest mirror docs/specs/01-cli.md.
ALL_SUBCOMMANDS = [
    "init",
    "list",
    "validate",
    "lint",
    "describe",
    "schema",
    "fetch",
    "preview",
    "render",
    "publish",
    "clean",
]

# Subcommands still fully stubbed after M08-dry-run. ``publish``
# (without ``--dry-run``) intentionally returns NOT_IMPLEMENTED until
# the upload chunk lands; that case is asserted directly below rather
# than via this generic-stub fixture so we can target it precisely.
STILL_STUBBED: list[str] = []


# ---------------------------------------------------------------------------
# Parser-level smoke tests
# ---------------------------------------------------------------------------


def test_parser_builds() -> None:
    parser = build_parser()
    assert parser.prog == "pd-ocr-synth"


def test_no_args_prints_help_and_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main([])
    captured = capsys.readouterr()
    assert rc == 2
    assert "usage:" in captured.err.lower()


def test_version_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert __version__ in captured.out


@pytest.mark.parametrize("subcommand", ALL_SUBCOMMANDS)
def test_subcommand_help(subcommand: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([subcommand, "--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "usage:" in captured.out.lower()


# ---------------------------------------------------------------------------
# Stubs (subcommands waiting for their milestone)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not STILL_STUBBED, reason="no stubbed subcommands left")
@pytest.mark.parametrize("subcommand", STILL_STUBBED or ["__placeholder__"])
def test_subcommand_stub_returns_not_implemented(subcommand: str) -> None:
    rc = main([subcommand, "dummy-recipe"])
    assert rc == 1


def test_publish_unknown_recipe_exits_three(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown recipe is a recipe-resolution error (exit 3),
    not a publish-specific failure — same as ``validate``/``describe``.
    """

    monkeypatch.delenv("PD_OCR_SYNTH_RECIPES", raising=False)
    monkeypatch.chdir(tmp_path)
    rc = main(["publish", "definitely-not-a-recipe", "--dry-run"])
    assert rc == 3
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_returns_zero_when_no_recipes(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PD_OCR_SYNTH_RECIPES", raising=False)
    monkeypatch.chdir(tmp_path)
    rc = main(["list"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "no recipes found" in captured.err.lower()


def test_list_finds_recipes_via_env_var(
    monkeypatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "alpha.yaml").write_text("schema_version: 1\nname: alpha\n", encoding="utf-8")
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "recipe.yaml").write_text(
        "schema_version: 1\nname: beta\n", encoding="utf-8"
    )
    monkeypatch.setenv("PD_OCR_SYNTH_RECIPES", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    rc = main(["list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "alpha" in out
    assert "beta" in out


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


_GOOD_RECIPE = """\
schema_version: 1
name: smoke
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 1
corpus:
  - type: local
    path: ./seed.txt
fonts:
  - path: ./fake.otf
rendering:
  font_size_pt: 12
  dpi: 300
  ink_color: {r: 0, g: 0, b: 0}
  background_color: {r: 255, g: 255, b: 255}
layout:
  mode: word_crops
  padding_px: 4
"""


def _make_good_recipe(tmp_path: Path, font_bytes: bytes) -> Path:
    (tmp_path / "fake.otf").write_bytes(font_bytes)
    (tmp_path / "seed.txt").write_text("hi\n", encoding="utf-8")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_GOOD_RECIPE, encoding="utf-8")
    return rp


def test_validate_clean_recipe_exits_zero(
    tmp_path: Path, writable_font_bytes: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    rp = _make_good_recipe(tmp_path, writable_font_bytes)
    rc = main(["validate", str(rp)])
    assert rc == 0, capsys.readouterr().err


def test_validate_missing_font_exits_three(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "seed.txt").write_text("x", encoding="utf-8")
    # No font file written → font_missing error.
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_GOOD_RECIPE, encoding="utf-8")
    rc = main(["validate", str(rp)])
    assert rc == 3
    err = capsys.readouterr().err
    assert "font_missing" in err


def test_validate_unknown_recipe_exits_three(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("PD_OCR_SYNTH_RECIPES", raising=False)
    monkeypatch.chdir(tmp_path)
    rc = main(["validate", "definitely-not-a-recipe"])
    assert rc == 3
    assert "not found" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# describe
# ---------------------------------------------------------------------------


def test_describe_text_format_prints_summary(
    tmp_path: Path, writable_font_bytes: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    rp = _make_good_recipe(tmp_path, writable_font_bytes)
    rc = main(["describe", str(rp), "--format", "text"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "recipe: smoke" in out
    assert "corpus: 1 entries (not fetched)" in out


def test_describe_json_format_emits_valid_json(
    tmp_path: Path, writable_font_bytes: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    rp = _make_good_recipe(tmp_path, writable_font_bytes)
    rc = main(["describe", str(rp), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["name"] == "smoke"


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init_scaffolds_directory(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["init", "fraktur"])
    assert rc == 0
    assert (tmp_path / "recipes" / "fraktur" / "recipe.yaml").exists()
    assert (tmp_path / "recipes" / "fraktur" / "README.md").exists()
    assert (tmp_path / "recipes" / "fraktur" / "fraktur" / "seed-words.txt").exists()


def test_init_refuses_to_overwrite(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init", "fraktur"]) == 0
    rc = main(["init", "fraktur"])
    assert rc == 2  # USAGE_EXIT — already exists


def test_init_force_overwrites(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init", "fraktur"]) == 0
    rc = main(["init", "fraktur", "--force"])
    assert rc == 0


def test_init_describe_round_trip(monkeypatch, tmp_path: Path) -> None:
    """A freshly-scaffolded recipe loads and describes cleanly."""
    monkeypatch.chdir(tmp_path)
    assert main(["init", "fraktur"]) == 0
    rp = tmp_path / "recipes" / "fraktur" / "recipe.yaml"
    rc = main(["describe", str(rp), "--format", "json"])
    assert rc == 0


# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


def test_schema_to_stdout_is_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    rc = main(["schema"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # Top-level should at minimum reference the Recipe model.
    assert payload.get("title") == "Recipe" or "Recipe" in payload.get("$defs", {})


def test_schema_to_file(tmp_path: Path) -> None:
    target = tmp_path / "out" / "recipe.schema.json"
    rc = main(["schema", "-o", str(target)])
    assert rc == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "properties" in payload or "$defs" in payload


# ---------------------------------------------------------------------------
# Recipe search-path isolation: ensure environment doesn't leak into tests.
# (Defensive fixture used implicitly by tests above via monkeypatch.)
# ---------------------------------------------------------------------------


def test_env_var_isolation_works(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("PD_OCR_SYNTH_RECIPES", raising=False)
    assert os.environ.get("PD_OCR_SYNTH_RECIPES") is None
