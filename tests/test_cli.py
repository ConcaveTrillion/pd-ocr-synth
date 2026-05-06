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


def test_describe_text_summary_covers_top_level_fields(
    tmp_path: Path, writable_font_bytes: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    """Lock the headline-summary contract.

    The JSON dump that follows the summary always contains the full
    resolved recipe, but the headline lines are what most authors
    actually read. This test pins the documented minimum: every line
    here has historically been present (and should remain so), or has
    been deliberately added because authors needed it before running
    ``preview`` / ``render``. New top-level Recipe fields should join
    this list rather than only appearing in the JSON tail.
    """
    rp = _make_good_recipe(tmp_path, writable_font_bytes)
    rc = main(["describe", str(rp), "--format", "text"])
    assert rc == 0
    out = capsys.readouterr().out
    # Order is also part of the contract: identity → output → corpus
    # → text → fonts → rendering → layout → degradation → publish.
    expected_prefixes = [
        "recipe:",
        "source:",
        "schema_version:",
        "seed:",
        "output.format:",
        "output.mode:",
        "output.destination:",
        "output.count:",
        "corpus:",
        "text_transforms:",
        "fonts:",
        "rendering.shaping_engine:",
        "rendering.font_size_pt:",
        "rendering.dpi:",
        "layout.mode:",
        "degradation:",
    ]
    last_idx = -1
    for prefix in expected_prefixes:
        # Each prefix should appear, and in order.
        line_starts = [i for i, line in enumerate(out.splitlines()) if line.startswith(prefix)]
        assert line_starts, f"describe text summary missing line starting with {prefix!r}"
        idx = line_starts[0]
        assert idx > last_idx, (
            f"describe summary line {prefix!r} out of order "
            f"(found at {idx}, previous at {last_idx})"
        )
        last_idx = idx


def test_describe_text_summary_omits_optional_unset_fields(
    tmp_path: Path, writable_font_bytes: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    """Optional/absent blocks must not produce noisy summary lines.

    The minimal smoke recipe has no description, no degradation_presets,
    and no publish block. None of those should print headline lines —
    otherwise authors get blank/0-valued noise that hides the live
    fields.
    """
    rp = _make_good_recipe(tmp_path, writable_font_bytes)
    rc = main(["describe", str(rp), "--format", "text"])
    assert rc == 0
    out = capsys.readouterr().out
    # Limit to the headline block (everything before the JSON dump).
    headline = out.split("--- resolved config (json) ---", 1)[0]
    assert "description:" not in headline
    assert "degradation_presets:" not in headline
    assert "publish.hf_dataset.repo:" not in headline


def test_describe_text_summary_surfaces_optional_when_set(
    tmp_path: Path, writable_font_bytes: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    """When optional fields are populated, the summary must surface them."""
    (tmp_path / "fake.otf").write_bytes(writable_font_bytes)
    (tmp_path / "seed.txt").write_text("hi\n", encoding="utf-8")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(
        """\
schema_version: 1
name: rich
description: a richly-populated recipe
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
degradation_presets:
  light:
    - kind: gaussian_blur
      probability: 0.5
degradation:
  - preset: light
publish:
  hf_dataset:
    repo: example/rich
""",
        encoding="utf-8",
    )
    rc = main(["describe", str(rp), "--format", "text"])
    assert rc == 0
    out = capsys.readouterr().out
    headline = out.split("--- resolved config (json) ---", 1)[0]
    assert "description: a richly-populated recipe" in headline
    assert "degradation_presets: 1 groups" in headline
    assert "publish.hf_dataset.repo: example/rich" in headline


def test_describe_text_summary_field_set_matches_recipe_model(
    tmp_path: Path, writable_font_bytes: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    """Meta-guard: every top-level Recipe field is reachable from describe.

    This is a regression net for "we added a field to the Recipe model
    but forgot to update describe". The text summary doesn't need to
    name every field by line, but the *JSON tail* must — and this test
    pins that contract: ``describe`` round-trips the resolved Recipe
    model dump, so every model field is reachable.
    """
    from pd_ocr_synth.recipe.models import Recipe

    rp = _make_good_recipe(tmp_path, writable_font_bytes)
    rc = main(["describe", str(rp), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # Every declared model field should appear as a key in the JSON
    # output (Pydantic's ``model_dump`` is exhaustive by default).
    declared = set(Recipe.model_fields.keys())
    missing = declared - set(payload.keys())
    assert not missing, f"describe JSON output missing top-level fields: {missing}"


def test_describe_json_round_trips_every_top_level_field(
    tmp_path: Path, writable_font_bytes: bytes, capsys: pytest.CaptureFixture[str]
) -> None:
    """JSON describe output must include the same shape ``schema`` advertises.

    This guards against describe accidentally calling ``model_dump``
    with ``exclude=...`` or filtering keys somewhere upstream.
    """
    from pd_ocr_synth.recipe.models import Recipe

    rp = _make_good_recipe(tmp_path, writable_font_bytes)
    rc = main(["describe", str(rp), "--format", "json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    # ``source_path`` is loader-injected and not part of the YAML
    # contract, but Recipe declares it — describe should expose it.
    for required in (
        "schema_version",
        "name",
        "description",
        "seed",
        "output",
        "corpus",
        "text_transforms",
        "fonts",
        "rendering",
        "layout",
        "degradation_presets",
        "degradation",
        "publish",
        "source_path",
    ):
        assert required in payload, (
            f"describe JSON missing {required!r}; present keys: {sorted(payload.keys())}"
        )
    # Sanity: model_fields and our explicit list agree.
    assert set(Recipe.model_fields.keys()) <= set(payload.keys())


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
