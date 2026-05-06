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
def good_recipe(tmp_path: Path, writable_font_bytes: bytes):
    """Build an in-memory recipe whose paths all exist + dest is writable."""
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
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


@pytest.mark.parametrize("layout_mode", ["word_crops", "lines", "paragraphs"])
def test_paragraph_spacing_warns_on_non_pages_modes(
    tmp_path: Path,
    writable_font_bytes: bytes,
    layout_mode: str,
) -> None:
    """``paragraph_spacing`` is only meaningful for ``pages`` mode.

    Setting it on ``word_crops`` / ``lines`` / ``paragraphs`` (a single-
    paragraph sample) emits a ``layout_key_unused`` warning so the user
    knows the value will be ignored.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    # Build a mode-appropriate layout block plus the paragraph_spacing key.
    if layout_mode == "word_crops":
        new_layout = "layout:\n  mode: word_crops\n  padding_px: 8\n  paragraph_spacing: 1.4\n"
    else:
        new_layout = (
            f"layout:\n"
            f"  mode: {layout_mode}\n"
            f"  padding_px: 8\n"
            f"  max_width_px: 800\n"
            f"  paragraph_spacing: 1.4\n"
        )
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        new_layout,
    )
    # paragraphs/pages need detection output mode for the pairing check.
    if layout_mode in {"paragraphs", "pages"}:
        yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    warning_codes_at_paragraph_spacing = [
        i.code for i in report.warnings if i.location == "layout.paragraph_spacing"
    ]
    assert "layout_key_unused" in warning_codes_at_paragraph_spacing, [
        i.format() for i in report.issues
    ]


def test_paragraph_spacing_accepted_on_pages_mode(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """``paragraph_spacing`` is permitted on ``pages`` mode without warning."""
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        (
            "layout:\n"
            "  mode: pages\n"
            "  padding_px: 8\n"
            "  max_width_px: 800\n"
            "  paragraph_spacing: { min: 1.2, max: 1.8 }\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    # No layout_key_unused warning for paragraph_spacing on pages mode.
    paragraph_spacing_warnings = [
        i for i in report.warnings if i.location == "layout.paragraph_spacing"
    ]
    assert paragraph_spacing_warnings == [], [i.format() for i in paragraph_spacing_warnings]
    assert report.is_ok, [i.format() for i in report.issues]


@pytest.mark.parametrize("layout_mode", ["word_crops", "lines", "paragraphs"])
def test_paragraph_indent_px_warns_on_non_pages_modes(
    tmp_path: Path,
    writable_font_bytes: bytes,
    layout_mode: str,
) -> None:
    """``paragraph_indent_px`` is only meaningful for ``pages`` mode.

    Setting it on ``word_crops`` / ``lines`` / ``paragraphs`` (a single-
    paragraph sample, where indent would just add to the leading
    padding) emits a ``layout_key_unused`` warning so the user knows
    the value will be ignored.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    if layout_mode == "word_crops":
        new_layout = "layout:\n  mode: word_crops\n  padding_px: 8\n  paragraph_indent_px: 40\n"
    else:
        new_layout = (
            f"layout:\n"
            f"  mode: {layout_mode}\n"
            f"  padding_px: 8\n"
            f"  max_width_px: 800\n"
            f"  paragraph_indent_px: 40\n"
        )
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        new_layout,
    )
    if layout_mode in {"paragraphs", "pages"}:
        yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    warning_codes_at_indent = [
        i.code for i in report.warnings if i.location == "layout.paragraph_indent_px"
    ]
    assert "layout_key_unused" in warning_codes_at_indent, [i.format() for i in report.issues]


def test_paragraph_indent_px_accepted_on_pages_mode(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """``paragraph_indent_px`` is permitted on ``pages`` mode without warning."""
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        (
            "layout:\n"
            "  mode: pages\n"
            "  padding_px: 8\n"
            "  max_width_px: 800\n"
            "  paragraph_indent_px: 50\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    indent_warnings = [i for i in report.warnings if i.location == "layout.paragraph_indent_px"]
    assert indent_warnings == [], [i.format() for i in indent_warnings]
    assert report.is_ok, [i.format() for i in report.issues]


def test_known_degradation_set_includes_canonical_kinds() -> None:
    # Spot-check the catalog matches docs/specs/07-degradation.md.
    for k in ("skew", "blur", "paper_texture", "jpeg", "noise", "ink_bleed"):
        assert k in KNOWN_DEGRADATION_KINDS


# ---------------------------------------------------------------------------
# output.mode / layout.mode pairing (spec 08, §Modes)
# ---------------------------------------------------------------------------


def _swap_output_layout_modes(yaml_text: str, *, output_mode: str, layout_mode: str) -> str:
    """Swap the recognition/word_crops defaults emitted by ``_minimal_yaml``."""
    swapped = yaml_text.replace("mode: recognition", f"mode: {output_mode}")
    return swapped.replace("mode: word_crops", f"mode: {layout_mode}")


def _layout_block_for(layout_mode: str) -> str:
    """Render a minimal but mode-appropriate layout block.

    ``_minimal_yaml`` already includes ``padding_px: 8`` for word_crops;
    other modes add ``max_width_px`` so the keys-by-mode warning logic
    doesn't fire and obscure the pairing assertion under test.
    """
    if layout_mode == "word_crops":
        return "layout:\n  mode: word_crops\n  padding_px: 8\n"
    return f"layout:\n  mode: {layout_mode}\n  padding_px: 8\n  max_width_px: 800\n"


@pytest.mark.parametrize(
    ("output_mode", "layout_mode"),
    [
        ("recognition", "paragraphs"),
        ("recognition", "pages"),
        ("detection", "word_crops"),
        ("detection", "lines"),
    ],
)
def test_output_layout_mode_mismatch_is_error(
    tmp_path: Path,
    writable_font_bytes: bytes,
    output_mode: str,
    layout_mode: str,
) -> None:
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    # Replace the layout block wholesale so we can pick mode-appropriate keys.
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        _layout_block_for(layout_mode),
    )
    yaml_text = yaml_text.replace("mode: recognition", f"mode: {output_mode}")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "output_layout_mode_mismatch" in codes, [i.format() for i in report.issues]


@pytest.mark.parametrize(
    ("output_mode", "layout_mode"),
    [
        ("recognition", "word_crops"),
        ("recognition", "lines"),
        ("detection", "paragraphs"),
        ("detection", "pages"),
    ],
)
def test_output_layout_mode_pairing_valid_combinations_pass(
    tmp_path: Path,
    writable_font_bytes: bytes,
    output_mode: str,
    layout_mode: str,
) -> None:
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        _layout_block_for(layout_mode),
    )
    yaml_text = yaml_text.replace("mode: recognition", f"mode: {output_mode}")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "output_layout_mode_mismatch" not in codes, [i.format() for i in report.issues]


def test_output_layout_mode_mismatch_message_cites_spec(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _swap_output_layout_modes(
        _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed)),
        output_mode="detection",
        layout_mode="word_crops",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    msg = next(i.message for i in report.errors if i.code == "output_layout_mode_mismatch")
    assert "08-output-format.md" in msg
    # The message should hint at which layout modes are valid for detection.
    assert "paragraphs" in msg and "pages" in msg


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
