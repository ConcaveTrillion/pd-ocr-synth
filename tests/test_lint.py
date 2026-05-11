"""Tests for ``pd_ocr_synth.lint`` (M10 stretch — recipe linter)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.lint import (
    LINT_CODES,
    SMALL_SAMPLE_THRESHOLD,
    lint_recipe,
)
from pd_ocr_synth.recipe import load_recipe


# A "production-shaped" recipe: two fonts, real seed, large count, a
# text transform, and at least one degradation stage with prob<1.0.
# Anything missing one of those is what each individual test mutates.
def _full_yaml(
    *,
    font: str,
    font2: str,
    dest: str,
    corpus: str,
    seed: int = 42,
    count: int = 10_000,
    text_transforms: str = "\ntext_transforms:\n  - normalize_whitespace\n",
    degradation: str = "\ndegradation:\n  - kind: blur\n    probability: 0.3\n",
) -> str:
    return f"""\
schema_version: 1
name: lint-fixture
seed: {seed}
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {dest}
  count: {count}
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {font}
  - path: {font2}{text_transforms}rendering:
  font_size_pt: 14
  dpi: 300
  ink_color: {{r: 0, g: 0, b: 0}}
  background_color: {{r: 255, g: 255, b: 255}}
layout:
  mode: word_crops
  padding_px: 4{degradation}"""


@pytest.fixture
def clean_recipe(tmp_path: Path, writable_font_bytes: bytes):
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hello world\n", encoding="utf-8")
    dest = tmp_path / "out"
    yaml_text = _full_yaml(font=str(f1), font2=str(f2), dest=str(dest), corpus=str(corpus))
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    return load_recipe(rp)


def test_clean_recipe_lints_clean(clean_recipe) -> None:
    report = lint_recipe(clean_recipe)
    assert report.warnings == (), [i.format() for i in report.warnings]
    assert report.is_ok


def test_clean_recipe_has_no_errors(clean_recipe) -> None:
    """Lint never produces errors — only warnings."""
    report = lint_recipe(clean_recipe)
    assert report.errors == ()


def test_lint_flags_all_certain_degradation(tmp_path: Path, writable_font_bytes: bytes) -> None:
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(tmp_path / "out"),
        corpus=str(corpus),
        # Two stages, both at probability 1.0 (the explicit default).
        degradation=(
            "\ndegradation:\n"
            "  - kind: blur\n    probability: 1.0\n"
            "  - kind: noise\n    probability: 1.0\n"
        ),
    )
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    codes = [i.code for i in lint_recipe(recipe).warnings]
    assert "lint_degradation_always_certain" in codes


def test_lint_does_not_flag_mixed_degradation_probabilities(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(tmp_path / "out"),
        corpus=str(corpus),
        degradation=(
            "\ndegradation:\n"
            "  - kind: blur\n    probability: 1.0\n"
            "  - kind: noise\n    probability: 0.4\n"
        ),
    )
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    codes = [i.code for i in lint_recipe(recipe).warnings]
    assert "lint_degradation_always_certain" not in codes


def test_lint_skips_degradation_check_when_no_stages(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """Empty degradation pipeline must not trigger the all-certain warning."""
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(tmp_path / "out"),
        corpus=str(corpus),
        degradation="",
    )
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    codes = [i.code for i in lint_recipe(recipe).warnings]
    assert "lint_degradation_always_certain" not in codes


def test_lint_flags_single_font(tmp_path: Path, writable_font_bytes: bytes) -> None:
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = f"""\
schema_version: 1
name: lone-font
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
"""
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    codes = [i.code for i in lint_recipe(recipe).warnings]
    assert "lint_single_font" in codes


def test_lint_flags_no_text_transforms(tmp_path: Path, writable_font_bytes: bytes) -> None:
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(tmp_path / "out"),
        corpus=str(corpus),
        text_transforms="\n",  # produces no text_transforms key at all
    )
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    codes = [i.code for i in lint_recipe(recipe).warnings]
    assert "lint_no_text_transforms" in codes


def test_lint_flags_low_sample_count(tmp_path: Path, writable_font_bytes: bytes) -> None:
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    # Pick a count strictly below the threshold.
    small_count = SMALL_SAMPLE_THRESHOLD - 1
    yaml_text = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(tmp_path / "out"),
        corpus=str(corpus),
        count=small_count,
    )
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    codes = [i.code for i in lint_recipe(recipe).warnings]
    assert "lint_low_sample_count" in codes


def test_lint_does_not_flag_count_at_threshold(tmp_path: Path, writable_font_bytes: bytes) -> None:
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(tmp_path / "out"),
        corpus=str(corpus),
        count=SMALL_SAMPLE_THRESHOLD,
    )
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    codes = [i.code for i in lint_recipe(recipe).warnings]
    assert "lint_low_sample_count" not in codes


def test_lint_flags_default_seed(tmp_path: Path, writable_font_bytes: bytes) -> None:
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(tmp_path / "out"),
        corpus=str(corpus),
        seed=0,
    )
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    codes = [i.code for i in lint_recipe(recipe).warnings]
    assert "lint_seed_default" in codes


def test_lint_does_not_flag_explicit_seed(clean_recipe) -> None:
    """The clean fixture sets seed=42 — must not trigger the seed lint."""
    codes = [i.code for i in lint_recipe(clean_recipe).warnings]
    assert "lint_seed_default" not in codes


def test_lint_flags_zero_weight_font(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """A font declared with ``weight: 0`` is never sampled — warn."""
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = f"""\
schema_version: 1
name: zero-weight
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
    weight: 1.0
  - path: {f2}
    weight: 0.0
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
"""
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    warnings = [i for i in lint_recipe(recipe).warnings if i.code == "lint_zero_weight_font"]
    assert len(warnings) == 1
    # Index of the zero-weight font surfaces in the message body.
    assert "1" in warnings[0].message


def test_lint_does_not_flag_positive_weights(clean_recipe) -> None:
    """All weights default to 1.0 in the clean fixture — no warning."""
    codes = [i.code for i in lint_recipe(clean_recipe).warnings]
    assert "lint_zero_weight_font" not in codes


def test_lint_flags_all_optional_fonts(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """A recipe whose every font is optional is a fetch-failure footgun."""
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = f"""\
schema_version: 1
name: all-optional
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
    optional: true
  - path: {f2}
    optional: true
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
"""
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    codes = [i.code for i in lint_recipe(recipe).warnings]
    assert "lint_all_optional_fonts" in codes


def test_lint_does_not_flag_mixed_mandatory_and_optional(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """One mandatory + one optional font is the recommended pattern."""
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = f"""\
schema_version: 1
name: mixed-optional
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
    optional: true
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
"""
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    codes = [i.code for i in lint_recipe(recipe).warnings]
    assert "lint_all_optional_fonts" not in codes


def test_lint_codes_are_all_warning_severity(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """A recipe that fails every check still produces only warnings."""
    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = f"""\
schema_version: 1
name: dirty
seed: 0
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {tmp_path / "out"}
  count: 1
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {f1}
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
    probability: 1.0
"""
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    report = lint_recipe(recipe)
    assert report.errors == ()
    # All five codes are independently expressible — assert the set
    # rather than ordering, since lint_recipe makes no ordering
    # guarantees beyond "deterministic across runs".
    codes = {i.code for i in report.warnings}
    assert codes == {
        "lint_degradation_always_certain",
        "lint_single_font",
        "lint_no_text_transforms",
        "lint_low_sample_count",
        "lint_seed_default",
    }
    assert all(i.severity == "warning" for i in report.warnings)


# ---------------------------------------------------------------------------
# LINT_CODES catalog drift guards
#
# ``LINT_CODES`` is the source-of-truth set the spec doc compares against.
# Both halves of the contract need locking:
#
#   1. Every emitted ``code`` must appear in LINT_CODES — a new helper
#      that adds a code without registering it would silently ship
#      undocumented behaviour.
#   2. Every entry in LINT_CODES must be reachable by *some* recipe —
#      a stale entry (left behind after a helper was removed) would
#      keep claiming a code the linter no longer emits.
#
# (1) is the cheap direction: runtime emission ⊆ catalog. (2) is the
# stronger guarantee — we exercise it by combining the codes seen across
# every "this lint fires" test fixture in this module.
# ---------------------------------------------------------------------------


def test_every_emitted_lint_code_is_in_lint_codes(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """A recipe that fails every check must only emit codes from LINT_CODES.

    The "fail every check" recipe below intentionally trips every
    helper — single mandatory font, no text_transforms, count below
    threshold, seed=0, all-certain degradation. If a future helper
    introduces a new code without registering it in ``LINT_CODES``,
    the catalog falls out of sync and the spec drifts; surface here.
    """

    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    yaml_text = f"""\
schema_version: 1
name: dirty-everything
seed: 0
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {tmp_path / "out"}
  count: 1
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {f1}
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
    probability: 1.0
"""
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    recipe = load_recipe(rp)
    emitted = {i.code for i in lint_recipe(recipe).warnings}
    leaked = emitted - LINT_CODES
    assert not leaked, (
        f"lint_recipe emitted code(s) not in LINT_CODES: {sorted(leaked)}. "
        "Add them to src/pd_ocr_synth/lint.py:LINT_CODES."
    )


def test_lint_codes_has_no_dead_entries(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """Every code in LINT_CODES must be reachable by at least one recipe.

    Combines the union of codes emitted by every "this lint fires"
    fixture in this file. If a code is in ``LINT_CODES`` but no
    fixture can hit it, either a helper was removed (and the entry
    is stale) or the test coverage is missing — either way, fail.

    This is the stronger half of the contract: it forces test
    coverage to follow the catalog, not the other way around.
    """

    f1 = tmp_path / "primary.otf"
    f1.write_bytes(writable_font_bytes)
    f2 = tmp_path / "secondary.otf"
    f2.write_bytes(writable_font_bytes)
    corpus = tmp_path / "seed.txt"
    corpus.write_text("hi\n", encoding="utf-8")
    dest = tmp_path / "out"

    seen: set[str] = set()

    # Trigger 1: all-certain degradation (otherwise-clean recipe).
    yaml1 = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(dest),
        corpus=str(corpus),
        degradation=(
            "\ndegradation:\n"
            "  - kind: blur\n    probability: 1.0\n"
            "  - kind: noise\n    probability: 1.0\n"
        ),
    )
    rp1 = tmp_path / "r1.yaml"
    rp1.write_text(yaml1, encoding="utf-8")
    seen.update(i.code for i in lint_recipe(load_recipe(rp1)).warnings)

    # Trigger 2: single mandatory font.
    yaml2 = f"""\
schema_version: 1
name: lone-font
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {dest}
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
"""
    rp2 = tmp_path / "r2.yaml"
    rp2.write_text(yaml2, encoding="utf-8")
    seen.update(i.code for i in lint_recipe(load_recipe(rp2)).warnings)

    # Trigger 3: no text_transforms.
    yaml3 = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(dest),
        corpus=str(corpus),
        text_transforms="\n",
    )
    rp3 = tmp_path / "r3.yaml"
    rp3.write_text(yaml3, encoding="utf-8")
    seen.update(i.code for i in lint_recipe(load_recipe(rp3)).warnings)

    # Trigger 4: low sample count.
    yaml4 = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(dest),
        corpus=str(corpus),
        count=SMALL_SAMPLE_THRESHOLD - 1,
    )
    rp4 = tmp_path / "r4.yaml"
    rp4.write_text(yaml4, encoding="utf-8")
    seen.update(i.code for i in lint_recipe(load_recipe(rp4)).warnings)

    # Trigger 5: default seed=0.
    yaml5 = _full_yaml(
        font=str(f1),
        font2=str(f2),
        dest=str(dest),
        corpus=str(corpus),
        seed=0,
    )
    rp5 = tmp_path / "r5.yaml"
    rp5.write_text(yaml5, encoding="utf-8")
    seen.update(i.code for i in lint_recipe(load_recipe(rp5)).warnings)

    # Trigger 6: zero-weight font.
    yaml6 = f"""\
schema_version: 1
name: zw
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {dest}
  count: 5000
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {f1}
    weight: 1.0
  - path: {f2}
    weight: 0.0
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
"""
    rp6 = tmp_path / "r6.yaml"
    rp6.write_text(yaml6, encoding="utf-8")
    seen.update(i.code for i in lint_recipe(load_recipe(rp6)).warnings)

    # Trigger 7: all fonts optional.
    yaml7 = f"""\
schema_version: 1
name: ao
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {dest}
  count: 5000
corpus:
  - type: local
    path: {corpus}
fonts:
  - path: {f1}
    optional: true
  - path: {f2}
    optional: true
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
"""
    rp7 = tmp_path / "r7.yaml"
    rp7.write_text(yaml7, encoding="utf-8")
    seen.update(i.code for i in lint_recipe(load_recipe(rp7)).warnings)

    # Every catalog entry must show up in the union.
    unreachable = LINT_CODES - seen
    assert not unreachable, (
        f"LINT_CODES has entries not produced by any test fixture: "
        f"{sorted(unreachable)}. Either remove the stale entry from "
        "src/pd_ocr_synth/lint.py:LINT_CODES, or add a fixture that "
        "exercises the corresponding helper."
    )
