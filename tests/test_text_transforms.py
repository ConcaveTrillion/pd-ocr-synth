"""Tests for ``pd_ocr_synth.text_transforms``."""

from __future__ import annotations

import random

import pytest

from pd_ocr_synth.text_transforms import (
    PipelineStep,
    UnknownTransformError,
    apply_pipeline,
    default_registry,
)
from pd_ocr_synth.text_transforms.builtins import (
    apply_lenition_dots,
    dot_to_seimhiu,
    keep_only,
    long_s_medial,
    max_token_length,
    min_token_length,
    normalize_whitespace,
    regex_replace,
    strip_punctuation,
    tironian_et,
)

# ---------------------------------------------------------------------------
# Generic transforms
# ---------------------------------------------------------------------------


def test_normalize_whitespace_collapses_intra_paragraph_runs() -> None:
    out = normalize_whitespace("hello   world\t\tfoo  bar\n", {}, random.Random(0))
    assert out == "hello world foo bar\n"


def test_normalize_whitespace_preserves_paragraph_breaks() -> None:
    out = normalize_whitespace("para 1\n\n\n\npara 2\n", {}, random.Random(0))
    assert out == "para 1\n\npara 2\n"


def test_strip_punctuation_drops_unicode_p_categories() -> None:
    out = strip_punctuation("hello, world! «quoted»", {}, random.Random(0))
    assert "," not in out
    assert "!" not in out
    assert "«" not in out


def test_keep_only_drops_disallowed_chars() -> None:
    out = keep_only("hello123 world!", {"chars": "abcdefghijklmnopqrstuvwxyz "}, random.Random(0))
    assert out == "hello world"


def test_keep_only_requires_chars() -> None:
    with pytest.raises(ValueError, match="chars"):
        keep_only("x", {}, random.Random(0))


def test_regex_replace_substitutes() -> None:
    out = regex_replace(
        "foo BAR foo", {"pattern": r"foo", "replacement": "qux", "flags": ""}, random.Random(0)
    )
    assert out == "qux BAR qux"


def test_regex_replace_flags_ignore_case() -> None:
    out = regex_replace(
        "FOO foo Foo", {"pattern": "foo", "replacement": "x", "flags": "i"}, random.Random(0)
    )
    assert out == "x x x"


def test_min_max_token_length() -> None:
    short = min_token_length("a bb ccc dddd", {"min": 3}, random.Random(0))
    assert short == "ccc dddd"
    long_ = max_token_length("a bb ccc dddd", {"max": 2}, random.Random(0))
    assert long_ == "a bb"


# ---------------------------------------------------------------------------
# Lenition / dotting
# ---------------------------------------------------------------------------


def test_apply_lenition_dots_aggressive_replaces_every_match() -> None:
    out = apply_lenition_dots("bhi bhocht thinn", {"mode": "aggressive"}, random.Random(0))
    # Every digraph hits — the 'ch' inside 'bhocht' lenites too.
    assert out == "ḃi ḃoċt ṫinn"


def test_apply_lenition_dots_conservative_requires_following_vowel() -> None:
    # 'bh' followed by 'a' (vowel) → converted; 'bh' at end of word
    # (no following char) → left alone.
    out = apply_lenition_dots("bha bhcons", {"mode": "conservative"}, random.Random(0))
    # "bha" → ḃa (h followed by a vowel = converted)
    # "bhcons" → "bhcons" (h followed by 'c', a consonant → unchanged in conservative)
    assert "ḃa" in out
    assert "bhcons" in out


def test_apply_lenition_dots_probability_is_deterministic_per_seed() -> None:
    text = "bhi bhocht thinn dheis ghairm mhór phianta shuíomh"
    a = apply_lenition_dots(text, {"probability": 0.5}, random.Random(42))
    b = apply_lenition_dots(text, {"probability": 0.5}, random.Random(42))
    assert a == b
    # Different seed → different result (very likely).
    c = apply_lenition_dots(text, {"probability": 0.5}, random.Random(7))
    assert (a, c) != (a, a)  # distinguishable


def test_apply_lenition_dots_preserves_uppercase() -> None:
    out = apply_lenition_dots("Bhi mé", {"mode": "aggressive"}, random.Random(0))
    assert out.startswith("Ḃ")


def test_dot_to_seimhiu_round_trip() -> None:
    original = "bhi bhocht thinn"
    dotted = apply_lenition_dots(original, {}, random.Random(0))
    back = dot_to_seimhiu(dotted, {}, random.Random(0))
    assert back == original


def test_apply_lenition_dots_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="mode"):
        apply_lenition_dots("bhi", {"mode": "blah"}, random.Random(0))


# ---------------------------------------------------------------------------
# Tironian et
# ---------------------------------------------------------------------------


def test_tironian_et_replaces_default_words() -> None:
    out = tironian_et(
        "agus the rain and snow et tu",
        {"probability": 1.0, "case_sensitive": False},
        random.Random(0),
    )
    # 'agus', 'and', 'et' → ⁊
    assert "⁊" in out
    assert out.count("⁊") == 3


def test_tironian_et_respects_word_boundary() -> None:
    # 'cetera' contains 'et' but not as a whole word.
    out = tironian_et("et cetera", {"probability": 1.0, "case_sensitive": False}, random.Random(0))
    assert "cetera" in out
    assert out.startswith("⁊")


def test_tironian_et_case_sensitive_does_not_match_capitalized() -> None:
    out = tironian_et("Agus and", {"probability": 1.0, "case_sensitive": True}, random.Random(0))
    assert "Agus" in out
    assert out.endswith("⁊")  # 'and' lowercase still matched


def test_tironian_et_probability_is_deterministic() -> None:
    text = "agus and et agus and et agus and et"
    a = tironian_et(text, {"probability": 0.5}, random.Random(123))
    b = tironian_et(text, {"probability": 0.5}, random.Random(123))
    assert a == b


# ---------------------------------------------------------------------------
# Long-s
# ---------------------------------------------------------------------------


def test_long_s_medial_replaces_internal_s() -> None:
    out = long_s_medial("usual", {"probability": 1.0}, random.Random(0))
    assert out == "uſual"


def test_long_s_medial_skips_word_final() -> None:
    out = long_s_medial("loves", {"probability": 1.0}, random.Random(0))
    assert out == "loves"


def test_long_s_medial_skips_before_s_or_h() -> None:
    out_ss = long_s_medial("possess", {"probability": 1.0}, random.Random(0))
    out_sh = long_s_medial("ashes", {"probability": 1.0}, random.Random(0))
    # 'possess': both internal 'ss' pairs left alone; trailing 's' word-final.
    assert "ſſ" not in out_ss
    # 'ashes': 'sh' preserved.
    assert "ſh" not in out_sh


def test_long_s_medial_word_initial_allowed() -> None:
    out = long_s_medial("sun", {"probability": 1.0}, random.Random(0))
    assert out.startswith("ſ")


def test_long_s_medial_uppercase_left_alone() -> None:
    out = long_s_medial("Substance Subdued", {"probability": 1.0}, random.Random(0))
    # 'S' uppercase remains S; lowercase internal 's' becomes ſ where eligible.
    assert "S" in out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def test_pipeline_runs_steps_in_order() -> None:
    out = apply_pipeline(
        "AGUS Bhocht",
        [
            {"name": "lowercase", "options": {}},
            {"name": "tironian_et", "options": {"probability": 1.0}},
            {"name": "apply_lenition_dots", "options": {"mode": "aggressive"}},
        ],
        seed=0,
    )
    assert out == "⁊ ḃoċt"


def test_pipeline_is_deterministic_under_seed() -> None:
    text = "agus bhi siubhal " * 10
    steps = [
        PipelineStep(name="tironian_et", options={"probability": 0.5}),
        PipelineStep(name="apply_lenition_dots", options={"probability": 0.7}),
        PipelineStep(name="long_s_medial", options={"probability": 0.9}),
    ]
    a = apply_pipeline(text, steps, seed=2026)
    b = apply_pipeline(text, steps, seed=2026)
    assert a == b
    c = apply_pipeline(text, steps, seed=2027)
    assert a != c


def test_unknown_transform_raises() -> None:
    with pytest.raises(UnknownTransformError, match="unknown text transform"):
        apply_pipeline("x", [{"name": "no-such-transform", "options": {}}])


def test_default_registry_has_expected_names() -> None:
    names = default_registry().names()
    for expected in (
        "normalize_whitespace",
        "keep_only",
        "apply_lenition_dots",
        "tironian_et",
        "long_s_medial",
    ):
        assert expected in names
