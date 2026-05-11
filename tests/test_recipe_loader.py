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
# layout.paragraph_spacing — defaults and three-form parsing
# ---------------------------------------------------------------------------


def test_layout_paragraph_spacing_defaults_to_none(write_recipe) -> None:
    """Omitting paragraph_spacing leaves it ``None`` (recipe-default).

    Mirrors the ``line_spacing`` default contract — both fields opt
    into varying spacing only when the recipe author explicitly
    declares them.
    """
    recipe = load_recipe(write_recipe(MINIMAL_RECIPE))
    assert recipe.layout.paragraph_spacing is None


@pytest.mark.parametrize(
    ("yaml_value", "check"),
    [
        ("1.4", lambda v: v == 1.4),
        ("{ min: 1.2, max: 1.8 }", lambda v: isinstance(v, Range) and v.max == 1.8),
        (
            "[{ value: 1.2, weight: 0.7 }, { value: 1.6, weight: 0.3 }]",
            lambda v: (
                isinstance(v, list)
                and all(isinstance(x, WeightedChoice) for x in v)
                and len(v) == 2
            ),
        ),
    ],
)
def test_layout_paragraph_spacing_supports_three_forms(
    write_recipe, yaml_value: str, check
) -> None:
    """``paragraph_spacing`` accepts the same scalar/range/choice forms as ``line_spacing``."""
    yaml_text = MINIMAL_RECIPE.replace(
        "  mode: word_crops\n  padding_px: 8\n",
        f"  mode: word_crops\n  padding_px: 8\n  paragraph_spacing: {yaml_value}\n",
    )
    recipe = load_recipe(write_recipe(yaml_text))
    assert check(recipe.layout.paragraph_spacing)


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


def test_web_corpus_field_path_loads_from_yaml(write_recipe) -> None:
    """``field_path`` is documented in spec 04 as the JSON sub-tree
    selector for ``parser: json``. Until iter 96 it was missing from
    ``WebCorpus`` and pydantic's ``extra='forbid'`` rejected it at
    load time, leaving the option unreachable from any YAML recipe and
    making the web provider's ``field_path``-aware ``cache_key`` (iter
    95) effectively dead code from the loader's perspective.

    This regression test pins the load path: a recipe that names
    ``field_path`` must (a) parse cleanly, (b) round-trip the value
    onto the typed model, and (c) include it in the ``model_dump``
    that the corpus runner hands to ``WebProvider.fetch``. If anyone
    re-removes the field, all three assertions fail at once.
    """

    yaml_text = MINIMAL_RECIPE.replace(
        "  - type: local\n    path: ./seed.txt\n",
        (
            "  - type: local\n    path: ./seed.txt\n"
            "  - type: web\n"
            "    url: https://example.com/api.json\n"
            "    parser: json\n"
            "    field_path: $.entries[*].body\n"
        ),
    )
    recipe = load_recipe(write_recipe(yaml_text))
    web_entries = [c for c in recipe.corpus if isinstance(c, WebCorpus)]
    assert len(web_entries) == 1
    web = web_entries[0]
    assert web.field_path == "$.entries[*].body"
    # Default is ``None`` for any WebCorpus that omits the key. Other
    # web entries in real recipes (e.g. plain HTML) must keep that
    # default so they don't accidentally invalidate cached payloads.
    assert WebCorpus(type="web", url="https://x").field_path is None
    # The runner reaches the provider via ``model_dump``. The dump
    # must include ``field_path`` so iter-95's ``cache_key`` actually
    # sees the value supplied by the recipe author.
    dumped = web.model_dump(mode="python")
    assert dumped["field_path"] == "$.entries[*].body"


# ---------------------------------------------------------------------------
# Spec 04 ↔ pydantic per-provider parity (this iter)
#
# Iter 96 caught one slice of this drift class (``WebCorpus.field_path``
# missing); the per-provider audit then surfaced four more slices —
# ``LocalCorpus.parser``, ``WebCorpus.{user_agent,retries,
# timeout_seconds,respect_robots}``, ``WikisourceCorpus.{category,
# max_pages}``, ``HFDatasetCorpus.max_rows``, plus the common
# ``cache_key`` knob on ``_CorpusBase``. Each load-path test below
# pins the round-trip the meta-test in ``tests/test_spec_docs.py``
# enforces structurally: a recipe that copy-pastes the spec example
# loads cleanly, the value lands on the typed model, and the
# ``model_dump`` the runner hands to providers carries it.
# ---------------------------------------------------------------------------


def test_local_corpus_parser_loads_from_yaml(write_recipe) -> None:
    """``parser:`` on a ``local`` entry must round-trip onto ``LocalCorpus``.

    The local provider already reads ``options.get("parser")`` to
    short-circuit extension inference (local.py:43). Pre-iter-N the
    model rejected the key, so the documented escape-hatch was
    unreachable from any recipe.
    """

    yaml_text = MINIMAL_RECIPE.replace(
        "  - type: local\n    path: ./seed.txt\n",
        "  - type: local\n    path: ./seed.txt\n    parser: plain\n",
    )
    recipe = load_recipe(write_recipe(yaml_text))
    local_entries = [c for c in recipe.corpus if isinstance(c, LocalCorpus)]
    assert len(local_entries) == 1
    assert local_entries[0].parser == "plain"
    assert local_entries[0].model_dump(mode="python")["parser"] == "plain"


def test_web_corpus_transport_options_load_from_yaml(write_recipe) -> None:
    """``user_agent``/``retries``/``timeout_seconds``/``respect_robots`` round-trip.

    Spec 04 ``web`` block advertises all four as canonical keys on the
    ``web`` provider entry. Pre-iter-N the model rejected each — a
    recipe that copy-pasted the spec example crashed at load with an
    ``extra_forbidden`` ValidationError instead of the documented
    behaviour.
    """

    yaml_text = MINIMAL_RECIPE.replace(
        "  - type: local\n    path: ./seed.txt\n",
        (
            "  - type: local\n    path: ./seed.txt\n"
            "  - type: web\n"
            "    url: https://example.com/page.html\n"
            "    parser: html-text\n"
            '    user_agent: "pd-ocr-synth/0.1 (+contact@example.com)"\n'
            "    retries: 5\n"
            "    timeout_seconds: 45\n"
            "    respect_robots: true\n"
        ),
    )
    recipe = load_recipe(write_recipe(yaml_text))
    web_entries = [c for c in recipe.corpus if isinstance(c, WebCorpus)]
    assert len(web_entries) == 1
    web = web_entries[0]
    assert web.user_agent == "pd-ocr-synth/0.1 (+contact@example.com)"
    assert web.retries == 5
    assert web.timeout_seconds == 45.0
    assert web.respect_robots is True
    dumped = web.model_dump(mode="python")
    # The runner reaches the provider via ``model_dump``; the
    # transport options must be on the dumped dict so the (current
    # and future) HTTP layer can read them.
    assert dumped["retries"] == 5
    assert dumped["user_agent"] == "pd-ocr-synth/0.1 (+contact@example.com)"
    assert dumped["timeout_seconds"] == 45.0
    assert dumped["respect_robots"] is True


def test_wikisource_corpus_category_loads_from_yaml(write_recipe) -> None:
    """``category`` + ``max_pages`` on a ``wikisource`` entry round-trip.

    Spec 04 ``wikisource`` block presents ``titles:`` and ``category:``
    as alternative selectors. Pre-iter-N the model required ``titles``
    even when ``category:`` was the supplied key, contradicting the
    spec; the model also rejected ``max_pages``. The wikisource
    provider already inspects ``options.get("category")`` (and raises
    a deferred-feature ``ProviderError``) — the model must let the
    YAML reach the runtime.
    """

    yaml_text = MINIMAL_RECIPE.replace(
        "  - type: local\n    path: ./seed.txt\n",
        (
            "  - type: local\n    path: ./seed.txt\n"
            "  - type: wikisource\n"
            "    language: ga\n"
            '    category: "Téacsleabhair sa Ghaeilge"\n'
            "    max_pages: 50\n"
        ),
    )
    recipe = load_recipe(write_recipe(yaml_text))
    wiki_entries = [c for c in recipe.corpus if isinstance(c, WikisourceCorpus)]
    assert len(wiki_entries) == 1
    wiki = wiki_entries[0]
    assert wiki.category == "Téacsleabhair sa Ghaeilge"
    assert wiki.max_pages == 50
    # Default ``titles`` to empty list is valid as long as ``category``
    # is set (model_validator); the dumped dict should carry both.
    assert wiki.titles == []
    dumped = wiki.model_dump(mode="python")
    assert dumped["category"] == "Téacsleabhair sa Ghaeilge"
    assert dumped["max_pages"] == 50


def test_wikisource_requires_titles_or_category(write_recipe) -> None:
    """A wikisource entry with neither ``titles`` nor ``category`` fails load.

    Spec 04 says the two are alternatives; an entry that supplies
    neither has nothing to fetch. Pre-iter-N the model required
    ``titles`` so this was caught for the titles-only path; with
    ``titles`` defaulted the explicit cross-check has to enforce it.
    """

    from pydantic import ValidationError

    yaml_text = MINIMAL_RECIPE.replace(
        "  - type: local\n    path: ./seed.txt\n",
        ("  - type: local\n    path: ./seed.txt\n  - type: wikisource\n    language: ga\n"),
    )
    with pytest.raises(ValidationError, match=r"titles.*category|category.*titles"):
        load_recipe(write_recipe(yaml_text))


def test_hf_dataset_max_rows_loads_from_yaml(write_recipe) -> None:
    """``max_rows`` on an ``hf_dataset`` entry round-trips onto ``HFDatasetCorpus``.

    Spec 04 advertises this as the row cap for streaming. Pre-iter-N
    the model rejected the key. The hf_dataset provider itself is
    deferred (M03 roadmap) — accepting the field today keeps the model
    aligned with the spec so a recipe author following the docs hits
    the deferred-provider error from runtime, not a confusing model
    validation error at load.
    """

    yaml_text = MINIMAL_RECIPE.replace(
        "  - type: local\n    path: ./seed.txt\n",
        (
            "  - type: local\n    path: ./seed.txt\n"
            "  - type: hf_dataset\n"
            "    name: example/irish-corpus\n"
            "    split: train\n"
            "    field: text\n"
            "    max_rows: 10000\n"
        ),
    )
    recipe = load_recipe(write_recipe(yaml_text))
    from pd_ocr_synth.recipe.models import HFDatasetCorpus

    hf_entries = [c for c in recipe.corpus if isinstance(c, HFDatasetCorpus)]
    assert len(hf_entries) == 1
    assert hf_entries[0].max_rows == 10000
    assert hf_entries[0].model_dump(mode="python")["max_rows"] == 10000


def test_corpus_cache_key_loads_from_yaml(write_recipe) -> None:
    """``cache_key`` (common-keys override) round-trips onto any corpus entry.

    Spec 04 "Common keys" documents ``cache_key`` as an advanced
    override on every provider. Pre-iter-N the field was missing from
    ``_CorpusBase`` so the override was unreachable from any recipe.
    """

    yaml_text = MINIMAL_RECIPE.replace(
        "  - type: local\n    path: ./seed.txt\n",
        ("  - type: local\n    path: ./seed.txt\n    cache_key: my-explicit-key\n"),
    )
    recipe = load_recipe(write_recipe(yaml_text))
    local_entries = [c for c in recipe.corpus if isinstance(c, LocalCorpus)]
    assert len(local_entries) == 1
    assert local_entries[0].cache_key == "my-explicit-key"


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


# ---------------------------------------------------------------------------
# degradation_presets
# ---------------------------------------------------------------------------


_PRESET_BLOCK = """
degradation_presets:
  light:
    - { kind: blur, probability: 0.5, sigma: { min: 0.0, max: 0.8 } }
    - { kind: jpeg, probability: 0.3, quality: { min: 80, max: 95 } }
  heavy:
    - { kind: noise, probability: 0.5, noise_kind: gaussian, stddev: { min: 2, max: 8 } }

degradation:
  - { kind: skew, probability: 0.4, angle_deg: { min: -1, max: 1 } }
  - preset: light
  - preset: heavy
  - { kind: grayscale, probability: 0.1 }
"""


def test_degradation_presets_expand_inline_in_order(write_recipe) -> None:
    recipe = load_recipe(write_recipe(MINIMAL_RECIPE + _PRESET_BLOCK))

    kinds = [stage.kind for stage in recipe.degradation]
    # skew, then light's [blur, jpeg], then heavy's [noise], then grayscale.
    assert kinds == ["skew", "blur", "jpeg", "noise", "grayscale"]
    # Original presets dict is preserved on the model for round-tripping.
    assert "light" in recipe.degradation_presets
    assert {s.kind for s in recipe.degradation_presets["light"]} == {"blur", "jpeg"}


def test_degradation_preset_unknown_name_raises(write_recipe) -> None:
    yaml_text = MINIMAL_RECIPE + "\ndegradation:\n  - preset: does_not_exist\n"
    with pytest.raises(RecipeLoadError, match="unknown preset"):
        load_recipe(write_recipe(yaml_text))


def test_degradation_preset_paths_resolved_relative_to_recipe(write_recipe, tmp_path) -> None:
    # A preset that references a paper_texture directory must have its
    # ``directory`` resolved relative to the recipe file, not CWD.
    textures = tmp_path / "tex"
    textures.mkdir()
    yaml_text = MINIMAL_RECIPE + (
        "\ndegradation_presets:\n"
        "  paper:\n"
        "    - { kind: paper_texture, probability: 0.5, "
        "directory: ./tex, blend: multiply, opacity: { min: 0.2, max: 0.5 } }\n"
        "\ndegradation:\n"
        "  - preset: paper\n"
    )
    recipe = load_recipe(write_recipe(yaml_text))
    stage = recipe.degradation[0]
    resolved = (stage.model_extra or {}).get("directory")
    assert resolved is not None
    # Resolution against the recipe directory (which is tmp_path here).
    assert Path(resolved).resolve() == textures.resolve()
