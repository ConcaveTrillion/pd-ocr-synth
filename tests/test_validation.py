"""Tests for ``pd_ocr_synth.validation``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.recipe import load_recipe
from pd_ocr_synth.validation import (
    KNOWN_DEGRADATION_KINDS,
    VALIDATION_CODES,
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


def test_unimplemented_corpus_provider_is_error(tmp_path: Path) -> None:
    """Spec-known but not-yet-registered corpus types must error at validate time.

    ``hf_dataset`` is in ``recipe.models.CorpusEntry`` (so the YAML
    loads cleanly through pydantic) but the M03 runtime registry does
    not register it yet — see ``docs/roadmap/03-corpus.md``
    "Built-in providers". Calling render on a recipe that uses it would
    raise ``ProviderError(f"unknown corpus provider 'hf_dataset' …")``
    deep in :func:`pd_ocr_synth.corpus.runner.run_providers`, well
    after corpus-fetch + setup costs for any preceding entries.

    Mirrors the iter-65 ``degradation_kind_not_implemented`` precedent:
    surface the gap up front, with a distinct error code separate from
    the truly-unknown-type path that pydantic itself rejects.
    """

    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    # Inject a second corpus entry of the unimplemented type *before*
    # ``fonts:``. Keep the local one so the test exercises the
    # registered/unregistered interleaving the validator must handle
    # correctly. ``_minimal_yaml`` keeps a stable layout, so the
    # ``fonts:\n`` anchor is safe.
    extra_entry = (
        "  - type: hf_dataset\n    name: example/irish-corpus\n    split: train\n    field: text\n"
    )
    yaml_text = yaml_text.replace("fonts:\n", extra_entry + "fonts:\n")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "corpus_provider_not_implemented" in codes, [i.format() for i in report.issues]
    msg = next(i.message for i in report.errors if i.code == "corpus_provider_not_implemented")
    # Error message points at the roadmap so the user knows where to
    # look for status / contribute, matching the
    # ``degradation_kind_not_implemented`` shape.
    assert "03-corpus.md" in msg
    assert "hf_dataset" in msg


def test_implemented_corpus_providers_pass_clean(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """A recipe using only registered providers must validate clean.

    Companion to ``test_unimplemented_corpus_provider_is_error``: pin
    the negative-space invariant that the new
    ``corpus_provider_not_implemented`` check does *not* fire for the
    M03 builtins. If a future refactor accidentally drops ``local`` or
    ``web`` from the default registry, this test surfaces it
    immediately rather than as a downstream regression in render.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "corpus_provider_not_implemented" not in codes, [i.format() for i in report.issues]


def test_corpus_max_chars_override_is_error(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """Spec-04 ``max_chars`` override is unread; surface as validate-time error.

    ``_CorpusBase.max_chars`` is documented in spec 04 (Common keys
    table) and accepted by the recipe model, but no provider or
    post-fetch stage reads it — setting ``max_chars: 1024`` on a
    corpus entry is silently ignored. Same "worse than a crash" gap
    the iter-65 / iter-73 / iter-74 / iter-75 / iter-76
    ``*_not_implemented`` precedents address: surface up front so the
    user discovers the gap rather than wondering why their truncation
    knob did nothing.

    The default (``None``, "unlimited") is a no-op and must not fire
    — only an explicit override is an error.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello world\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    # Inject ``max_chars: 1024`` into the local corpus entry. The
    # ``_minimal_yaml`` template anchors on ``    path: <seed>\n``,
    # so we splice the override right after that line.
    yaml_text = yaml_text.replace(
        f"    path: {seed}\n",
        f"    path: {seed}\n    max_chars: 1024\n",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "corpus_max_chars_not_implemented" in codes, [i.format() for i in report.issues]
    msg = next(i.message for i in report.errors if i.code == "corpus_max_chars_not_implemented")
    # Error message points at the roadmap so the user knows where to
    # look for status / contribute, matching the existing
    # ``*_not_implemented`` shape.
    assert "03-corpus.md" in msg
    assert "max_chars" in msg


def test_corpus_max_chars_default_passes_clean(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """The default ``max_chars=None`` (unlimited) must not fire the check.

    Companion to ``test_corpus_max_chars_override_is_error``: pin the
    negative-space invariant that the new check does *not* fire on
    the documented default. If a future refactor accidentally inverts
    the polarity (rejecting ``None`` / unset), every recipe that
    omits the key would suddenly fail to validate.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "corpus_max_chars_not_implemented" not in codes, [i.format() for i in report.issues]


def test_corpus_min_word_length_override_is_error(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """Spec-04 ``min_word_length`` override is unread; surface as error.

    ``_CorpusBase.min_word_length`` is documented in spec 04 (Common
    keys table) as "Drop tokens shorter than this after tokenization"
    and accepted by the recipe model, but no provider or
    post-tokenization stage reads it. The default (``1``) is a
    natural no-op (every non-empty token has ``len >= 1``); only an
    explicit override > 1 is a non-default the user expects to filter.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello world\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    yaml_text = yaml_text.replace(
        f"    path: {seed}\n",
        f"    path: {seed}\n    min_word_length: 3\n",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "corpus_min_word_length_not_implemented" in codes, [i.format() for i in report.issues]
    msg = next(
        i.message for i in report.errors if i.code == "corpus_min_word_length_not_implemented"
    )
    assert "03-corpus.md" in msg
    assert "min_word_length" in msg


def test_corpus_min_word_length_default_passes_clean(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """The default ``min_word_length=1`` (no-op) must not fire the check.

    Companion to ``test_corpus_min_word_length_override_is_error``.
    Also exercises the explicit-equals-default case (``min_word_length:
    1``) — a recipe author writing the default out for documentation
    purposes must not be punished for it.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    yaml_text = yaml_text.replace(
        f"    path: {seed}\n",
        f"    path: {seed}\n    min_word_length: 1\n",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "corpus_min_word_length_not_implemented" not in codes, [
        i.format() for i in report.issues
    ]


def test_unimplemented_text_transform_is_error(tmp_path: Path) -> None:
    """Spec-known but not-yet-registered text transforms must error at validate time.

    ``u_v_swap`` is documented in spec 05 ("Built-in: scriptio
    continua / antique conventions") but the M04 runtime registry does
    not register it yet — see ``docs/roadmap/04-text-transforms.md``
    "Antique-conventions built-ins". Calling render on a recipe that
    uses it would raise
    :class:`pd_ocr_synth.text_transforms.UnknownTransformError` deep in
    :func:`pd_ocr_synth.text_transforms.apply_pipeline`, well after
    corpus-fetch + setup costs.

    Mirrors the iter-65 ``degradation_kind_not_implemented`` / iter-73
    ``corpus_provider_not_implemented`` precedents: surface the gap up
    front, with a distinct error code separate from any structural
    rejection pydantic would do at YAML load (the recipe model accepts
    any string for ``TextTransform.name``).
    """

    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text = yaml_text + "text_transforms:\n  - u_v_swap\n"
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "text_transform_not_implemented" in codes, [i.format() for i in report.issues]
    msg = next(i.message for i in report.errors if i.code == "text_transform_not_implemented")
    # Error message points at the roadmap so the user knows where to
    # look for status / contribute, matching the
    # ``corpus_provider_not_implemented`` shape.
    assert "04-text-transforms.md" in msg
    assert "u_v_swap" in msg


def test_implemented_text_transforms_pass_clean(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """A recipe using only registered transforms must validate clean.

    Companion to ``test_unimplemented_text_transform_is_error``: pin
    the negative-space invariant that the new
    ``text_transform_not_implemented`` check does *not* fire for the
    M04 builtins. If a future refactor accidentally drops a registered
    transform from the default registry, this test surfaces it
    immediately rather than as a downstream regression in render.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    yaml_text = (
        yaml_text
        + "text_transforms:\n"
        + "  - normalize_whitespace\n"
        + "  - apply_lenition_dots:\n"
        + "      mode: aggressive\n"
        + "  - tironian_et:\n"
        + "      probability: 0.7\n"
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "text_transform_not_implemented" not in codes, [i.format() for i in report.issues]


def test_unimplemented_shaping_engine_is_error(tmp_path: Path) -> None:
    """Spec-known but not-yet-implemented shaping engines must error at validate time.

    Spec 06 ("Shaping engine") and the recipe model
    (``Rendering.shaping_engine``) accept ``pillow`` alongside
    ``harfbuzz``, but the M05 renderer in ``pd_ocr_synth.render.*``
    calls ``uharfbuzz`` unconditionally — there is no engine dispatch
    yet. ``docs/roadmap/05-rendering.md`` deliverable "Pillow-only
    fallback engine" is explicitly deferred. Without this validator
    check, a recipe declaring ``shaping_engine: pillow`` would
    silently render via harfbuzz, applying ligatures the user opted
    out of with no signal.

    Mirrors the iter-65 ``degradation_kind_not_implemented`` /
    iter-73 ``corpus_provider_not_implemented`` / iter-74
    ``text_transform_not_implemented`` precedents: surface the gap up
    front, point at the roadmap, with a distinct error code.
    """

    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    # The default rendering block in ``_minimal_yaml`` doesn't set
    # ``shaping_engine`` at all (so it defaults to ``harfbuzz`` per the
    # model). Inject the pillow override before ``layout:``.
    yaml_text = yaml_text.replace(
        "layout:",
        "  shaping_engine: pillow\nlayout:",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "shaping_engine_not_implemented" in codes, [i.format() for i in report.issues]
    msg = next(i.message for i in report.errors if i.code == "shaping_engine_not_implemented")
    # Error message points at the roadmap so the user knows where to
    # look for status / contribute, matching the
    # ``corpus_provider_not_implemented`` / ``text_transform_not_implemented``
    # shapes.
    assert "05-rendering.md" in msg
    assert "pillow" in msg


def test_default_shaping_engine_passes_clean(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """A recipe with the default ``harfbuzz`` engine must validate clean.

    Companion to ``test_unimplemented_shaping_engine_is_error``: pin
    the negative-space invariant that the new
    ``shaping_engine_not_implemented`` check does *not* fire for the
    M05 implemented engine. If a future refactor accidentally drops
    ``harfbuzz`` from ``_IMPLEMENTED_SHAPING_ENGINES``, this test
    surfaces it immediately rather than as a downstream regression in
    every render-using test.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    # Explicit harfbuzz (rather than relying on the default) so a
    # future default-flip is also caught here.
    yaml_text = yaml_text.replace(
        "layout:",
        "  shaping_engine: harfbuzz\nlayout:",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "shaping_engine_not_implemented" not in codes, [i.format() for i in report.issues]


def test_implemented_shaping_engines_subset_of_model_literal() -> None:
    """The runtime-implemented set must stay a subset of the model's accepted set.

    Spec-vs-runtime catalog parity guard, mirroring
    ``test_registered_degradation_kinds_subset_of_spec``: the recipe
    model (``Rendering.shaping_engine``'s ``Literal``) is the spec
    surface; ``_IMPLEMENTED_SHAPING_ENGINES`` is what the renderer can
    actually dispatch. The implemented set must be a (possibly proper)
    subset — implementing a brand-new engine should require widening
    both, not just the runtime set.

    If this assertion fails, either:

    - a new engine landed without being added to the model literal
      (extend ``Rendering.shaping_engine`` first), OR
    - the model literal was narrowed (drop the orphan from
      ``_IMPLEMENTED_SHAPING_ENGINES``).
    """

    from typing import get_args

    from pd_ocr_synth.recipe.models import Rendering
    from pd_ocr_synth.validation import _IMPLEMENTED_SHAPING_ENGINES

    model_literal = frozenset(get_args(Rendering.model_fields["shaping_engine"].annotation))
    assert model_literal >= _IMPLEMENTED_SHAPING_ENGINES, (
        f"implemented engines {sorted(_IMPLEMENTED_SHAPING_ENGINES - model_literal)} "
        "are missing from Rendering.shaping_engine literal"
    )


def test_unimplemented_antialiasing_false_is_error(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """``rendering.antialiasing: false`` must error at validate time.

    Spec 06 "Size, color, DPI" advertises the flag and the recipe
    model defaults it to ``True``, but no renderer in
    ``pd_ocr_synth.render.*`` reads ``recipe.rendering.antialiasing``
    — freetype-py / Pillow produce anti-aliased glyphs
    unconditionally. A recipe asking for aliased output (paleography
    sets that want hard-edge bitmap-style glyphs) would be silently
    ignored, the same "worse than a crash" gap the iter-65 /
    iter-73 / iter-74 / iter-75 ``*_not_implemented`` precedents
    address. Surface it up front, point at the roadmap, with a
    distinct error code.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    yaml_text = yaml_text.replace(
        "layout:",
        "  antialiasing: false\nlayout:",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "antialiasing_disable_not_implemented" in codes, [i.format() for i in report.issues]
    msg = next(i.message for i in report.errors if i.code == "antialiasing_disable_not_implemented")
    # Error message points at the roadmap so the user knows where to
    # look for status / contribute, matching the
    # ``shaping_engine_not_implemented`` shape.
    assert "05-rendering.md" in msg
    assert "antialiasing" in msg


def test_default_antialiasing_passes_clean(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """``antialiasing: true`` (the spec-stated default) must validate clean.

    Companion to ``test_unimplemented_antialiasing_false_is_error``:
    pin the negative-space invariant that the new
    ``antialiasing_disable_not_implemented`` check does *not* fire
    for the default. If a future refactor accidentally inverts the
    polarity (rejecting ``true`` instead of ``false``), this test
    surfaces it immediately rather than as a downstream regression in
    every render-using test.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    # Explicit true (rather than relying on the default) so a future
    # default-flip is also caught here.
    yaml_text = yaml_text.replace(
        "layout:",
        "  antialiasing: true\nlayout:",
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "antialiasing_disable_not_implemented" not in codes, [i.format() for i in report.issues]


def test_default_recipe_antialiasing_unset_passes_clean(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """A recipe that omits ``antialiasing`` entirely must validate clean.

    The model default is ``True``, so ``_minimal_yaml`` (which never
    sets the field) must not trip the new check. Pins the default
    contract: omitting the field is identical to setting it to
    ``true``, both keep the recipe accepted today.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "antialiasing_disable_not_implemented" not in codes, [i.format() for i in report.issues]


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


@pytest.mark.parametrize(
    "kind",
    ["perspective", "scale", "bleed_through", "scratches", "fold_line", "binarize"],
)
def test_unimplemented_degradation_kind_is_error(tmp_path: Path, kind: str) -> None:
    """Spec-known but not-yet-registered kinds must error at validate time.

    These kinds appear in ``docs/specs/07-degradation.md`` but the M06
    runtime registry does not implement them yet (see
    ``docs/roadmap/06-degradation.md`` "Future work"). Calling render
    on a recipe that references one would raise
    ``DegradationError(f"unknown degradation kind {kind!r}")`` deep in
    the pipeline, well after corpus fetch + setup costs. Surface the
    gap at validate time so the user discovers it up front, with a
    distinct error code separate from the truly-unknown kind path.
    """

    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += f"degradation:\n  - kind: {kind}\n    probability: 0.5\n"
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    # Must use the dedicated code, not the generic unknown-kind code.
    assert "degradation_kind_not_implemented" in codes, [i.format() for i in report.issues]
    assert "degradation_kind_unknown" not in codes
    # Error message points at the roadmap so the user knows where to
    # look for status / contribute.
    msg = next(i.message for i in report.errors if i.code == "degradation_kind_not_implemented")
    assert "06-degradation.md" in msg


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


@pytest.mark.parametrize(
    ("kind", "extra_yaml", "bad_key"),
    [
        # Typo: ``signa`` instead of ``sigma``. With ``extra="allow"`` the
        # YAML loads and ``options.get("sigma", 0.0)`` falls back to the
        # default — render proceeds with no blur. Validate must catch it.
        ("blur", "signa: 1.0\n", "signa"),
        # Wrong-stage key: ``quality`` belongs on ``jpeg`` / ``webp``,
        # not on ``blur``. Same silent-ignore failure mode.
        ("blur", "quality: 80\n", "quality"),
        # Mis-spelled flag: ``kernel_size`` is missing the ``_px``
        # suffix. ``options.get("kernel_size_px", 1)`` would default
        # the actual value while the user thought they'd set it.
        ("ink_bleed", "kernel_size: 3\n", "kernel_size"),
        # Recipe author thinks ``opacity`` works on ``jpeg`` (it does not
        # — JPEG quality maps loosely to artifact severity but ``opacity``
        # is undefined). ``_jpeg`` ignores it silently.
        ("jpeg", "opacity: 0.5\n", "opacity"),
        # Wrong key on ``foxing``: spec uses ``radius_px`` not ``radius``.
        ("foxing", "count: 3\n    radius: 4\n", "radius"),
        # Wrong key on ``grayscale``: spec uses ``method`` not ``mode``.
        ("grayscale", "mode: luminosity\n", "mode"),
    ],
)
def test_unknown_per_stage_option_is_error(
    tmp_path: Path,
    writable_font_bytes: bytes,
    kind: str,
    extra_yaml: str,
    bad_key: str,
) -> None:
    """Surface mis-spelled or wrong-stage option keys at validate time.

    ``DegradationStage`` is ``extra="allow"`` so YAML containing
    typos / wrong-stage keys loads cleanly, and the stage handlers
    read options via ``options.get(<canonical-key>, <default>)`` —
    silently falling back to the default rather than raising. That's
    the iter-65 / iter-73 / iter-74 / iter-75 / iter-76 / iter-77
    "worse than a crash" gap: the recipe author thinks they tuned the
    stage but the rendered samples ignore the override entirely.
    Validate-time catches the mismatch before render starts.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt", "hello\n")),
    )
    yaml_text += f"degradation:\n  - kind: {kind}\n    probability: 0.5\n    {extra_yaml}"
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "degradation_stage_unknown_option" in codes, [i.format() for i in report.issues]
    # Location pinpoints the offending key, not the whole stage.
    issue = next(
        i
        for i in report.errors
        if i.code == "degradation_stage_unknown_option"
        and i.location == f"degradation[0].{bad_key}"
    )
    assert kind in issue.message
    assert bad_key in issue.message
    # Spec pointer lets the user look up the canonical key list.
    assert "07-degradation.md" in issue.message


def test_known_per_stage_options_pass_clean(
    tmp_path: Path,
    writable_font_bytes: bytes,
) -> None:
    """Spec-listed keys (and the common ``name`` key) must not error.

    Exercises every stage in
    ``_DEGRADATION_OPTION_KEYS_BY_KIND`` with one valid option each,
    plus a ``name:`` to confirm the common-keys whitelist applies.
    Acts as a guard against drift: if a future edit narrows the
    allowed-options table without updating the underlying handler, this
    test breaks loudly.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    textures_dir = tmp_path / "textures"
    textures_dir.mkdir()
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt", "hello\n")),
    )
    yaml_text += (
        "degradation:\n"
        "  - kind: skew\n    probability: 1.0\n    name: my-skew\n"
        "    angle_deg: 1.0\n    fill: white\n"
        "  - kind: blur\n    probability: 1.0\n"
        "    filter: gaussian\n    sigma: 0.5\n"
        "    motion_angle_deg: 0\n    motion_length_px: 0\n"
        "  - kind: noise\n    probability: 1.0\n"
        "    noise_kind: gaussian\n    stddev: 1.0\n    amount: 0.01\n"
        "  - kind: brightness\n    probability: 1.0\n    factor: 1.0\n"
        "  - kind: contrast\n    probability: 1.0\n    factor: 1.0\n"
        "  - kind: gamma\n    probability: 1.0\n    gamma: 1.0\n"
        "  - kind: ink_bleed\n    probability: 1.0\n"
        "    iterations: 1\n    kernel_size_px: 1\n"
        "  - kind: ink_thin\n    probability: 1.0\n"
        "    iterations: 1\n    kernel_size_px: 1\n"
        f"  - kind: paper_texture\n    probability: 1.0\n    directory: {textures_dir}\n"
        "    blend: multiply\n    opacity: 0.3\n"
        "    scale: 1.0\n    rotate_deg: 0\n"
        "  - kind: foxing\n    probability: 1.0\n"
        "    count: 1\n    radius_px: 3\n    color: [120, 60, 30]\n    opacity: 0.3\n"
        "  - kind: jpeg\n    probability: 1.0\n"
        "    quality: 85\n    chroma_subsampling: '4:4:4'\n"
        "  - kind: webp\n    probability: 1.0\n    quality: 85\n"
        "  - kind: grayscale\n    probability: 1.0\n    method: luminosity\n"
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "degradation_stage_unknown_option" not in codes, [i.format() for i in report.errors]


def test_unknown_option_table_covers_every_implemented_kind() -> None:
    """Meta-test: ``_DEGRADATION_OPTION_KEYS_BY_KIND`` must cover every
    runtime-registered kind whose options the user can tune.

    If a new stage lands in ``pd_ocr_synth.degradation.builtins`` but
    nobody adds it to the validate-time table, that stage's option keys
    silently fall back to the wide-open ``extra="allow"`` behavior —
    every typo gets ignored. Pin the invariant here so future stage
    additions break this test instead of going un-validated.

    ``preset`` is a structural marker (loader expands it before
    validation runs); excluded from the requirement.
    """

    from pd_ocr_synth.degradation.pipeline import REGISTRY, _ensure_builtins_registered
    from pd_ocr_synth.validation import _DEGRADATION_OPTION_KEYS_BY_KIND

    _ensure_builtins_registered()
    registered = frozenset(REGISTRY) - {"preset"}
    missing = registered - frozenset(_DEGRADATION_OPTION_KEYS_BY_KIND)
    assert missing == frozenset(), (
        f"degradation kinds registered with the runtime but missing from "
        f"_DEGRADATION_OPTION_KEYS_BY_KIND (validate-time would silently allow "
        f"any option key for these): {sorted(missing)}"
    )


def test_unknown_option_table_does_not_reference_nonexistent_kinds() -> None:
    """Meta-test: every entry in ``_DEGRADATION_OPTION_KEYS_BY_KIND``
    must correspond to a kind in ``KNOWN_DEGRADATION_KINDS``.

    Catches typos on the validation side: an entry like ``ink_thinn``
    on the table would silently never match any stage. Pinning here
    means the typo breaks this test rather than being undiscoverable.
    """

    from pd_ocr_synth.validation import _DEGRADATION_OPTION_KEYS_BY_KIND

    table_kinds = frozenset(_DEGRADATION_OPTION_KEYS_BY_KIND)
    unknown = table_kinds - KNOWN_DEGRADATION_KINDS
    assert unknown == frozenset(), (
        f"_DEGRADATION_OPTION_KEYS_BY_KIND entries that are not in "
        f"KNOWN_DEGRADATION_KINDS (likely typos): {sorted(unknown)}"
    )


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
# Spec ↔ code drift guard for degradation kinds.
#
# ``KNOWN_DEGRADATION_KINDS`` is the catalog the validator dispatches
# against — anything not in this set surfaces ``degradation_kind_unknown``
# (a *typo* error). Anything in this set but not yet registered with the
# M06 runtime surfaces ``degradation_kind_not_implemented`` (a clean
# "future work" gate, see iter 65 fix).
#
# That dispatch only works while the catalog is **kept in sync with**
# ``docs/specs/07-degradation.md``. If a spec PR adds a new kind but
# forgets ``KNOWN_DEGRADATION_KINDS``, recipes using the spec'd kind
# fall through to ``degradation_kind_unknown`` and the user gets a
# misleading "did you typo this?" error instead of a clear "not yet
# implemented" message. Conversely, if a kind is removed from the spec
# but the catalog still lists it, ``validate`` happily accepts a kind
# the docs no longer describe.
#
# This pair of meta-tests parses the spec doc and asserts the two sides
# match. They have to be kept in lockstep — by design — so any drift
# caught here is a real bug in either the doc or the catalog.
# ---------------------------------------------------------------------------


def _spec_known_degradation_kinds() -> frozenset[str]:
    """Extract the canonical degradation-kind set from the spec doc.

    Parses h3 headers in ``docs/specs/07-degradation.md`` of the form
    ``### \\`name\\``` or ``### \\`a\\` / \\`b\\``` (the ``brightness``
    / ``contrast`` pair). Anything outside the kind catalog is gated
    by section header so unrelated h3s in future revisions of the doc
    don't accidentally enter the set.
    """

    import re

    spec_path = Path(__file__).resolve().parent.parent / "docs" / "specs" / "07-degradation.md"
    text = spec_path.read_text(encoding="utf-8")
    # Catalog sections, in spec order. "Composition presets" and
    # "Custom degradation stages" are not kind catalogs and are
    # intentionally excluded.
    catalog_sections = (
        "## Geometric",
        "## Optical",
        "## Print / paper",
        "## Compression",
        "## Color space",
    )
    kinds: set[str] = set()
    in_catalog = False
    h3_re = re.compile(r"^###\s+(.+)$")
    backtick_re = re.compile(r"`([a-z_][a-z0-9_]*)`")
    for line in text.splitlines():
        if line.startswith("## "):
            in_catalog = line.strip() in catalog_sections
            continue
        if not in_catalog:
            continue
        m = h3_re.match(line)
        if not m:
            continue
        for name in backtick_re.findall(m.group(1)):
            kinds.add(name)
    return frozenset(kinds)


def test_known_degradation_kinds_matches_spec_doc() -> None:
    """``KNOWN_DEGRADATION_KINDS`` must mirror the spec catalog 1:1.

    ``preset`` is the structural marker the loader expands away before
    the validator runs; it is intentionally accepted by the catalog
    even though the spec lists it under "Composition presets" rather
    than as a kind h3. Strip it out for the comparison.
    """

    spec_kinds = _spec_known_degradation_kinds()
    catalog_kinds = KNOWN_DEGRADATION_KINDS - {"preset"}

    missing_from_catalog = spec_kinds - catalog_kinds
    extra_in_catalog = catalog_kinds - spec_kinds
    assert not missing_from_catalog, (
        f"docs/specs/07-degradation.md lists kinds not in "
        f"KNOWN_DEGRADATION_KINDS: {sorted(missing_from_catalog)}. "
        "Update src/pd_ocr_synth/validation.py:KNOWN_DEGRADATION_KINDS."
    )
    assert not extra_in_catalog, (
        f"KNOWN_DEGRADATION_KINDS lists kinds not in "
        f"docs/specs/07-degradation.md: {sorted(extra_in_catalog)}. "
        "Either add the kind to the spec doc or drop it from the catalog."
    )


def test_registered_degradation_kinds_subset_of_spec() -> None:
    """The M06 runtime registry must only register spec-listed kinds.

    A registered kind absent from the spec catalog would render fine
    but never be reachable from a validated recipe (the validator would
    reject it as ``degradation_kind_unknown``). That's a silent gap
    between what the runtime can do and what users can ask for —
    catch it here.

    The reverse direction (catalog kinds *not* registered) is
    intentional and covered by ``degradation_kind_not_implemented`` at
    validate time, so it's not asserted here.
    """

    from pd_ocr_synth.degradation.pipeline import (
        REGISTRY,
        _ensure_builtins_registered,
    )

    _ensure_builtins_registered()
    registered = frozenset(REGISTRY)
    spec_kinds = _spec_known_degradation_kinds()
    extra = registered - spec_kinds
    assert not extra, (
        f"degradation registry registers kinds not listed in "
        f"docs/specs/07-degradation.md: {sorted(extra)}. "
        "Either add the kind to the spec doc or remove the "
        "register_*_stage call."
    )


# ---------------------------------------------------------------------------
# Per-stage option drift: spec 07 YAML examples ↔ validation whitelist.
# ---------------------------------------------------------------------------
#
# The ``KNOWN_DEGRADATION_KINDS`` parity tests above lock the *kind*
# names. They do not catch drift at the *option* level — e.g. a spec
# block that documents ``kernel_size_px`` but a whitelist that omits
# it (validate would reject a spec-legal recipe), or a spec block that
# documents ``foo_thresh`` but the runtime reads ``foo_threshold``
# (validate accepts it but the option is silently ignored).
#
# This meta-test parses each ``### `kind` `` block in
# ``docs/specs/07-degradation.md``, extracts the YAML example
# immediately following it, and asserts the option keys present in
# the example are a subset of
# ``_DEGRADATION_OPTION_KEYS_BY_KIND[kind] | _COMMON_DEGRADATION_OPTION_KEYS
# | {"kind", "probability"}``. Same drift-guard pattern as iters
# 66/67/69/70.
# ---------------------------------------------------------------------------


def _spec_per_stage_option_keys() -> dict[str, frozenset[str]]:
    """Parse spec 07 per-kind YAML examples → ``{kind: {option_keys}}``.

    Walks the doc top-to-bottom inside the catalog sections, tracks the
    most recent ``### `kind` `` heading, captures the first fenced
    ``yaml`` block that follows it, and parses the first list element
    as the canonical example. Headings that pair two kinds with a slash
    (``brightness`` / ``contrast``) are handled by mapping the same
    YAML block to every kind named in the heading.
    """

    import re

    import yaml

    spec_path = Path(__file__).resolve().parent.parent / "docs" / "specs" / "07-degradation.md"
    text = spec_path.read_text(encoding="utf-8")

    catalog_sections = (
        "## Geometric",
        "## Optical",
        "## Print / paper",
        "## Compression",
        "## Color space",
    )
    h3_re = re.compile(r"^###\s+(.+)$")
    backtick_re = re.compile(r"`([a-z_][a-z0-9_]*)`")

    results: dict[str, frozenset[str]] = {}
    in_catalog = False
    pending_kinds: list[str] = []
    in_yaml = False
    yaml_lines: list[str] = []

    def _flush(kinds: list[str], lines: list[str]) -> None:
        if not kinds or not lines:
            return
        try:
            parsed = yaml.safe_load("\n".join(lines))
        except yaml.YAMLError as exc:  # pragma: no cover - test infra
            raise AssertionError(
                f"spec 07 YAML example for {kinds!r} did not parse: {exc}"
            ) from exc
        # Each example is a single-element list of mappings.
        assert isinstance(parsed, list) and parsed, (
            f"spec 07 YAML example for {kinds!r} should be a non-empty list"
        )
        item = parsed[0]
        assert isinstance(item, dict), (
            f"spec 07 YAML example for {kinds!r} first element is not a mapping"
        )
        keys = frozenset(item.keys())
        for kind in kinds:
            results[kind] = keys

    for line in text.splitlines():
        if line.startswith("## "):
            # Section transition flushes any open block.
            _flush(pending_kinds, yaml_lines)
            pending_kinds = []
            yaml_lines = []
            in_yaml = False
            in_catalog = line.strip() in catalog_sections
            continue
        if not in_catalog:
            continue
        h3 = h3_re.match(line)
        if h3 is not None:
            _flush(pending_kinds, yaml_lines)
            pending_kinds = list(backtick_re.findall(h3.group(1)))
            yaml_lines = []
            in_yaml = False
            continue
        if line.strip() == "```yaml":
            # Only capture the *first* yaml block per heading. If we
            # already captured one, ignore subsequent blocks (e.g. the
            # narrative example at the top of the file lives outside
            # any kind heading and is filtered by ``pending_kinds``).
            if pending_kinds and not results.get(pending_kinds[0]):
                in_yaml = True
                yaml_lines = []
            continue
        if in_yaml:
            if line.strip() == "```":
                _flush(pending_kinds, yaml_lines)
                yaml_lines = []
                in_yaml = False
                continue
            yaml_lines.append(line)

    # Trailing flush at EOF (rare — section headers usually flush).
    _flush(pending_kinds, yaml_lines)
    return results


def test_spec_07_yaml_examples_match_option_whitelist() -> None:
    """Every option key in spec 07's per-kind YAML examples must be on
    the validate-time whitelist (or a common key).

    Drift this catches:
      - spec adds an option, whitelist forgets it → validate would
        reject a spec-legal recipe (false positive).
      - spec renames an option, code still reads the old name → the
        whitelist catches the new name but the option is silently
        ignored at runtime. This test forces the spec example to stay
        in lockstep with the whitelist.
    """

    from pd_ocr_synth.validation import (
        _COMMON_DEGRADATION_OPTION_KEYS,
        _DEGRADATION_OPTION_KEYS_BY_KIND,
    )

    # ``kind`` and ``probability`` live on the model proper; they appear
    # in YAML examples but never in ``model_extra``.
    structural = frozenset({"kind", "probability"})

    spec = _spec_per_stage_option_keys()
    assert spec, "spec 07 YAML example parser found zero kinds — parser regressed"

    drift: list[str] = []
    for kind, example_keys in sorted(spec.items()):
        whitelist = _DEGRADATION_OPTION_KEYS_BY_KIND.get(kind)
        if whitelist is None:
            # Stage spec'd but not yet implemented (caught elsewhere by
            # ``degradation_kind_not_implemented``); skip the option
            # check until the stage lands.
            continue
        permitted = whitelist | _COMMON_DEGRADATION_OPTION_KEYS | structural
        unknown = example_keys - permitted
        if unknown:
            drift.append(
                f"  {kind}: example uses {sorted(unknown)} not on whitelist {sorted(whitelist)}"
            )

    assert not drift, (
        "docs/specs/07-degradation.md YAML examples reference option keys not "
        "on _DEGRADATION_OPTION_KEYS_BY_KIND. Either add the option to the "
        "whitelist (and make sure the runtime reads it) or correct the spec "
        "example.\n" + "\n".join(drift)
    )


def test_option_whitelist_keys_appear_in_spec_examples() -> None:
    """Every option on the validate-time whitelist must be referenced
    in spec 07's YAML example for that kind.

    Catches the reverse drift: the whitelist (and presumably the code)
    accepts an option key that the spec doesn't tell users about. The
    spec example is the documentation users learn from; if an option
    is real, it should be discoverable there.
    """

    from pd_ocr_synth.validation import _DEGRADATION_OPTION_KEYS_BY_KIND

    spec = _spec_per_stage_option_keys()
    drift: list[str] = []
    for kind, whitelist in sorted(_DEGRADATION_OPTION_KEYS_BY_KIND.items()):
        example_keys = spec.get(kind)
        if example_keys is None:
            drift.append(f"  {kind}: whitelist exists but no spec YAML example found")
            continue
        missing = whitelist - example_keys
        if missing:
            drift.append(
                f"  {kind}: whitelist allows {sorted(missing)} but spec example never shows them"
            )

    assert not drift, (
        "_DEGRADATION_OPTION_KEYS_BY_KIND lists options not shown in "
        "docs/specs/07-degradation.md examples. Either document the option "
        "in the spec example or drop it from the whitelist.\n" + "\n".join(drift)
    )


# ---------------------------------------------------------------------------
# paragraph_alignment (M09 paragraph alignment)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("layout_mode", ["paragraphs", "pages"])
@pytest.mark.parametrize("alignment", ["left", "center", "right", "justify"])
def test_paragraph_alignment_accepted_on_paragraph_modes(
    tmp_path: Path,
    writable_font_bytes: bytes,
    layout_mode: str,
    alignment: str,
) -> None:
    """``paragraph_alignment`` is permitted on ``paragraphs`` + ``pages`` without warning.

    All four implemented values (``left`` / ``center`` / ``right`` /
    ``justify``) must round-trip through the validator cleanly on
    both paragraph-style modes.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        (
            f"layout:\n"
            f"  mode: {layout_mode}\n"
            f"  padding_px: 8\n"
            f"  max_width_px: 800\n"
            f"  paragraph_alignment: {alignment}\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    align_warnings = [i for i in report.warnings if i.location == "layout.paragraph_alignment"]
    assert align_warnings == [], [i.format() for i in align_warnings]
    assert report.is_ok, [i.format() for i in report.issues]


@pytest.mark.parametrize("layout_mode", ["word_crops", "lines"])
def test_paragraph_alignment_warns_on_recognition_modes(
    tmp_path: Path,
    writable_font_bytes: bytes,
    layout_mode: str,
) -> None:
    """``paragraph_alignment`` is not meaningful for recognition-mode layouts.

    Setting it on ``word_crops`` / ``lines`` emits ``layout_key_unused``
    so the user knows the value will be ignored.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    if layout_mode == "word_crops":
        new_layout = "layout:\n  mode: word_crops\n  padding_px: 8\n  paragraph_alignment: center\n"
    else:
        new_layout = (
            f"layout:\n"
            f"  mode: {layout_mode}\n"
            f"  padding_px: 8\n"
            f"  max_width_px: 800\n"
            f"  paragraph_alignment: center\n"
        )
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        new_layout,
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    align_warning_codes = [
        i.code for i in report.warnings if i.location == "layout.paragraph_alignment"
    ]
    assert "layout_key_unused" in align_warning_codes, [i.format() for i in report.issues]


def test_paragraph_alignment_unknown_value_rejected_at_load(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """Pydantic's Literal rejects unknown alignment values at load time.

    ``"justify"`` joined the supported set in iter 41 — pick a string
    outside the four-value vocabulary to verify the gate is still
    enforced.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        (
            "layout:\n"
            "  mode: paragraphs\n"
            "  padding_px: 8\n"
            "  max_width_px: 800\n"
            "  paragraph_alignment: kerning\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    with pytest.raises(Exception, match="paragraph_alignment"):
        load_recipe(_write(tmp_path, yaml_text))


# ---------------------------------------------------------------------------
# page_size_px (M09 explicit fixed-canvas)
# ---------------------------------------------------------------------------


def test_page_size_px_accepted_on_pages_mode(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """``page_size_px`` is permitted on ``pages`` mode without warning."""
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
            "  page_size_px: [1200, 1800]\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    page_size_warnings = [i for i in report.warnings if i.location == "layout.page_size_px"]
    assert page_size_warnings == [], [i.format() for i in page_size_warnings]
    assert report.is_ok, [i.format() for i in report.issues]


@pytest.mark.parametrize("layout_mode", ["word_crops", "lines", "paragraphs"])
def test_page_size_px_warns_on_non_pages_modes(
    tmp_path: Path,
    writable_font_bytes: bytes,
    layout_mode: str,
) -> None:
    """``page_size_px`` is only meaningful for ``pages`` mode.

    A ``paragraphs`` sample is a tight single-paragraph crop with no
    notion of a "page"; ``word_crops`` / ``lines`` are likewise tight
    crops. Setting it on those modes emits a ``layout_key_unused``
    warning so the user knows the value will be ignored.
    """
    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(font=str(font), dest=str(tmp_path / "out"), corpus=str(seed))
    if layout_mode == "word_crops":
        new_layout = "layout:\n  mode: word_crops\n  padding_px: 8\n  page_size_px: [1200, 1800]\n"
    else:
        new_layout = (
            f"layout:\n"
            f"  mode: {layout_mode}\n"
            f"  padding_px: 8\n"
            f"  max_width_px: 800\n"
            f"  page_size_px: [1200, 1800]\n"
        )
    yaml_text = yaml_text.replace(
        "layout:\n  mode: word_crops\n  padding_px: 8\n",
        new_layout,
    )
    if layout_mode in {"paragraphs", "pages"}:
        yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    warning_codes_at_size = [i.code for i in report.warnings if i.location == "layout.page_size_px"]
    assert "layout_key_unused" in warning_codes_at_size, [i.format() for i in report.issues]


@pytest.mark.parametrize("bad", [(0, 100), (100, 0), (-5, 100), (100, -5)])
def test_page_size_px_rejects_non_positive_at_load(
    tmp_path: Path, writable_font_bytes: bytes, bad: tuple[int, int]
) -> None:
    """Pydantic's ``page_size_px`` validator rejects zero / negative dimensions."""
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
            f"  page_size_px: [{bad[0]}, {bad[1]}]\n"
        ),
    )
    yaml_text = yaml_text.replace("mode: recognition", "mode: detection")
    with pytest.raises(Exception, match="page_size_px"):
        load_recipe(_write(tmp_path, yaml_text))


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
# Layout-model ↔ permitted-keys drift guard.
#
# ``_LAYOUT_KEYS_BY_MODE`` lists, per layout mode, which keys are
# *meaningful* in that mode. Anything set but not listed surfaces a
# ``layout_key_unused`` warning. That dispatch only works while the
# table is **kept in sync with** the ``Layout`` model in
# ``recipe/models.py``:
#
# * If a new Layout field is added but never enters
#   ``_LAYOUT_KEYS_BY_MODE``, every recipe that sets it gets a
#   misleading "unused" warning regardless of mode (false positive),
#   *or* the field is silently dead (no mode reads it).
# * If ``_LAYOUT_KEYS_BY_MODE`` lists a key that isn't actually a
#   Layout field, the validator advertises a config knob the model
#   doesn't accept — ``extra="forbid"`` rejects it at load time, so the
#   permitted-keys entry is a false promise.
#
# These two meta-tests catch both directions. They have to be kept in
# lockstep — by design — so any drift caught here is a real bug.
# ---------------------------------------------------------------------------


def _layout_field_names() -> frozenset[str]:
    """Pydantic model fields on ``Layout``, including ``mode``."""

    from pd_ocr_synth.recipe.models import Layout

    return frozenset(Layout.model_fields.keys())


def _permitted_layout_keys() -> frozenset[str]:
    """Union of keys permitted by any layout mode."""

    from pd_ocr_synth.validation import _LAYOUT_KEYS_BY_MODE

    out: set[str] = set()
    for keys in _LAYOUT_KEYS_BY_MODE.values():
        out.update(keys)
    return frozenset(out)


def test_permitted_layout_keys_are_all_layout_fields() -> None:
    """Every key in ``_LAYOUT_KEYS_BY_MODE`` must be an actual ``Layout`` field.

    A permitted key that isn't a model field is a false promise — pydantic
    rejects the field at load time (``extra='forbid'``), so the validator
    table advertises a knob users can never set.
    """

    permitted = _permitted_layout_keys()
    fields = _layout_field_names()
    extra = permitted - fields
    assert not extra, (
        f"_LAYOUT_KEYS_BY_MODE lists keys not on the Layout model: "
        f"{sorted(extra)}. Either add the field to "
        "src/pd_ocr_synth/recipe/models.py:Layout or drop the key from "
        "_LAYOUT_KEYS_BY_MODE in src/pd_ocr_synth/validation.py."
    )


def test_every_layout_field_appears_in_some_mode() -> None:
    """Every ``Layout`` field (except ``mode``) must be permitted in at least one mode.

    A Layout field absent from every permitted set means either:
    * the validator emits ``layout_key_unused`` whenever the user sets
      it regardless of mode (false positive), or
    * the renderer silently ignores it (dead field).

    ``mode`` is excluded — it's the discriminator the table is keyed
    *by*, not a per-mode configuration key.
    """

    fields = _layout_field_names() - {"mode"}
    permitted = _permitted_layout_keys()
    missing = fields - permitted
    assert not missing, (
        f"Layout fields not permitted in any mode by _LAYOUT_KEYS_BY_MODE: "
        f"{sorted(missing)}. Add each field to the appropriate mode(s) in "
        "src/pd_ocr_synth/validation.py:_LAYOUT_KEYS_BY_MODE, or drop it "
        "from src/pd_ocr_synth/recipe/models.py:Layout if it's truly unused."
    )


def test_permitted_layout_keys_table_modes_match_layout_mode_literal() -> None:
    """``_LAYOUT_KEYS_BY_MODE`` must cover every value of ``Layout.mode``.

    A mode missing from the table falls through to ``frozenset()`` in
    ``_check_layout`` — every key the user sets would then warn as
    unused, regardless of legitimacy. A mode in the table that isn't a
    legal ``Layout.mode`` literal is dead config (pydantic would reject
    it at load time, so the validator never sees it).
    """

    import typing

    from pd_ocr_synth.recipe.models import Layout
    from pd_ocr_synth.validation import _LAYOUT_KEYS_BY_MODE

    mode_field = Layout.model_fields["mode"]
    literal_modes = frozenset(typing.get_args(mode_field.annotation))
    table_modes = frozenset(_LAYOUT_KEYS_BY_MODE.keys())
    missing = literal_modes - table_modes
    extra = table_modes - literal_modes
    assert not missing, (
        f"_LAYOUT_KEYS_BY_MODE is missing layout modes: {sorted(missing)}. "
        "Every value of Layout.mode must have a permitted-keys entry."
    )
    assert not extra, (
        f"_LAYOUT_KEYS_BY_MODE lists modes not in Layout.mode: "
        f"{sorted(extra)}. Drop them from "
        "src/pd_ocr_synth/validation.py:_LAYOUT_KEYS_BY_MODE."
    )


# ---------------------------------------------------------------------------
# Per-code emission fixtures for the harder-to-trigger codes
#
# The straightforward codes (``font_missing``, ``local_corpus_missing``,
# the ``*_not_implemented`` family, etc.) already have fixtures earlier
# in this file. The block below backfills the six codes flagged in
# iter-85 as having zero direct emission coverage:
#
#   * ``font_unreadable``                  — reachable, real bytes
#   * ``font_empty``                       — reachable iff a font exists
#                                            with ``num_glyphs == 0`` or
#                                            an empty cmap (rare in
#                                            practice; covered via a
#                                            monkeypatched ``open_font``
#                                            so the emission path is
#                                            still exercised)
#   * ``paper_texture_directory_not_dir``  — reachable, point at a file
#   * ``publish_description_file_missing`` — reachable, missing path
#   * ``publish_repo_placeholder``         — reachable, ``CHANGE-ME/...``
#   * ``schema_version_unsupported``       — defensive; pydantic blocks
#                                            it on load. Tested via a
#                                            direct ``_check_schema_version``
#                                            call after ``model_copy``
#                                            bypasses the field validator,
#                                            so the emission still gets
#                                            exercised in case a future
#                                            refactor relaxes the load-
#                                            time check.
# ---------------------------------------------------------------------------


def test_font_unreadable_is_error(tmp_path: Path) -> None:
    """A path that exists but isn't a parseable font must error.

    ``_check_fonts`` calls ``open_font`` after the path-exists check;
    when ``freetype`` raises ``FT_Exception`` (unknown file format),
    ``FontOpenError`` propagates and the validator emits
    ``font_unreadable`` at error severity. We construct the situation
    by writing arbitrary non-font bytes to a ``.otf`` extension.
    """

    bad_font = tmp_path / "bogus.otf"
    bad_font.write_bytes(b"not a real font payload")
    yaml_text = _minimal_yaml(
        font=str(bad_font),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "font_unreadable" in codes, [i.format() for i in report.issues]
    msg = next(i.message for i in report.errors if i.code == "font_unreadable")
    assert "could not open font" in msg


def test_font_empty_is_error(
    tmp_path: Path, writable_font_bytes: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A font that opens but reports zero glyphs / empty cmap must error.

    Constructing a real OTF with ``num_glyphs == 0`` is impractical
    (every well-formed sfnt has at least ``.notdef``), but the
    emission path is still meaningful — a corrupted-but-parseable
    cmap, a CFF subset whittled below the runtime threshold, or a
    bitmap-only font with no Unicode mapping all surface here. We
    monkeypatch ``open_font`` to return a zero-glyph ``FontInfo`` so
    the validator's branch executes end-to-end. The font path itself
    is real bytes so the upstream path-exists check passes through to
    the inspection step.
    """

    from pd_ocr_synth import fonts as _fonts

    real_font = tmp_path / "real.otf"
    real_font.write_bytes(writable_font_bytes)

    def _fake_open_font(path):
        return _fonts.FontInfo(
            path=Path(path),
            family="Empty",
            style="Regular",
            num_glyphs=0,
            codepoints=frozenset(),
        )

    # ``_check_fonts`` does a local ``from pd_ocr_synth.fonts import
    # ... open_font`` inside the loop, so patching the module attr
    # is what the import sees.
    monkeypatch.setattr(_fonts, "open_font", _fake_open_font)

    yaml_text = _minimal_yaml(
        font=str(real_font),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "font_empty" in codes, [i.format() for i in report.issues]
    msg = next(i.message for i in report.errors if i.code == "font_empty")
    assert "zero glyphs or empty cmap" in msg


def test_paper_texture_directory_not_dir_is_error(tmp_path: Path) -> None:
    """``paper_texture.directory`` pointing at a file (not a dir) must error.

    Companion to ``test_paper_texture_directory_missing_is_error``.
    The path exists but is a regular file, so the validator falls
    through the ``not exists`` branch into the ``not is_dir`` branch.
    """

    not_a_dir = tmp_path / "textures.txt"
    not_a_dir.write_text("hello\n", encoding="utf-8")
    yaml_text = _minimal_yaml(
        font=str(_make_file(tmp_path / "fake.otf")),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += (
        f"degradation:\n  - kind: paper_texture\n    probability: 0.5\n    directory: {not_a_dir}\n"
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.errors]
    assert "paper_texture_directory_not_dir" in codes, [i.format() for i in report.issues]
    msg = next(i.message for i in report.errors if i.code == "paper_texture_directory_not_dir")
    assert "not a directory" in msg


def test_publish_description_file_missing_is_warning(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """``publish.hf_dataset.description_file`` pointing nowhere is a warning.

    The publish flow tolerates a missing description file (it skips
    the upload step) but tells the user via ``warning`` so they can
    fix the path before the dataset card lands without one.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    ghost = tmp_path / "no-such-card.md"
    yaml_text += (
        "publish:\n"
        "  hf_dataset:\n"
        "    repo: example/pd-ocr-synth-test\n"
        f"    description_file: {ghost}\n"
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    # Description-missing is a warning, not an error.
    assert report.errors == (), [i.format() for i in report.errors]
    codes = [i.code for i in report.warnings]
    assert "publish_description_file_missing" in codes, [i.format() for i in report.issues]


def test_publish_repo_placeholder_is_warning(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """``publish.hf_dataset.repo`` left as ``CHANGE-ME/...`` must warn.

    The bundled ``recipes/gaelic.yaml`` ships with
    ``repo: CHANGE-ME/pd-ocr-synth-gaelic`` so users can't push to a
    real namespace by accident. Validate surfaces the placeholder as a
    warning so they fix it before publish (which would 401 anyway).
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += "publish:\n  hf_dataset:\n    repo: CHANGE-ME/pd-ocr-synth-test\n"
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    assert report.errors == (), [i.format() for i in report.errors]
    codes = [i.code for i in report.warnings]
    assert "publish_repo_placeholder" in codes, [i.format() for i in report.issues]
    msg = next(i.message for i in report.warnings if i.code == "publish_repo_placeholder")
    assert "placeholder" in msg


def test_publish_repo_no_slash_is_warning(tmp_path: Path, writable_font_bytes: bytes) -> None:
    """A ``publish.hf_dataset.repo`` without an ``owner/`` segment also warns.

    The placeholder check has *two* triggers: the literal
    ``CHANGE-ME/`` prefix and the absence of any ``/`` separator
    (i.e. a bare repo name with no namespace). Both surface the same
    code, but they're different code paths through the ``or``;
    parametrising guards against a refactor that drops one branch.
    """

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(_make_file(tmp_path / "seed.txt")),
    )
    yaml_text += "publish:\n  hf_dataset:\n    repo: bare-repo-name\n"
    recipe = load_recipe(_write(tmp_path, yaml_text))
    report = validate_recipe(recipe)
    codes = [i.code for i in report.warnings]
    assert "publish_repo_placeholder" in codes, [i.format() for i in report.issues]


def test_schema_version_unsupported_emission_is_defensive(
    tmp_path: Path, writable_font_bytes: bytes
) -> None:
    """Direct ``_check_schema_version`` test — pydantic blocks load otherwise.

    ``Recipe._validate_schema_version`` (a ``field_validator``) raises
    on load if ``schema_version`` is not in
    ``SUPPORTED_SCHEMA_VERSIONS``, so a real recipe never reaches the
    validator with a bad version. The ``_check_schema_version``
    branch is kept as a defence-in-depth: if a future refactor moves
    schema validation elsewhere, or if a recipe is constructed via
    ``Recipe.model_copy(update=...)`` (which bypasses field
    validators), the emission path still surfaces the mismatch.

    We exercise that path here so the catalog entry isn't dead code.
    """

    from pd_ocr_synth.validation import _check_schema_version

    font = tmp_path / "fake.otf"
    font.write_bytes(writable_font_bytes)
    seed = _make_file(tmp_path / "seed.txt", "hello\n")
    yaml_text = _minimal_yaml(
        font=str(font),
        dest=str(tmp_path / "out"),
        corpus=str(seed),
    )
    recipe = load_recipe(_write(tmp_path, yaml_text))
    # ``model_copy(update=...)`` skips field validators, so we can
    # mint a recipe instance with an unsupported version without
    # round-tripping through YAML.
    bad = recipe.model_copy(update={"schema_version": 999})
    issues = _check_schema_version(bad)
    codes = [i.code for i in issues]
    assert codes == ["schema_version_unsupported"], issues
    assert issues[0].severity == "error"
    assert "999" in issues[0].message


# ---------------------------------------------------------------------------
# VALIDATION_CODES catalog drift guard (runtime side)
#
# Mirrors the ``LINT_CODES`` runtime guard in ``test_lint.py``:
# ``VALIDATION_CODES`` is the source-of-truth set the spec doc compares
# against (see ``test_spec_docs.test_spec_01_validation_codes_match_VALIDATION_CODES``).
# This test locks the runtime half of the contract — every code emitted
# by ``validate_recipe`` must appear in ``VALIDATION_CODES``, so a new
# ``ValidationIssue(code=...)`` site can't ship undocumented.
#
# Strategy: static scan over ``src/pd_ocr_synth/validation.py``. The
# fixtures above directly assert each previously-uncovered code emits
# at the right severity, so the runtime ⊆ catalog *and* the
# every-code-has-a-fixture halves of the contract are both covered.
# ---------------------------------------------------------------------------


def test_VALIDATION_CODES_covers_every_emission_site() -> None:  # noqa: N802
    """Every ``code="..."`` in validation.py must appear in VALIDATION_CODES.

    Static scan of the source: walks every ``code="..."`` assignment
    site in ``src/pd_ocr_synth/validation.py`` and asserts the literal
    is in ``VALIDATION_CODES``. Any new emission site that lands
    without registering its code surfaces here, before users discover
    it via ``validate --json`` and grep'd CI logs.

    The static-scan approach is identical to what the spec-doc test
    does for the Markdown table; both directions of the catalog ↔
    runtime contract end up enforced cheaply.
    """

    import re as _re

    src = (
        Path(__file__).resolve().parent.parent / "src" / "pd_ocr_synth" / "validation.py"
    ).read_text(encoding="utf-8")
    emitted = set(_re.findall(r'code="([a-z_]+)"', src))
    leaked = emitted - VALIDATION_CODES
    assert not leaked, (
        f"validation.py emits code(s) not in VALIDATION_CODES: {sorted(leaked)}. "
        "Add them to src/pd_ocr_synth/validation.py:VALIDATION_CODES and "
        "document them in docs/specs/01-cli.md's 'Validation codes' table."
    )


def test_VALIDATION_CODES_no_stale_entries() -> None:  # noqa: N802
    """Every code in ``VALIDATION_CODES`` must have an emission site.

    Inverse of the previous test: walks ``src/pd_ocr_synth/validation.py``
    for every ``code="..."`` literal and confirms ``VALIDATION_CODES``
    contains no extras. A stale catalog entry left behind after a
    helper was removed (or after a code was renamed) surfaces here,
    keeping the docs honest.

    This is the no-dead-catalog-entry half of the contract; the
    runtime ⊆ catalog test above is the no-undocumented-emission half.
    Together they pin both directions, even though we don't (yet)
    require an in-module fixture per code — the static-scan pair is
    sufficient to prevent silent drift.
    """

    import re as _re

    src = (
        Path(__file__).resolve().parent.parent / "src" / "pd_ocr_synth" / "validation.py"
    ).read_text(encoding="utf-8")
    emitted = set(_re.findall(r'code="([a-z_]+)"', src))
    stale = sorted(VALIDATION_CODES - emitted)
    assert not stale, (
        f"VALIDATION_CODES contains code(s) not emitted by validation.py: {stale}. "
        "Drop them from src/pd_ocr_synth/validation.py:VALIDATION_CODES and "
        "from the 'Validation codes' table in docs/specs/01-cli.md."
    )


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
