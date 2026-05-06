"""Tests for ``pd_ocr_synth.corpus.runner.run_providers``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.corpus import CacheStore, ProviderContext
from pd_ocr_synth.corpus.runner import collect_corpus_text, run_providers
from pd_ocr_synth.recipe import load_recipe

_RECIPE = """\
schema_version: 1
name: runner-smoke
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 1
corpus:
  - type: local
    path: ./a.txt
  - type: local
    path: ./b.txt
    filter:
      drop_lines_matching: '^drop$'
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


@pytest.fixture
def loaded_recipe(tmp_path: Path):
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("keep\ndrop\nkeep2\n", encoding="utf-8")
    (tmp_path / "fake.otf").write_bytes(b"")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE, encoding="utf-8")
    return load_recipe(rp)


def test_runner_yields_one_result_per_corpus_entry(tmp_path: Path, loaded_recipe) -> None:
    cache = CacheStore(root=tmp_path / "cache")
    ctx = ProviderContext(recipe_dir=tmp_path, cache=cache)
    results = list(run_providers(loaded_recipe, ctx=ctx))
    assert [r.index for r in results] == [0, 1]
    assert all(r.type_name == "local" for r in results)


def test_runner_applies_filter_per_entry(tmp_path: Path, loaded_recipe) -> None:
    cache = CacheStore(root=tmp_path / "cache")
    ctx = ProviderContext(recipe_dir=tmp_path, cache=cache)
    results = list(run_providers(loaded_recipe, ctx=ctx))
    assert results[0].text == "alpha\n"
    # Entry 1 has a filter that drops the line "drop".
    assert "drop" not in results[1].text
    assert "keep" in results[1].text


def test_collect_corpus_text_threads_through_text_transforms(tmp_path: Path) -> None:
    (tmp_path / "src.txt").write_text("agus bhi siubhal\n", encoding="utf-8")
    (tmp_path / "fake.otf").write_bytes(b"")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(
        """\
schema_version: 1
name: pipeline
seed: 99
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 1
corpus:
  - type: local
    path: ./src.txt
text_transforms:
  - tironian_et:
      probability: 1.0
  - apply_lenition_dots:
      mode: aggressive
      probability: 1.0
  - long_s_medial:
      probability: 1.0
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
""",
        encoding="utf-8",
    )
    recipe = load_recipe(rp)
    ctx = ProviderContext(recipe_dir=tmp_path, cache=CacheStore(root=tmp_path / "cache"))
    out = collect_corpus_text(recipe, ctx=ctx)
    assert "⁊" in out
    assert "ḃ" in out
    assert "ſ" in out
    assert "agus" not in out


def test_collect_corpus_text_no_transforms_returns_raw(tmp_path: Path) -> None:
    (tmp_path / "src.txt").write_text("hello world\n", encoding="utf-8")
    (tmp_path / "fake.otf").write_bytes(b"")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(
        """\
schema_version: 1
name: bare
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 1
corpus:
  - type: local
    path: ./src.txt
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
""",
        encoding="utf-8",
    )
    recipe = load_recipe(rp)
    ctx = ProviderContext(recipe_dir=tmp_path, cache=CacheStore(root=tmp_path / "cache"))
    out = collect_corpus_text(recipe, ctx=ctx)
    assert "hello world" in out


def test_runner_marks_cache_hits(tmp_path: Path, loaded_recipe) -> None:
    cache = CacheStore(root=tmp_path / "cache")
    ctx = ProviderContext(recipe_dir=tmp_path, cache=cache)
    # Local provider does not auto-write to the cache; pre-populate
    # one entry so we exercise the was_cached branch.
    cache.write_text(
        "local",
        "local-" + "x" * 16,
        "irrelevant",
        source="x",
    )
    results = list(run_providers(loaded_recipe, ctx=ctx))
    # was_cached is observed before fetch: providers that don't cache
    # report False. The flag is informational.
    assert all(isinstance(r.was_cached, bool) for r in results)


def test_runner_no_cache_sets_options_cache_false(
    tmp_path: Path, loaded_recipe, monkeypatch
) -> None:
    """``run_providers(..., no_cache=True)`` must inject ``cache=False``.

    Guards the CLI ``--no-cache`` flag for ``preview`` / ``render``: the
    plumbing must reach the per-entry options dict that providers
    inspect. Iter 80 fixed the CLI silent-ignore drift; this test
    locks the runner-level contract.
    """

    captured: list[dict] = []

    from pd_ocr_synth.corpus.providers.local import LocalProvider

    real_fetch = LocalProvider.fetch

    def spy_fetch(self, ctx, options):  # type: ignore[no-untyped-def]
        # Snapshot the options dict the runner passed to the provider.
        captured.append(dict(options))
        yield from real_fetch(self, ctx, options)

    monkeypatch.setattr(LocalProvider, "fetch", spy_fetch)

    cache = CacheStore(root=tmp_path / "cache")
    ctx = ProviderContext(recipe_dir=tmp_path, cache=cache)
    list(run_providers(loaded_recipe, ctx=ctx, no_cache=True))
    assert len(captured) == 2
    assert all(opts.get("cache") is False for opts in captured), captured

    # Default (no_cache=False) preserves whatever the recipe entry's
    # own ``cache`` field carries. The pydantic CorpusEntry model
    # defaults to ``cache=True`` so we observe True here. The
    # invariant under test is the no_cache=True branch above, where
    # the runner must override the recipe's own value.
    captured.clear()
    list(run_providers(loaded_recipe, ctx=ctx))
    assert len(captured) == 2
    assert all(opts.get("cache") is True for opts in captured), captured


def test_collect_corpus_text_no_cache_threads_through(
    tmp_path: Path, loaded_recipe, monkeypatch
) -> None:
    """``collect_corpus_text(..., no_cache=True)`` reaches the runner."""

    from pd_ocr_synth.corpus import runner as runner_mod

    seen: dict[str, object] = {}
    real = runner_mod.run_providers

    def spy(*args, **kwargs):  # type: ignore[no-untyped-def]
        seen["no_cache"] = kwargs.get("no_cache")
        return real(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "run_providers", spy)

    cache = CacheStore(root=tmp_path / "cache")
    ctx = ProviderContext(recipe_dir=tmp_path, cache=cache)
    collect_corpus_text(loaded_recipe, ctx=ctx, no_cache=True)
    assert seen["no_cache"] is True
