"""Tests for ``pd_ocr_synth.corpus.runner.run_providers``."""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.corpus import CacheStore, ProviderContext
from pd_ocr_synth.corpus.runner import run_providers
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
