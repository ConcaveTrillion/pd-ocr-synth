"""Unit tests for ``pd_ocr_synth.output.snapshot``.

Covers the snapshot build/write/load lifecycle and the
``snapshot_matches`` resume-eligibility comparator.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pd_ocr_synth.output.snapshot import (
    SNAPSHOT_FILENAME,
    SnapshotMismatchError,
    build_snapshot,
    load_snapshot,
    snapshot_matches,
    write_snapshot,
)
from pd_ocr_synth.recipe import load_recipe

_RECIPE_TEMPLATE = """\
schema_version: 1
name: snapshot-smoke
seed: 11
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 4
corpus:
  - type: local
    path: ./seed-words.txt
fonts:
  - path: {font}
    weight: 1.0
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 240, b: 240 }}
layout:
  mode: word_crops
  padding_px: 4
"""


@pytest.fixture
def tiny_recipe(tmp_path: Path) -> Path:
    """A fully-resolved recipe whose font + corpus paths exist on disk."""

    font = tmp_path / "fake.otf"
    # The snapshot path doesn't open the font, just hashes the bytes.
    # We can use any non-empty file.
    font.write_bytes(b"\x00\x01\x02\x03this is fake but hashable")
    seed_words = tmp_path / "seed-words.txt"
    seed_words.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE_TEMPLATE.format(font=font), encoding="utf-8")
    return rp


def test_build_snapshot_includes_tool_version_seed_and_recipe(tiny_recipe: Path) -> None:
    recipe = load_recipe(tiny_recipe)
    snap = build_snapshot(recipe, seed=99)

    assert "tool_version" in snap and isinstance(snap["tool_version"], str)
    assert snap["seed"] == 99
    assert snap["recipe"]["name"] == "snapshot-smoke"
    # source_path is loader metadata; must not leak into the snapshot.
    assert "source_path" not in snap["recipe"]


def test_build_snapshot_hashes_inputs(tiny_recipe: Path, tmp_path: Path) -> None:
    recipe = load_recipe(tiny_recipe)
    snap = build_snapshot(recipe, seed=recipe.seed)
    hashes = snap["input_hashes"]

    # One font + one local corpus entry.
    assert len(hashes) == 2
    assert all(isinstance(v, str) and v != "<missing>" for v in hashes.values())
    assert all(":" in k for k in hashes)  # role-prefixed keys

    # Mutating the seed-words file changes the hash.
    seed_words = tmp_path / "seed-words.txt"
    seed_words.write_text("alpha\nbeta\ngamma\nDELTA\n", encoding="utf-8")
    snap2 = build_snapshot(recipe, seed=recipe.seed)
    assert snap2["input_hashes"] != snap["input_hashes"]


def test_build_snapshot_marks_missing_inputs(tmp_path: Path, tiny_recipe: Path) -> None:
    recipe = load_recipe(tiny_recipe)
    # Delete the font file: the snapshot should still build, but with a
    # ``<missing>`` marker so downstream resume checks can see the drift.
    Path(recipe.fonts[0].path).unlink()
    snap = build_snapshot(recipe, seed=recipe.seed)
    font_key = next(k for k in snap["input_hashes"] if k.startswith("font:"))
    assert snap["input_hashes"][font_key] == "<missing>"


def test_write_and_load_snapshot_roundtrip(tiny_recipe: Path, tmp_path: Path) -> None:
    recipe = load_recipe(tiny_recipe)
    snap = build_snapshot(recipe, seed=recipe.seed)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    written = write_snapshot(snap, out_dir)
    assert written.name == SNAPSHOT_FILENAME
    assert written.exists()

    # YAML round-trip — content matches.
    parsed = yaml.safe_load(written.read_text(encoding="utf-8"))
    assert parsed["seed"] == recipe.seed
    assert parsed["recipe"]["name"] == "snapshot-smoke"

    # ``load_snapshot`` is the convenience wrapper used by the writer.
    loaded = load_snapshot(out_dir)
    assert loaded is not None
    assert loaded["seed"] == recipe.seed


def test_load_snapshot_returns_none_when_missing(tmp_path: Path) -> None:
    out_dir = tmp_path / "fresh"
    out_dir.mkdir()
    assert load_snapshot(out_dir) is None


def test_load_snapshot_rejects_non_mapping(tmp_path: Path) -> None:
    out_dir = tmp_path / "bad"
    out_dir.mkdir()
    (out_dir / SNAPSHOT_FILENAME).write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(SnapshotMismatchError):
        load_snapshot(out_dir)


def test_snapshot_matches_identical_pair(tiny_recipe: Path) -> None:
    recipe = load_recipe(tiny_recipe)
    snap = build_snapshot(recipe, seed=recipe.seed)
    ok, reason = snapshot_matches(snap, snap)
    assert ok and reason is None


def test_snapshot_matches_detects_seed_change(tiny_recipe: Path) -> None:
    recipe = load_recipe(tiny_recipe)
    a = build_snapshot(recipe, seed=1)
    b = build_snapshot(recipe, seed=2)
    ok, reason = snapshot_matches(a, b)
    assert not ok
    assert reason is not None and "seed" in reason


def test_snapshot_matches_detects_input_drift(tiny_recipe: Path, tmp_path: Path) -> None:
    recipe = load_recipe(tiny_recipe)
    a = build_snapshot(recipe, seed=recipe.seed)
    # Mutate the corpus file -> drift.
    (tmp_path / "seed-words.txt").write_text("changed!\n", encoding="utf-8")
    b = build_snapshot(recipe, seed=recipe.seed)
    ok, reason = snapshot_matches(a, b)
    assert not ok
    assert reason is not None and "hash" in reason


def test_snapshot_matches_allows_count_growth(tiny_recipe: Path) -> None:
    recipe = load_recipe(tiny_recipe)
    a = build_snapshot(recipe, seed=recipe.seed)
    # Simulate a follow-up run with a higher count by mutating the dict.
    b = build_snapshot(recipe, seed=recipe.seed)
    b["recipe"]["output"]["count"] = recipe.output.count + 100
    ok, reason = snapshot_matches(a, b, allow_count_growth=True)
    assert ok and reason is None


def test_snapshot_matches_rejects_count_shrink(tiny_recipe: Path) -> None:
    recipe = load_recipe(tiny_recipe)
    a = build_snapshot(recipe, seed=recipe.seed)
    b = build_snapshot(recipe, seed=recipe.seed)
    # Pretend the prior run had a higher count; a count shrink is
    # nonsensical for resume (we'd never reach that index again).
    a["recipe"]["output"]["count"] = recipe.output.count + 100
    ok, reason = snapshot_matches(a, b)
    assert not ok
    assert reason is not None and "shrank" in reason


def test_snapshot_matches_detects_recipe_diff(tiny_recipe: Path) -> None:
    recipe = load_recipe(tiny_recipe)
    a = build_snapshot(recipe, seed=recipe.seed)
    b = build_snapshot(recipe, seed=recipe.seed)
    # Twiddle a non-count field.
    b["recipe"]["rendering"]["font_size_pt"] = 18
    ok, reason = snapshot_matches(a, b)
    assert not ok
    assert reason is not None and "recipe" in reason
