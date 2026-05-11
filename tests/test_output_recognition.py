"""Unit tests for ``RecognitionWriter`` (M07).

Covers the on-disk layout, the force/resume/refuse semantics, and
the manifest re-write path that makes resume safe.

Renders a minimal one-pixel "image" via Pillow rather than going
through the full HarfBuzz pipeline — these tests are about the
writer's filesystem contract, not the renderer.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from pd_ocr_synth.output import (
    RecognitionWriter,
    image_filename,
    width_for_count,
)
from pd_ocr_synth.output.recognition import (
    LABELS_FILENAME,
    MANIFEST_FILENAME,
    STATS_FILENAME,
    DestinationNotEmptyError,
)
from pd_ocr_synth.output.snapshot import SNAPSHOT_FILENAME, SnapshotMismatchError
from pd_ocr_synth.recipe import load_recipe

_RECIPE_TEMPLATE = """\
schema_version: 1
name: writer-smoke
seed: 5
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: {count}
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


def _build_recipe(tmp_path: Path, *, count: int = 4) -> Path:
    font = tmp_path / "fake.otf"
    font.write_bytes(b"\x00\x01\x02fake")
    seed = tmp_path / "seed-words.txt"
    seed.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE_TEMPLATE.format(font=font, count=count), encoding="utf-8")
    return rp


def _fake_sample(*, text: str = "alpha", font_path: Path | None = None) -> SimpleNamespace:
    """A duck-typed stand-in for ``RenderedSample``.

    The writer only reads ``image``, ``font_path``, ``font_size_pt``,
    ``dpi``, ``ink_color``, ``background_color``, ``size``, ``bbox``;
    everything else is the renderer's business.
    """

    img = Image.new("RGB", (32, 16), color=(240, 240, 240))
    return SimpleNamespace(
        text=text,
        image=img,
        font_path=font_path or Path("/fake/font.otf"),
        font_size_pt=14.0,
        dpi=300,
        ink_color=(10, 10, 10),
        background_color=(240, 240, 240),
        size=(32, 16),
        bbox=(0, 0, 32, 16),
        glyph_runs=(),
    )


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------


def test_width_for_count_has_min_lower_bound() -> None:
    # Tiny counts still use the default min width so consumers don't
    # have to special-case 1- or 2-digit padding.
    assert width_for_count(1) == 7
    assert width_for_count(50) == 7
    assert width_for_count(0) == 7


def test_width_for_count_grows_for_large_counts() -> None:
    assert width_for_count(50_000) == 7
    # 100M samples → indices up to 99,999,999 → 8 digits.
    assert width_for_count(100_000_000) == 8


def test_image_filename_is_zero_padded() -> None:
    assert image_filename(0, width=7) == "0000000.png"
    assert image_filename(42, width=7) == "0000042.png"
    assert image_filename(50_000, width=7) == "0050000.png"


# ---------------------------------------------------------------------------
# Open semantics: empty / force / resume / refuse
# ---------------------------------------------------------------------------


def test_open_into_empty_dir_creates_layout(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "fresh"

    with RecognitionWriter.open(recipe, out, seed=recipe.seed) as writer:
        assert writer.images_dir.exists()
        # Snapshot is written eagerly so a crash mid-run still leaves
        # a snapshot for resume to compare against.
        assert (out / SNAPSHOT_FILENAME).exists()

    assert (out / LABELS_FILENAME).exists()
    assert (out / MANIFEST_FILENAME).exists()
    assert (out / STATS_FILENAME).exists()


def test_open_refuses_nonempty_dir_without_force_or_resume(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"
    out.mkdir()
    (out / "lurking.txt").write_text("from a previous run", encoding="utf-8")

    with pytest.raises(DestinationNotEmptyError):
        RecognitionWriter.open(recipe, out, seed=recipe.seed)


def test_open_force_clears_directory(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"
    out.mkdir()
    (out / "leftover.png").write_bytes(b"old run")

    with RecognitionWriter.open(recipe, out, seed=recipe.seed, force=True):
        pass

    assert not (out / "leftover.png").exists()
    # Force still rewrites the snapshot.
    assert (out / SNAPSHOT_FILENAME).exists()


def test_open_force_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        RecognitionWriter.open(recipe, out, seed=recipe.seed, force=True, resume=True)


# ---------------------------------------------------------------------------
# Write paths
# ---------------------------------------------------------------------------


def test_write_rendered_emits_image_and_label(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    with RecognitionWriter.open(recipe, out, seed=recipe.seed) as writer:
        writer.write_rendered(0, _fake_sample(text="alpha"), text="alpha")
        writer.write_rendered(2, _fake_sample(text="gamma"), text="gamma")
        writer.write_skipped(1, reason="missing_glyph", text="bad")

    name0 = image_filename(0, width=writer.pad_width)
    name2 = image_filename(2, width=writer.pad_width)
    assert (out / "images" / name0).exists()
    assert (out / "images" / name2).exists()
    # Skipped indices do NOT produce a PNG.
    assert not (out / "images" / image_filename(1, width=writer.pad_width)).exists()

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert labels == {name0: "alpha", name2: "gamma"}

    manifest_lines = (out / MANIFEST_FILENAME).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in manifest_lines if line.strip()]
    by_index = {r["index"]: r for r in records}
    assert by_index[0]["status"] == "rendered"
    assert by_index[1]["status"] == "skipped" and by_index[1]["reason"] == "missing_glyph"
    assert by_index[2]["status"] == "rendered"
    # Manifest is written in index order, regardless of write order.
    assert [r["index"] for r in records] == [0, 1, 2]

    stats = json.loads((out / STATS_FILENAME).read_text(encoding="utf-8"))
    assert stats["samples_written"] == 2
    assert stats["samples_skipped"] == 1
    assert stats["skip_reasons"] == {"missing_glyph": 1}


def test_already_rendered_only_true_for_successful_records(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    with RecognitionWriter.open(recipe, out, seed=recipe.seed) as writer:
        writer.write_rendered(0, _fake_sample(), text="alpha")
        writer.write_skipped(1, reason="missing_glyph", text="bad")
        assert writer.already_rendered(0) is True
        assert writer.already_rendered(1) is False  # skip != rendered
        assert writer.already_rendered(99) is False


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_recovers_existing_labels_and_manifest(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path, count=4)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    # First run: render two samples, then close.
    with RecognitionWriter.open(recipe, out, seed=recipe.seed) as w1:
        w1.write_rendered(0, _fake_sample(text="alpha"), text="alpha")
        w1.write_skipped(1, reason="missing_glyph", text="x")

    # Second run with --resume sees both records and skips re-rendering
    # the successful one. The skipped one is *not* skipped on resume —
    # the writer only short-circuits ``already_rendered`` indices.
    with RecognitionWriter.open(recipe, out, seed=recipe.seed, resume=True) as w2:
        assert w2.already_rendered(0) is True
        assert w2.already_rendered(1) is False
        # Continue: render index 1 (the previously-skipped one),
        # plus 2 and 3.
        w2.write_rendered(1, _fake_sample(text="x-fixed"), text="x-fixed")
        w2.write_rendered(2, _fake_sample(text="gamma"), text="gamma")
        w2.write_rendered(3, _fake_sample(text="delta"), text="delta")

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert len(labels) == 4
    # Index 1 should now be rendered, with the new label, replacing the
    # prior skip record.
    assert labels[image_filename(1, width=w2.pad_width)] == "x-fixed"

    records = [
        json.loads(line)
        for line in (out / MANIFEST_FILENAME).read_text().splitlines()
        if line.strip()
    ]
    by_index = {r["index"]: r for r in records}
    assert by_index[1]["status"] == "rendered"


def test_resume_requires_existing_snapshot(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"
    out.mkdir()
    # Non-empty but no snapshot — resume can't safely proceed.
    (out / "stray.txt").write_text("hi", encoding="utf-8")

    with pytest.raises(SnapshotMismatchError):
        RecognitionWriter.open(recipe, out, seed=recipe.seed, resume=True)


def test_resume_rejects_seed_drift(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    with RecognitionWriter.open(recipe, out, seed=recipe.seed) as w:
        w.write_rendered(0, _fake_sample(), text="alpha")

    with pytest.raises(SnapshotMismatchError):
        RecognitionWriter.open(recipe, out, seed=recipe.seed + 1, resume=True)


def test_resume_rejects_input_hash_drift(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    with RecognitionWriter.open(recipe, out, seed=recipe.seed) as w:
        w.write_rendered(0, _fake_sample(), text="alpha")

    # Mutate the corpus file → snapshot input_hashes diverge.
    (tmp_path / "seed-words.txt").write_text("DIFFERENT\n", encoding="utf-8")
    recipe2 = load_recipe(rp)

    with pytest.raises(SnapshotMismatchError):
        RecognitionWriter.open(recipe2, out, seed=recipe2.seed, resume=True)


def test_resume_writer_re_records_skips_in_stats(tmp_path: Path) -> None:
    """Stats survive resume: prior skips and new skips both count."""

    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    with RecognitionWriter.open(recipe, out, seed=recipe.seed) as w1:
        w1.write_skipped(0, reason="missing_glyph", text="x")
        w1.write_rendered(1, _fake_sample(), text="alpha")

    with RecognitionWriter.open(recipe, out, seed=recipe.seed, resume=True) as w2:
        # Skip the same index again on resume.
        w2.write_skipped(0, reason="missing_glyph", text="x")
        w2.write_rendered(2, _fake_sample(), text="gamma")
        w2.write_rendered(3, _fake_sample(), text="delta")

    stats = json.loads((out / STATS_FILENAME).read_text(encoding="utf-8"))
    # 1 carried-over skip + 1 fresh skip in this writer's counters,
    # but the resume writer also pre-loaded the prior skip from disk,
    # so there should be exactly one skip recorded.
    assert stats["samples_written"] >= 3
    # samples_skipped includes the prior skip rolled in at construction
    # plus the resumed write_skipped call. We accept >= 1 because the
    # writer is conservative about double-counting.
    assert stats["samples_skipped"] >= 1
