"""Unit tests for ``DetectionWriter`` (M09).

Covers the on-disk layout for the detection profile, the bbox→polygon
conversion that doctr's ``DetectionDataset`` reader expects, the
force/resume/refuse semantics (mirrored from the recognition writer),
and the line/word grouping logic so we don't silently drop ground
truth when a word_box can't be matched to a line.

These tests construct fake samples via ``SimpleNamespace`` rather
than the full HarfBuzz pipeline — this is a writer-contract test,
not a renderer test.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from PIL import Image

from pd_ocr_synth.output import (
    DetectionStats,
    DetectionWriter,
    bbox_to_polygon,
    page_filename,
)
from pd_ocr_synth.output.detection import (
    LABELS_FILENAME,
    MANIFEST_FILENAME,
    PAGE_PREFIX,
    STATS_FILENAME,
    DestinationNotEmptyError,
    width_for_count,
)
from pd_ocr_synth.output.snapshot import SNAPSHOT_FILENAME, SnapshotMismatchError
from pd_ocr_synth.recipe import load_recipe

_RECIPE_TEMPLATE = """\
schema_version: 1
name: detection-writer-smoke
seed: 5
output:
  format: pd-ocr-trainer/v1
  mode: detection
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
  mode: paragraphs
  padding_px: 4
"""


def _build_recipe(tmp_path: Path, *, count: int = 4) -> Path:
    font = tmp_path / "fake.otf"
    font.write_bytes(b"\x00\x01\x02fake")
    seed = tmp_path / "seed-words.txt"
    seed.write_text("alpha beta gamma\ndelta epsilon\n", encoding="utf-8")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE_TEMPLATE.format(font=font, count=count), encoding="utf-8")
    return rp


def _line_box(text: str, bbox: tuple[int, int, int, int]) -> SimpleNamespace:
    return SimpleNamespace(text=text, bbox=bbox)


def _word_box(text: str, bbox: tuple[int, int, int, int]) -> SimpleNamespace:
    return SimpleNamespace(text=text, bbox=bbox)


def _fake_page_sample(
    *,
    font_path: Path | None = None,
    line_boxes: tuple[Any, ...] = (),
    word_boxes: tuple[Any, ...] = (),
    size: tuple[int, int] = (200, 100),
) -> SimpleNamespace:
    """Duck-typed stand-in for ``RenderedSample`` for detection tests."""

    img = Image.new("RGB", size, color=(240, 240, 240))
    return SimpleNamespace(
        text="\n".join(getattr(lb, "text", "") for lb in line_boxes),
        image=img,
        font_path=font_path or Path("/fake/font.otf"),
        font_size_pt=14.0,
        dpi=300,
        ink_color=(10, 10, 10),
        background_color=(240, 240, 240),
        size=size,
        bbox=(0, 0, size[0], size[1]),
        glyph_runs=(),
        line_boxes=line_boxes,
        word_boxes=word_boxes,
    )


# ---------------------------------------------------------------------------
# Polygon / filename helpers
# ---------------------------------------------------------------------------


def test_bbox_to_polygon_is_clockwise_from_top_left() -> None:
    poly = bbox_to_polygon((10, 20, 30, 40))
    # Clockwise from TL: TL, TR, BR, BL.
    assert poly == [[10, 20], [30, 20], [30, 40], [10, 40]]


def test_bbox_to_polygon_round_trips_through_doctr_min_max() -> None:
    # doctr does ``np.concatenate((poly.min(axis=1), poly.max(axis=1)))``
    # to recover (xmin, ymin, xmax, ymax) from a 4-corner polygon. Make
    # sure our polygon survives that exact transform.
    poly = bbox_to_polygon((5, 7, 11, 13))
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    assert (min(xs), min(ys), max(xs), max(ys)) == (5, 7, 11, 13)


def test_page_filename_uses_page_prefix_and_pad() -> None:
    assert page_filename(0, width=7) == f"{PAGE_PREFIX}0000000.png"
    assert page_filename(42, width=7) == f"{PAGE_PREFIX}0000042.png"


def test_width_for_count_floor_matches_recognition() -> None:
    # Detection writer uses the same min width as recognition so a
    # mixed profile dir doesn't have differently-padded indices.
    assert width_for_count(1) == 7
    assert width_for_count(50_000) == 7
    assert width_for_count(100_000_000) == 8


# ---------------------------------------------------------------------------
# Open semantics
# ---------------------------------------------------------------------------


def test_open_into_empty_dir_creates_layout(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "fresh"

    with DetectionWriter.open(recipe, out, seed=recipe.seed) as writer:
        assert writer.images_dir.exists()
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
        DetectionWriter.open(recipe, out, seed=recipe.seed)


def test_open_force_clears_directory(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"
    out.mkdir()
    (out / "leftover.png").write_bytes(b"old run")

    with DetectionWriter.open(recipe, out, seed=recipe.seed, force=True):
        pass

    assert not (out / "leftover.png").exists()
    assert (out / SNAPSHOT_FILENAME).exists()


def test_open_force_and_resume_are_mutually_exclusive(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"
    with pytest.raises(ValueError):
        DetectionWriter.open(recipe, out, seed=recipe.seed, force=True, resume=True)


# ---------------------------------------------------------------------------
# Write paths
# ---------------------------------------------------------------------------


def test_write_rendered_emits_image_and_label_with_polygons(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    line1 = _line_box("alpha beta", (10, 10, 110, 30))
    line2 = _line_box("gamma", (10, 40, 70, 60))
    words = (
        _word_box("alpha", (10, 12, 50, 28)),
        _word_box("beta", (60, 12, 110, 28)),
        _word_box("gamma", (10, 42, 70, 58)),
    )

    with DetectionWriter.open(recipe, out, seed=recipe.seed) as writer:
        sample = _fake_page_sample(
            line_boxes=(line1, line2),
            word_boxes=words,
            size=(200, 100),
        )
        writer.write_rendered(0, sample)

    name = page_filename(0, width=7)
    assert (out / "images" / name).exists()

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert set(labels.keys()) == {name}
    entry = labels[name]
    # doctr fields
    assert entry["img_dimensions"] == [200, 100]
    assert isinstance(entry["img_hash"], str) and len(entry["img_hash"]) == 64
    assert len(entry["polygons"]) == 2
    assert entry["polygons"][0] == bbox_to_polygon((10, 10, 110, 30))
    # rich GT
    assert len(entry["lines"]) == 2
    assert entry["lines"][0]["text"] == "alpha beta"
    assert [w["text"] for w in entry["lines"][0]["words"]] == ["alpha", "beta"]
    assert [w["text"] for w in entry["lines"][1]["words"]] == ["gamma"]
    assert "unassigned_words" not in entry


def test_write_rendered_keeps_orphan_words_under_unassigned(tmp_path: Path) -> None:
    """A word_box that doesn't fall in any line's vertical span lands
    in ``unassigned_words`` rather than being silently dropped — see the
    no-silent-word-drop project rule."""

    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    line1 = _line_box("alpha", (10, 10, 50, 30))
    orphan = _word_box("footnote", (10, 80, 80, 95))  # below line1's span

    with DetectionWriter.open(recipe, out, seed=recipe.seed) as writer:
        sample = _fake_page_sample(
            line_boxes=(line1,),
            word_boxes=(orphan,),
            size=(200, 100),
        )
        writer.write_rendered(0, sample)

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    entry = labels[page_filename(0, width=7)]
    assert entry["lines"][0]["words"] == []
    assert entry["unassigned_words"] == [{"text": "footnote", "bbox": [10, 80, 80, 95]}]


def test_write_rendered_falls_back_to_word_polygons_when_no_lines(tmp_path: Path) -> None:
    """If the renderer gave us word_boxes but no line_boxes (a degenerate
    configuration we still want to handle without crashing), top-level
    polygons fall back to per-word boxes so doctr still has training
    targets."""

    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"
    words = (
        _word_box("alpha", (10, 12, 50, 28)),
        _word_box("beta", (60, 12, 110, 28)),
    )

    with DetectionWriter.open(recipe, out, seed=recipe.seed) as writer:
        sample = _fake_page_sample(line_boxes=(), word_boxes=words)
        writer.write_rendered(0, sample)

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    entry = labels[page_filename(0, width=7)]
    assert entry["lines"] == []
    assert len(entry["polygons"]) == 2


def test_write_skipped_records_reason_without_image(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    with DetectionWriter.open(recipe, out, seed=recipe.seed) as writer:
        writer.write_skipped(0, reason="missing_glyph")
        writer.write_skipped(1, reason="no_corpus_token")

    name = page_filename(0, width=7)
    assert not (out / "images" / name).exists()
    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert labels == {}

    manifest_lines = (out / MANIFEST_FILENAME).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in manifest_lines if line.strip()]
    assert [(r["index"], r["status"], r["reason"]) for r in records] == [
        (0, "skipped", "missing_glyph"),
        (1, "skipped", "no_corpus_token"),
    ]

    stats = json.loads((out / STATS_FILENAME).read_text(encoding="utf-8"))
    assert stats["samples_skipped"] == 2
    assert stats["skip_reasons"] == {
        "missing_glyph": 1,
        "no_corpus_token": 1,
    }


def test_already_rendered_only_true_for_successful_records(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    with DetectionWriter.open(recipe, out, seed=recipe.seed) as writer:
        line = _line_box("alpha", (10, 10, 50, 30))
        sample = _fake_page_sample(
            line_boxes=(line,),
            word_boxes=(_word_box("alpha", (10, 12, 50, 28)),),
        )
        writer.write_rendered(0, sample)
        writer.write_skipped(1, reason="missing_glyph")
        assert writer.already_rendered(0) is True
        assert writer.already_rendered(1) is False  # skip != rendered
        assert writer.already_rendered(99) is False


def test_skipped_after_rendered_clears_stale_image_and_label(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    with DetectionWriter.open(recipe, out, seed=recipe.seed) as writer:
        sample = _fake_page_sample(
            line_boxes=(_line_box("alpha", (10, 10, 50, 30)),),
            word_boxes=(),
        )
        writer.write_rendered(0, sample)
        name = writer.filename(0)
        assert (out / "images" / name).exists()

        # Re-classify same index as skipped — image + label must go.
        writer.write_skipped(0, reason="late_validation")
        assert not (out / "images" / name).exists()

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert labels == {}


def test_write_after_close_raises(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    writer = DetectionWriter.open(recipe, out, seed=recipe.seed)
    writer.close()
    with pytest.raises(RuntimeError):
        writer.write_rendered(
            0,
            _fake_page_sample(line_boxes=(_line_box("a", (0, 0, 10, 10)),)),
        )
    with pytest.raises(RuntimeError):
        writer.write_skipped(0, reason="x")


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_resume_recovers_existing_labels_and_manifest(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path, count=4)
    recipe = load_recipe(rp)
    out = tmp_path / "out"

    line0 = _line_box("alpha", (10, 10, 50, 30))
    word0 = _word_box("alpha", (10, 12, 50, 28))

    # First run: render one page, skip another.
    with DetectionWriter.open(recipe, out, seed=recipe.seed) as w1:
        w1.write_rendered(0, _fake_page_sample(line_boxes=(line0,), word_boxes=(word0,)))
        w1.write_skipped(1, reason="missing_glyph")

    # Second run with --resume sees the previous record and reports it.
    with DetectionWriter.open(recipe, out, seed=recipe.seed, resume=True) as w2:
        assert w2.already_rendered(0) is True
        assert w2.already_rendered(1) is False
        # Continue: render index 1 (the previously-skipped one) plus 2.
        w2.write_rendered(
            1,
            _fake_page_sample(
                line_boxes=(_line_box("beta", (10, 10, 40, 30)),),
                word_boxes=(_word_box("beta", (10, 12, 40, 28)),),
            ),
        )
        w2.write_rendered(
            2,
            _fake_page_sample(
                line_boxes=(_line_box("gamma", (10, 10, 60, 30)),),
                word_boxes=(_word_box("gamma", (10, 12, 60, 28)),),
            ),
        )

    # Final state has all three pages in labels.json and manifest.
    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert {page_filename(i, width=7) for i in (0, 1, 2)} == set(labels.keys())

    manifest_lines = (out / MANIFEST_FILENAME).read_text(encoding="utf-8").splitlines()
    records = [json.loads(line) for line in manifest_lines if line.strip()]
    by_idx = {r["index"]: r["status"] for r in records}
    # Index 1 transitioned from skipped → rendered; only the latest
    # status survives in the manifest.
    assert by_idx == {0: "rendered", 1: "rendered", 2: "rendered"}

    stats = json.loads((out / STATS_FILENAME).read_text(encoding="utf-8"))
    assert stats["samples_written"] == 3
    # Stats are an accumulator: the original skip from the first run
    # is preserved in the count even though the on-disk manifest line
    # was overwritten by the later rendered record. Same semantics as
    # the recognition writer — stats reflect attempts across runs,
    # not unique final outcomes.
    assert stats["samples_skipped"] == 1
    assert stats["skip_reasons"] == {"missing_glyph": 1}


def test_resume_without_snapshot_raises(tmp_path: Path) -> None:
    rp = _build_recipe(tmp_path)
    recipe = load_recipe(rp)
    out = tmp_path / "out"
    out.mkdir()
    (out / "stale.txt").write_text("nope", encoding="utf-8")

    with pytest.raises(SnapshotMismatchError):
        DetectionWriter.open(recipe, out, seed=recipe.seed, resume=True)


# ---------------------------------------------------------------------------
# Stats dataclass
# ---------------------------------------------------------------------------


def test_detection_stats_record_render_aggregates_lines_and_words() -> None:
    s = DetectionStats(samples_planned=10)
    s.record_render(font_name="A.otf", n_lines=3, n_words=12)
    s.record_render(font_name="A.otf", n_lines=2, n_words=8)
    s.record_render(font_name="B.otf", n_lines=4, n_words=16)
    assert s.samples_written == 3
    assert s.lines_total == 9
    assert s.words_total == 36
    assert s.fonts_used == {"A.otf": 2, "B.otf": 1}


def test_detection_stats_record_skip_counts_reasons() -> None:
    s = DetectionStats(samples_planned=5)
    s.record_skip("missing_glyph")
    s.record_skip("missing_glyph")
    s.record_skip("no_corpus_token")
    assert s.samples_skipped == 3
    assert s.skip_reasons == {"missing_glyph": 2, "no_corpus_token": 1}
