"""End-to-end tests for ``pd-ocr-synth render`` with ``layout.mode = paragraphs``.

Exercises the M09 detection-mode dispatch chunk: a recipe with
``output.mode = detection`` + ``layout.mode = paragraphs`` should walk
``run_recipe`` through ``render_paragraph`` and into
:class:`DetectionWriter`, producing the ``pd-ocr-trainer/v1`` detection
layout (``images/page_*.png`` + ``labels.json`` with per-line polygons).

These tests skip when the bundled Bunchló GC font isn't present on
disk (e.g. fresh checkout that hasn't run
``./scripts/fetch-fonts-gaelic.sh``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pd_ocr_synth.cli import main
from pd_ocr_synth.output.detection import (
    LABELS_FILENAME,
    MANIFEST_FILENAME,
    PAGE_PREFIX,
    STATS_FILENAME,
)
from pd_ocr_synth.output.snapshot import SNAPSHOT_FILENAME

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; paragraphs render test skipped.")
    return _BUNDLED_FONT


# Recipe template: detection + paragraphs. The corpus is shaped so the
# tokenizer's ``_paragraphs`` splitter (blank-line separated chunks)
# produces multi-line paragraphs — each paragraph hands the renderer
# multiple lines that ``render_paragraph`` stacks on a single canvas.
_PARA_RECIPE = """\
schema_version: 1
name: render-paragraphs-smoke
seed: 31
output:
  format: pd-ocr-trainer/v1
  mode: detection
  destination: ./trainer-out
  count: 4
corpus:
  - type: local
    path: ./seed-paragraphs.txt
fonts:
  - path: {font}
    weight: 1.0
rendering:
  font_size_pt: {{ min: 14, max: 18 }}
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: paragraphs
  padding_px: 6
  line_spacing: {{ min: 1.1, max: 1.3 }}
"""

# Two paragraphs separated by a blank line. Each paragraph has two
# lines so the writer sees ``len(line_boxes) == 2`` per sample. The
# tokens stay short to keep the smoke test fast and to avoid bumping
# into the wrap-fitter's TODO (we hand pre-fitted lines).
_SEED_PARAGRAPHS = "ḃeaḋ saoġal mór\nċeann an ḃoṫair\n\nḋuine ḟir oġa\nġloine ṁaṫair óg\n"


def _setup(tmp_path: Path) -> Path:
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_PARA_RECIPE.format(font=font), encoding="utf-8")
    (tmp_path / "seed-paragraphs.txt").write_text(_SEED_PARAGRAPHS, encoding="utf-8")
    return rp


# ---------------------------------------------------------------------------
# Happy path: paragraphs-mode dispatch produces a full detection profile
# ---------------------------------------------------------------------------


def test_render_paragraphs_writes_trainer_v1_detection_layout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Smoke: detection profile sidecars all land alongside the images."""

    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(out),
            "--seed",
            "31",
            "--workers",
            "1",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    # Standard detection profile layout is produced.
    assert (out / "images").is_dir()
    assert (out / LABELS_FILENAME).exists()
    assert (out / MANIFEST_FILENAME).exists()
    assert (out / SNAPSHOT_FILENAME).exists()
    assert (out / STATS_FILENAME).exists()

    # Page filenames carry the ``page_`` prefix per spec 08.
    images = sorted((out / "images").glob("*.png"))
    assert images, "expected at least one page image"
    for img in images:
        assert img.name.startswith(PAGE_PREFIX), f"unexpected image filename: {img.name}"


def test_render_paragraphs_labels_carry_polygons_and_per_line_gt(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``labels.json`` has the doctr-shaped ``polygons`` + rich line GT.

    The detection writer emits two parallel structures per page:
    ``polygons`` (flat list, one 4-corner polygon per detected line —
    what doctr's ``DetectionDataset`` actually consumes) and ``lines``
    (rich GT with text + per-word bboxes). Both must be populated for a
    paragraph render with multiple lines.
    """

    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(out),
            "--seed",
            "31",
            "--workers",
            "1",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(labels, dict)
    assert len(labels) == 3

    seed_lines = {line for line in _SEED_PARAGRAPHS.splitlines() if line.strip()}

    for name, entry in labels.items():
        assert (out / "images" / name).exists()
        # Doctr-required fields are present and well-shaped.
        assert "img_dimensions" in entry
        w, h = entry["img_dimensions"]
        assert w > 0 and h > 0
        assert "img_hash" in entry and len(entry["img_hash"]) == 64

        # ``polygons`` is a flat list, one 4-corner polygon per line.
        polygons = entry["polygons"]
        assert isinstance(polygons, list) and polygons, f"empty polygons for {name}"
        for poly in polygons:
            assert len(poly) == 4, f"polygon must have 4 corners: {poly}"
            for pt in poly:
                assert len(pt) == 2

        # Rich ``lines`` payload: text + bbox + words for each line.
        lines = entry["lines"]
        assert isinstance(lines, list) and lines
        assert len(lines) == len(polygons)
        for line in lines:
            assert line["text"] in seed_lines, (
                f"line text not from seed corpus: {line['text']!r} (seeds={seed_lines!r})"
            )
            x0, y0, x1, y1 = line["bbox"]
            assert x1 > x0 and y1 > y0, f"degenerate line bbox: {line}"
            # Each line bbox sits inside the page canvas.
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
            # Each line has at least one word with a real bbox.
            words = line.get("words") or []
            assert words, f"line {line['text']!r} has no words"
            for wb in words:
                wx0, wy0, wx1, wy1 = wb["bbox"]
                assert wx1 > wx0 and wy1 > wy0


def test_render_paragraphs_manifest_carries_line_and_word_counts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Manifest rows record ``n_lines`` / ``n_words`` for the run summary."""

    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(out),
            "--seed",
            "31",
            "--workers",
            "1",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    rendered_records = []
    for line in (out / MANIFEST_FILENAME).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") == "rendered":
            rendered_records.append(rec)
    assert rendered_records, "expected at least one rendered manifest row"

    for rec in rendered_records:
        assert rec["n_lines"] >= 1
        assert rec["n_words"] >= rec["n_lines"], f"expected at least one word per line, got {rec}"
        # Filename uses the page_ prefix per spec 08.
        assert rec["image"].startswith(f"images/{PAGE_PREFIX}")


def test_render_paragraphs_serial_and_parallel_produce_same_labels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Worker count is not load-bearing on the detection labels.

    Same determinism contract as recognition: same recipe + same seed
    + same indices → identical ``labels.json`` regardless of
    ``workers``. Crucially, this verifies the parallel-worker payload
    round-trips ``line_boxes`` (per spec 08 §Detection mode layout —
    polygons are derived from the rendered paragraph's line bboxes,
    not recomputed by the writer).
    """

    rp = _setup(tmp_path)
    serial_out = tmp_path / "serial"
    parallel_out = tmp_path / "parallel"

    rc1 = main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(serial_out),
            "--seed",
            "31",
            "--workers",
            "1",
        ]
    )
    assert rc1 == 0, capsys.readouterr().err

    rc2 = main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(parallel_out),
            "--seed",
            "31",
            "--workers",
            "2",
        ]
    )
    assert rc2 == 0, capsys.readouterr().err

    serial_labels = json.loads((serial_out / LABELS_FILENAME).read_text())
    parallel_labels = json.loads((parallel_out / LABELS_FILENAME).read_text())
    assert serial_labels == parallel_labels


def test_render_paragraphs_parallel_carries_line_boxes_through_workers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The multi-worker path must round-trip per-line bboxes.

    Regression test for the parallel payload extension: without
    ``line_boxes`` in the worker → parent payload, the
    :class:`DetectionWriter` would receive a sample shim with empty
    ``line_boxes`` and silently emit no detection polygons. Per the
    no-silent-drop rule, words would land in ``unassigned_words``
    instead — this test asserts the writer sees real lines, not the
    fallback.
    """

    rp = _setup(tmp_path)
    out = tmp_path / "parallel-out"

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(out),
            "--seed",
            "31",
            "--workers",
            "2",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    for name, entry in labels.items():
        assert entry["lines"], f"parallel-rendered {name} lost line_boxes — got {entry}"
        # No words spilled into the page-level fallback.
        assert "unassigned_words" not in entry, (
            f"parallel-rendered {name} unexpectedly had unassigned words: {entry}"
        )
