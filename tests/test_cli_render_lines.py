"""End-to-end tests for ``pd-ocr-synth render`` with ``layout.mode = lines``.

Exercises the M09 dispatch chunk: a recipe with ``output.mode =
recognition`` + ``layout.mode = lines`` should walk the same writer
pipeline as ``word_crops`` but with one full line of text per sample,
plus per-word bboxes carried into ``manifest.jsonl``.

These tests skip when the bundled Bunchló GC font isn't present on
disk (e.g. fresh checkout that hasn't run
``./scripts/fetch-fonts-gaelic.sh``).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pd_ocr_synth.cli import main
from pd_ocr_synth.output.recognition import (
    LABELS_FILENAME,
    MANIFEST_FILENAME,
    STATS_FILENAME,
)
from pd_ocr_synth.output.snapshot import SNAPSHOT_FILENAME

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; lines render test skipped.")
    return _BUNDLED_FONT


# Recipe templates intentionally use ``layout.mode: lines`` and a corpus
# whose lines map 1-to-1 to expected manifest rows. The tokenizer's
# ``_lines`` splitter strips blank lines so we don't have to worry about
# empty entries reaching the renderer.
_LINES_RECIPE = """\
schema_version: 1
name: render-lines-smoke
seed: 31
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./trainer-out
  count: 8
corpus:
  - type: local
    path: ./seed-lines.txt
fonts:
  - path: {font}
    weight: 1.0
rendering:
  font_size_pt: {{ min: 14, max: 18 }}
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: lines
  padding_px: 6
"""

# Each entry is a single rendered line; spaces inside an entry mean
# multiple words per sample, which is exactly what ``lines`` mode is for.
_SEED_LINES = "ḃeaḋ saoġal mór\nċeann an ḃoṫair\nḋuine ḟir oġa\nġloine ṁaṫair óg\nfear an tí\n"


def _setup(tmp_path: Path) -> Path:
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_LINES_RECIPE.format(font=font), encoding="utf-8")
    (tmp_path / "seed-lines.txt").write_text(_SEED_LINES, encoding="utf-8")
    return rp


# ---------------------------------------------------------------------------
# Happy path: lines-mode dispatch produces a full trainer/v1 layout
# ---------------------------------------------------------------------------


def test_render_lines_writes_trainer_v1_recognition_layout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "5",
            "--output",
            str(out),
            "--seed",
            "31",
            "--workers",
            "1",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    # Standard recognition profile layout is produced.
    assert (out / "images").is_dir()
    assert (out / LABELS_FILENAME).exists()
    assert (out / MANIFEST_FILENAME).exists()
    assert (out / SNAPSHOT_FILENAME).exists()
    assert (out / STATS_FILENAME).exists()


def test_render_lines_labels_carry_full_line_text(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each label is a full line, not a single word.

    The corpus hands the renderer space-delimited multi-word lines.
    A correctly-wired ``lines`` dispatch should render those whole
    lines and record them verbatim in ``labels.json`` — i.e. every
    label has at least one space, since every seed line has at least
    two words.
    """

    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "5",
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
    assert len(labels) == 5
    # Every label is a multi-word line drawn from the seed corpus.
    seed_set = {line for line in _SEED_LINES.splitlines() if line.strip()}
    for name, text in labels.items():
        assert (out / "images" / name).exists()
        assert " " in text, f"label {name!r} unexpectedly word-shaped: {text!r}"
        assert text in seed_set, f"label {name!r} text not from seed corpus: {text!r}"


def test_render_lines_manifest_carries_per_word_bboxes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``lines`` mode populates ``word_boxes`` on each manifest record.

    The renderer's per-word bboxes ride along in the manifest so a
    downstream consumer (preview UI, future detection-mode export)
    can recover per-word geometry without re-rendering. ``word_crops``
    mode does not write this field; it'd be empty anyway.
    """

    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "5",
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
        assert "word_boxes" in rec, f"missing word_boxes on rendered row: {rec}"
        wbs = rec["word_boxes"]
        # One word per whitespace-delimited token in the rendered line.
        expected_words = rec["text"].split()
        assert [wb["text"] for wb in wbs] == expected_words, (
            f"word texts mismatch: got {[wb['text'] for wb in wbs]!r}, expected {expected_words!r}"
        )
        for wb in wbs:
            x0, y0, x1, y1 = wb["bbox"]
            assert x1 > x0 and y1 > y0, f"degenerate word bbox: {wb}"
            # Each word bbox sits inside the sample image canvas.
            sw, sh = rec["size"]
            assert 0 <= x0 < x1 <= sw, f"word bbox escapes width: {wb} (canvas {sw}x{sh})"
            assert 0 <= y0 < y1 <= sh, f"word bbox escapes height: {wb} (canvas {sw}x{sh})"


def test_render_lines_serial_and_parallel_produce_same_labels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Workers parity holds for ``lines`` mode just like for ``word_crops``."""

    rp = _setup(tmp_path)
    serial_out = tmp_path / "serial"
    parallel_out = tmp_path / "parallel"

    rc1 = main(
        [
            "render",
            str(rp),
            "--count",
            "4",
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
            "4",
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


def test_render_lines_parallel_carries_word_boxes_through_workers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The multi-worker path must preserve per-word bboxes too.

    ``_drive_parallel`` round-trips samples through pickled PNG
    payloads; the parent rebuilds a sample shim before calling the
    writer. This regression-tests that the shim carries word_boxes.
    """

    rp = _setup(tmp_path)
    out = tmp_path / "parallel-out"

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "4",
            "--output",
            str(out),
            "--seed",
            "31",
            "--workers",
            "2",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    for line in (out / MANIFEST_FILENAME).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("status") != "rendered":
            continue
        assert rec.get("word_boxes"), (
            f"parallel-rendered row missing word_boxes: index={rec.get('index')}"
        )
        # Every word_box has a 4-tuple bbox and non-empty text.
        for wb in rec["word_boxes"]:
            assert isinstance(wb["text"], str) and wb["text"]
            assert len(wb["bbox"]) == 4
