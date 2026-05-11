"""End-to-end tests for ``pd-ocr-synth render`` with ``layout.mode = pages``.

Exercises the M09 detection-mode pages dispatch chunk: a recipe with
``output.mode = detection`` + ``layout.mode = pages`` should walk
``run_recipe`` through ``render_page`` and into :class:`DetectionWriter`,
producing the ``pd-ocr-trainer/v1`` detection layout (``images/page_*.png``
+ ``labels.json``) for **multi-paragraph** page samples.

Mirrors :mod:`tests.test_cli_render_paragraphs` for paragraphs mode.
The interesting differences from paragraphs mode are:

- The tokenizer now splits on a triple-blank-line **page** boundary,
  not the paragraph boundary; one page token can carry multiple
  blank-line-separated paragraphs.
- The renderer composes those paragraphs vertically with a recipe-
  driven inter-paragraph gap (``layout.paragraph_spacing``).
- ``DetectionWriter`` flattens lines across paragraphs in reading
  order — the writer doesn't special-case pages vs. paragraphs.

These tests skip when the bundled Bunchló GC font isn't present on
disk (e.g. fresh checkout that hasn't run
``./scripts/fetch-fonts-gaelic.sh``).
"""

from __future__ import annotations

import itertools
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
        pytest.skip("Bundled Gaelic font not available; pages render test skipped.")
    return _BUNDLED_FONT


# Recipe template: detection + pages with a paragraph_spacing gap.
# Padding is small so the page's tight crop is easy to reason about
# in assertions; the wrap budget is generous so single-line
# paragraphs stay single-line and we can pin counts.
_PAGES_RECIPE = """\
schema_version: 1
name: render-pages-smoke
seed: 31
output:
  format: pd-ocr-trainer/v1
  mode: detection
  destination: ./trainer-out
  count: 4
corpus:
  - type: local
    path: ./seed-pages.txt
fonts:
  - path: {font}
    weight: 1.0
rendering:
  font_size_pt: 16
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: pages
  padding_px: 6
  line_spacing: 1.2
  paragraph_spacing: 1.0
"""

# Two page-sized chunks separated by a triple-blank-line boundary.
# Each page has two paragraphs; each paragraph has two short lines.
# Lines are short enough to skip the wrap-fitter (no max_width_px set).
_SEED_PAGES = (
    "ḃeaḋ saoġal mór\nċeann an ḃoṫair\n\n"
    "ḋuine ḟir oġa\nġloine ṁaṫair óg\n"
    "\n\n\n"  # triple-blank-line page boundary
    "ḋuine ḟir oġa\nġloine ṁaṫair óg\n\n"
    "ḃeaḋ saoġal mór\nċeann an ḃoṫair\n"
)


def _setup(tmp_path: Path) -> Path:
    font = _require_font()
    tmp_path.mkdir(parents=True, exist_ok=True)
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_PAGES_RECIPE.format(font=font), encoding="utf-8")
    (tmp_path / "seed-pages.txt").write_text(_SEED_PAGES, encoding="utf-8")
    return rp


def _all_seed_lines() -> set[str]:
    """The set of non-empty rendered line texts across both pages."""
    return {ln for ln in _SEED_PAGES.splitlines() if ln.strip()}


# ---------------------------------------------------------------------------
# Happy path: pages-mode dispatch produces a full detection profile
# ---------------------------------------------------------------------------


def test_render_pages_writes_trainer_v1_detection_layout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Smoke: detection profile sidecars all land alongside the images.

    A pages-mode render walks the same writer dispatch as paragraphs,
    so this test mirrors :func:`test_render_paragraphs_writes_trainer_v1_detection_layout`.
    The point of running it for pages is to assert the dispatch
    *reaches* the writer — without the wiring, ``run_recipe`` would
    raise ``RenderError`` up front for ``layout.mode=pages``.
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

    assert (out / "images").is_dir()
    assert (out / LABELS_FILENAME).exists()
    assert (out / MANIFEST_FILENAME).exists()
    assert (out / SNAPSHOT_FILENAME).exists()
    assert (out / STATS_FILENAME).exists()

    images = sorted((out / "images").glob("*.png"))
    assert images, "expected at least one page image"
    for img in images:
        assert img.name.startswith(PAGE_PREFIX), f"unexpected image filename: {img.name}"


def test_render_pages_labels_carry_per_line_gt_across_paragraphs(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``labels.json`` carries lines from every paragraph on the page.

    A page with two paragraphs of two lines each must surface 4
    line entries (writer flattens across paragraphs in reading
    order). All line texts must come from the seed corpus.
    """

    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "2",
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
    assert len(labels) == 2

    seed_lines = _all_seed_lines()
    for name, entry in labels.items():
        assert (out / "images" / name).exists()
        w, h = entry["img_dimensions"]
        assert w > 0 and h > 0
        assert "img_hash" in entry and len(entry["img_hash"]) == 64

        polygons = entry["polygons"]
        # Both seed pages have 4 lines (2 paragraphs * 2 lines).
        assert len(polygons) == 4, f"expected 4 line polygons per page, got {len(polygons)}"
        for poly in polygons:
            assert len(poly) == 4
            for pt in poly:
                assert len(pt) == 2

        lines = entry["lines"]
        assert len(lines) == len(polygons)
        for line in lines:
            assert line["text"] in seed_lines, f"line text not from seed corpus: {line['text']!r}"
            x0, y0, x1, y1 = line["bbox"]
            assert x1 > x0 and y1 > y0
            assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
            words = line.get("words") or []
            assert words, f"line {line['text']!r} has no words"


def test_render_pages_lines_appear_in_reading_order(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Per-paragraph lines stack vertically in input order across paragraphs.

    Paragraph 1's lines must sit above paragraph 2's lines: the
    minimum y of paragraph 2's first line must exceed paragraph 1's
    last line max-y. Without correct cursor advancement in
    ``render_page``, the lines would overlap and this invariant
    would fail.
    """

    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "1",
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
    (entry,) = labels.values()
    lines = entry["lines"]
    # 4 lines = 2 paragraphs * 2 lines each. Sequential y-coordinates
    # must strictly increase across the page.
    ys = [(ln["bbox"][1], ln["bbox"][3]) for ln in lines]
    for prev, nxt in itertools.pairwise(ys):
        _prev_top, prev_bottom = prev
        nxt_top, _nxt_bottom = nxt
        assert nxt_top >= prev_bottom - 1, (
            f"line at y={nxt_top} overlaps previous line ending at y={prev_bottom}"
        )


def test_render_pages_manifest_carries_line_and_word_counts(
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
            "2",
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
        # Each seed page has 4 lines; if the renderer is composing
        # paragraphs correctly the n_lines should be 4 (no
        # silent drops, no double-counting).
        assert rec["n_lines"] == 4, f"expected 4 lines per page, got {rec}"
        assert rec["n_words"] >= rec["n_lines"]
        assert rec["image"].startswith(f"images/{PAGE_PREFIX}")


def test_render_pages_serial_and_parallel_produce_same_labels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Worker count is not load-bearing on the detection labels.

    Verifies that ``paragraph_boxes`` (introduced for pages mode)
    round-trips cleanly through the parallel-worker payload — the
    parent rebuilds a sample shim with the right shape regardless of
    serial vs parallel dispatch.
    """

    rp = _setup(tmp_path)
    serial_out = tmp_path / "serial"
    parallel_out = tmp_path / "parallel"

    rc1 = main(
        [
            "render",
            str(rp),
            "--count",
            "2",
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
            "2",
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


def test_render_pages_is_deterministic_per_seed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same recipe + same seed → identical labels across runs.

    Pre-sampling ``PageStyle`` once and threading it through both the
    paragraph splitter and ``render_page`` must not alter the
    determinism contract.
    """

    rp_a = _setup(tmp_path / "a")
    rp_b = _setup(tmp_path / "b")
    out_a = tmp_path / "out-a"
    out_b = tmp_path / "out-b"

    for rp, out in [(rp_a, out_a), (rp_b, out_b)]:
        rc = main(
            [
                "render",
                str(rp),
                "--count",
                "2",
                "--output",
                str(out),
                "--seed",
                "31",
                "--workers",
                "1",
            ]
        )
        assert rc == 0, capsys.readouterr().err

    labels_a = json.loads((out_a / LABELS_FILENAME).read_text())
    labels_b = json.loads((out_b / LABELS_FILENAME).read_text())
    assert labels_a == labels_b


# ---------------------------------------------------------------------------
# Wrap-fitter inside a pages-mode page
# ---------------------------------------------------------------------------


_PAGES_WRAP_RECIPE = """\
schema_version: 1
name: render-pages-wrap
seed: 31
output:
  format: pd-ocr-trainer/v1
  mode: detection
  destination: ./trainer-out
  count: 1
corpus:
  - type: local
    path: ./seed.txt
fonts:
  - path: {font}
    weight: 1.0
rendering:
  font_size_pt: 16
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: pages
  padding_px: 4
  line_spacing: 1.2
  paragraph_spacing: 0.5
  max_width_px: {max_width}
"""

# Two paragraphs (blank-line-separated) inside a single page (no
# triple newline). Each paragraph is a long single line that needs
# wrapping at a tight budget.
_PAGES_WRAP_SEED = (
    "ḃeaḋ saoġal mór ċeann an ḃoṫair agus ḋuine\n\nċeann an ḃoṫair agus ḋuine ṁór ḃeaḋ saoġal\n"
)


def test_render_pages_wrap_fitter_splits_inside_each_paragraph(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Each inner paragraph wraps independently against ``max_width_px``.

    With two long paragraphs and a tight budget, the page should
    surface ``> 2`` lines: each paragraph wraps to multiple lines.
    The total line count is at minimum 4 (2 paragraphs x 2 lines)
    but typically more. The joined word stream must recover every
    seed word in reading order.
    """

    _ = capsys
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    rp.write_text(
        _PAGES_WRAP_RECIPE.format(font=font, max_width=200),
        encoding="utf-8",
    )
    (tmp_path / "seed.txt").write_text(_PAGES_WRAP_SEED, encoding="utf-8")
    out = tmp_path / "trainer-out"
    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "1",
            "--output",
            str(out),
            "--seed",
            "31",
            "--workers",
            "1",
        ]
    )
    assert rc == 0

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    (entry,) = labels.values()
    lines = entry["lines"]
    assert len(lines) >= 4, (
        f"expected >=4 wrapped lines (2 paragraphs * >=2 wrapped lines), "
        f"got {[ln['text'] for ln in lines]}"
    )
    joined = " ".join(ln["text"] for ln in lines)
    assert joined.split() == _PAGES_WRAP_SEED.split()
