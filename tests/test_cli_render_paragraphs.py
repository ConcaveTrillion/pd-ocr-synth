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


def test_render_paragraphs_serial_and_parallel_produce_same_pngs_and_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Detection-mode render path round-trips bit-identically across worker counts.

    Companion to the labels-only parity test above. Detection mode
    exercises the heaviest worker payload — ``word_boxes`` *and*
    ``line_boxes`` *and* ``paragraph_boxes`` all need to round-trip
    through the worker → parent boundary without dropping or reshaping
    any field. PNG byte equality on ``images/page_*.png`` plus
    manifest equality plus stats equality lock that contract: any
    future change that subtly truncates a payload field — say, drops
    ``paragraph_boxes`` or stops carrying a colour channel — will fail
    here rather than silently diverging the worker=N dataset from the
    worker=1 dataset.
    """

    rp = _setup(tmp_path)
    serial_out = tmp_path / "serial-bytes"
    parallel_out = tmp_path / "parallel-bytes"

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

    # PNG byte equality per index — detection mode emits ``page_*.png``.
    serial_pngs = {p.name: p for p in (serial_out / "images").glob(f"{PAGE_PREFIX}*.png")}
    parallel_pngs = {p.name: p for p in (parallel_out / "images").glob(f"{PAGE_PREFIX}*.png")}
    assert serial_pngs.keys() == parallel_pngs.keys()
    assert len(serial_pngs) == 3
    for name in serial_pngs:
        assert serial_pngs[name].read_bytes() == parallel_pngs[name].read_bytes(), (
            f"detection render path png mismatch at {name} between workers=1 and workers=2"
        )

    # Manifest equality (sorted by index for completion-order
    # independence). Detection manifest carries ``line_count`` and
    # ``word_count`` keys so this also catches a subtle line/word-box
    # round-trip regression.
    serial_records = [
        json.loads(line)
        for line in (serial_out / MANIFEST_FILENAME).read_text().splitlines()
        if line
    ]
    parallel_records = [
        json.loads(line)
        for line in (parallel_out / MANIFEST_FILENAME).read_text().splitlines()
        if line
    ]
    serial_sorted = sorted(serial_records, key=lambda r: r["index"])
    parallel_sorted = sorted(parallel_records, key=lambda r: r["index"])
    assert serial_sorted == parallel_sorted

    # Stats equality, modulo the wall-clock field which is excluded
    # from the determinism contract by design (see ``run_recipe``
    # docstring: "audit log is not part of the determinism contract").
    serial_stats = json.loads((serial_out / STATS_FILENAME).read_text())
    parallel_stats = json.loads((parallel_out / STATS_FILENAME).read_text())
    serial_stats.pop("wall_time_seconds", None)
    parallel_stats.pop("wall_time_seconds", None)
    assert serial_stats == parallel_stats


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


# ---------------------------------------------------------------------------
# M09 wrap-fitter wiring
# ---------------------------------------------------------------------------


# Recipe with ``layout.max_width_px`` set: the wrap-fitter should split
# a long single-line corpus token across multiple lines so each fits
# the budget. Single-paragraph (no blank line) so the tokenizer hands
# the renderer a single multi-word token.
_WRAP_RECIPE = """\
schema_version: 1
name: render-paragraphs-wrap
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
  mode: paragraphs
  padding_px: 4
  line_spacing: 1.2
  max_width_px: {max_width}
"""

# Long single-line paragraph (no embedded newlines) so the only path
# from corpus token to multiple lines is the wrap-fitter. Eight short
# Gaelic words; at a tight pixel budget they have to wrap.
_WRAP_SEED = "ḃeaḋ saoġal mór ċeann an ḃoṫair agus ḋuine\n"


def _setup_wrap(tmp_path: Path, max_width: int) -> Path:
    font = _require_font()
    tmp_path.mkdir(parents=True, exist_ok=True)
    rp = tmp_path / "recipe.yaml"
    rp.write_text(
        _WRAP_RECIPE.format(font=font, max_width=max_width),
        encoding="utf-8",
    )
    (tmp_path / "seed.txt").write_text(_WRAP_SEED, encoding="utf-8")
    return rp


def _run_wrap(tmp_path: Path, *, max_width: int, out_name: str = "trainer-out") -> Path:
    rp = _setup_wrap(tmp_path, max_width)
    out = tmp_path / out_name
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
    return out


def test_wrap_fitter_splits_long_token_across_multiple_lines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A long single-line corpus token wraps to multiple lines when
    ``layout.max_width_px`` is tight.

    Without the wrap-fitter wired into ``_split_paragraph_into_lines``,
    the long token would land as a single line whose painted width
    blows past the budget. With the wrap-fitter wired, the renderer
    sees ``len(lines) > 1`` and the labels carry one polygon per
    wrapped line.
    """

    _ = capsys
    out = _run_wrap(tmp_path, max_width=180)

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert len(labels) == 1
    (entry,) = labels.values()
    lines = entry["lines"]
    assert len(lines) > 1, f"expected wrap to multiple lines, got: {[ln['text'] for ln in lines]}"
    # Each wrapped line is non-empty and the joined text recovers the
    # original word stream (whitespace-normalized).
    joined = " ".join(ln["text"] for ln in lines)
    assert joined.split() == _WRAP_SEED.split()


def test_wrap_fitter_grows_image_height_with_more_lines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A tighter ``max_width_px`` produces more wrapped lines and a
    taller image.

    Sanity check that the wrap-fitter is actually feeding line count
    into the renderer (rather than e.g. silently being a no-op):
    halving the budget should increase the page height.
    """

    _ = capsys
    wide = _run_wrap(tmp_path / "wide", max_width=400, out_name="out")
    narrow = _run_wrap(tmp_path / "narrow", max_width=160, out_name="out")

    wide_labels = json.loads((wide / LABELS_FILENAME).read_text(encoding="utf-8"))
    narrow_labels = json.loads((narrow / LABELS_FILENAME).read_text(encoding="utf-8"))
    (wide_entry,) = wide_labels.values()
    (narrow_entry,) = narrow_labels.values()

    wide_lines = len(wide_entry["lines"])
    narrow_lines = len(narrow_entry["lines"])
    assert narrow_lines > wide_lines, (
        f"narrow budget ({narrow_lines} lines) did not exceed wide budget ({wide_lines} lines)"
    )

    _, wide_h = wide_entry["img_dimensions"]
    _, narrow_h = narrow_entry["img_dimensions"]
    assert narrow_h > wide_h, (
        f"narrow page height ({narrow_h}) did not exceed wide page height ({wide_h})"
    )


def test_wrap_fitter_lines_stay_within_page_canvas(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every wrapped line's polygon stays inside the page canvas.

    The wrap budget is on shaped width, not painted bbox; tiny
    overshoot is possible if the renderer's padding interacts oddly
    with cluster bboxes. This test pins the invariant the trainer
    actually depends on: polygons fit the page.
    """

    _ = capsys
    out = _run_wrap(tmp_path, max_width=200)
    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    (entry,) = labels.values()
    w, h = entry["img_dimensions"]
    for poly in entry["polygons"]:
        for x, y in poly:
            assert 0 <= x <= w, f"polygon x={x} out of bounds [0, {w}]"
            assert 0 <= y <= h, f"polygon y={y} out of bounds [0, {h}]"


def test_wrap_fitter_is_deterministic(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same recipe + same seed → identical wrapped lines across runs.

    Pre-sampling the paragraph style and threading it through
    ``fit_lines`` + ``render_paragraph`` must not change the
    determinism contract: re-running the same recipe yields the same
    on-disk labels.
    """

    _ = capsys
    out_a = _run_wrap(tmp_path / "a", max_width=180, out_name="out")
    out_b = _run_wrap(tmp_path / "b", max_width=180, out_name="out")
    labels_a = json.loads((out_a / LABELS_FILENAME).read_text(encoding="utf-8"))
    labels_b = json.loads((out_b / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert labels_a == labels_b


# Recipe with embedded newlines in the corpus token + a wrap budget.
# The wrap-fitter should preserve the hard breaks while wrapping any
# still-too-long chunks. Two-line corpus token: line 1 fits, line 2
# is long enough to need wrapping.
_HARDBREAK_RECIPE = _WRAP_RECIPE
_HARDBREAK_SEED = "ḃeaḋ saoġal\nċeann an ḃoṫair agus ḋuine ṁór\n"


def test_wrap_fitter_preserves_hard_newlines(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Hard newlines in the corpus token are preserved across wrapping.

    Legacy behavior: a paragraph token that already carries line
    structure (e.g. poetry) keeps its hard breaks; ``fit_lines``
    wraps each chunk independently and concatenates results. Verify
    that splitting the line that *already fits* doesn't happen — the
    fitter should not merge a short hard-broken line with the next
    one.
    """

    _ = capsys
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    rp.write_text(
        _HARDBREAK_RECIPE.format(font=font, max_width=400),
        encoding="utf-8",
    )
    (tmp_path / "seed.txt").write_text(_HARDBREAK_SEED, encoding="utf-8")
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
    line_texts = [ln["text"] for ln in entry["lines"]]
    # Hard-broken line 1 stays on its own and isn't merged with line 2.
    assert "ḃeaḋ saoġal" in line_texts, (
        f"expected first hard-broken line to be preserved; got {line_texts}"
    )
    # The joined text recovers the original word stream.
    joined = " ".join(line_texts)
    assert joined.split() == _HARDBREAK_SEED.split()


# Recipe **without** ``layout.max_width_px``: the legacy ``\n``-only
# split path is exercised. A single-line corpus token becomes a one-
# element line list — the wrap-fitter is intentionally not engaged.
_NOWRAP_RECIPE = """\
schema_version: 1
name: render-paragraphs-nowrap
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
  mode: paragraphs
  padding_px: 4
  line_spacing: 1.2
"""


def test_wrap_fitter_disabled_when_max_width_px_unset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No ``max_width_px`` → legacy ``\\n``-only split path.

    A single-line corpus token must remain a single rendered line —
    ``fit_lines`` is never engaged because the recipe author opted
    out by leaving the budget unset.
    """

    _ = capsys
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_NOWRAP_RECIPE.format(font=font), encoding="utf-8")
    (tmp_path / "seed.txt").write_text(_WRAP_SEED, encoding="utf-8")
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
    assert len(entry["lines"]) == 1, (
        f"expected single-line render with no max_width_px, got "
        f"{[ln['text'] for ln in entry['lines']]}"
    )
