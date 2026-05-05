"""Tests for ``pd-ocr-synth preview`` (M05 deliverable).

Exercises the CLI happy-path against a hermetic tmp recipe that
points at the bundled Bunchló GC font + a tiny local seed-words
corpus. Verifies the directory layout the roadmap requires:

```
<output>/
    images/         # PNG files
    manifest.jsonl  # one record per sample
    stats.json      # summary counters
```
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pd_ocr_synth.cli import main

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; preview test skipped.")
    return _BUNDLED_FONT


_RECIPE_TEMPLATE = """\
schema_version: 1
name: preview-smoke
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 1
corpus:
  - type: local
    path: ./seed-words.txt
fonts:
  - path: {font_path}
    weight: 1.0
rendering:
  font_size_pt: {{ min: 14, max: 18 }}
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: word_crops
  padding_px: 4
"""

_DEGRADE_RECIPE_TEMPLATE = (
    _RECIPE_TEMPLATE
    + """\
degradation:
  - kind: skew
    probability: 1.0
    angle_deg: {{ min: -2, max: 2 }}
  - kind: blur
    probability: 1.0
    filter: gaussian
    sigma: {{ min: 0.5, max: 1.0 }}
  - kind: jpeg
    probability: 1.0
    quality: {{ min: 70, max: 90 }}
"""
)

_SEED_WORDS = "\n".join(["ḃeaḋ", "ċeann", "ḋuine", "ḟear", "ġloine", "ṁaṫair"]) + "\n"


def _setup(tmp_path: Path, *, with_degrade: bool = False) -> Path:
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    template = _DEGRADE_RECIPE_TEMPLATE if with_degrade else _RECIPE_TEMPLATE
    rp.write_text(template.format(font_path=font), encoding="utf-8")
    (tmp_path / "seed-words.txt").write_text(_SEED_WORDS, encoding="utf-8")
    return rp


def test_preview_writes_images_manifest_and_stats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    output = tmp_path / "preview-out"

    rc = main(
        [
            "preview",
            str(rp),
            "--count",
            "3",
            "--output",
            str(output),
            "--seed",
            "7",
        ]
    )
    assert rc == 0, f"preview exited with {rc}: {capsys.readouterr().err}"

    images = sorted((output / "images").glob("*.png"))
    assert len(images) == 3, f"expected 3 PNGs, got {len(images)}"
    for image_path in images:
        assert image_path.stat().st_size > 100  # non-trivial PNG bytes

    manifest_path = output / "manifest.jsonl"
    assert manifest_path.exists()
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    assert len(records) == 3
    for record in records:
        assert record["status"] == "rendered"
        assert "image" in record and record["image"].startswith("images/")
        assert record["text"]
        assert "font_path" in record and "font_size_pt" in record and "dpi" in record
        assert "bbox" in record and len(record["bbox"]) == 4
        assert "glyph_runs" in record and isinstance(record["glyph_runs"], list)

    stats_path = output / "stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["recipe"] == "preview-smoke"
    assert stats["seed"] == 7
    assert stats["count"] == 3
    assert stats["rendered"] == 3
    assert stats["skipped"] == 0


def test_preview_rejects_zero_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    rc = main(
        [
            "preview",
            str(rp),
            "--count",
            "0",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2  # USAGE_EXIT
    assert "must be positive" in capsys.readouterr().err


def test_preview_default_count_is_50_helper() -> None:
    """The preview module's default count is the spec's ``50``."""

    from pd_ocr_synth.render.preview import DEFAULT_PREVIEW_COUNT

    assert DEFAULT_PREVIEW_COUNT == 50


# ---------------------------------------------------------------------------
# Parallelism / worker count
# ---------------------------------------------------------------------------


def test_resolve_workers_default_is_sane() -> None:
    """Auto-resolved worker count must respect the documented bounds.

    Per the M05 follow-up: default = ``max(1, min(cpu_count - 1, 8))``.
    Asserting the range rather than a single value keeps the test
    portable across dev boxes (1-core CI VMs through 32-core
    workstations).
    """

    import os

    from pd_ocr_synth.render.preview import resolve_workers

    auto = resolve_workers(None)
    cpu = os.cpu_count() or 1
    assert auto >= 1
    assert auto <= min(max(cpu, 1), 8)
    if cpu >= 3:
        # Multi-core: we leave one core free.
        assert auto <= cpu - 1


def test_resolve_workers_explicit_override_is_clamped_positive() -> None:
    from pd_ocr_synth.render.preview import resolve_workers

    assert resolve_workers(1) == 1
    assert resolve_workers(2) == 2
    assert resolve_workers(99) == 99  # explicit override: trust the user
    # Defensive clamp — we guard non-positive at the CLI layer too,
    # but the helper itself should never return < 1.
    assert resolve_workers(0) == 1


def test_preview_parallel_matches_serial_byte_for_byte(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The load-bearing determinism test for parallelism.

    Same recipe + seed + count, run once with ``--workers 1`` and
    once with ``--workers 4`` against fresh output directories. PNG
    bytes for each ``images/<seed>_<index>.png`` must match exactly,
    and manifest lines must match when sorted by index. Stats too.
    """

    rp = _setup(tmp_path)
    serial_out = tmp_path / "serial-out"
    parallel_out = tmp_path / "parallel-out"

    rc1 = main(
        [
            "preview",
            str(rp),
            "--count",
            "8",
            "--output",
            str(serial_out),
            "--seed",
            "13",
            "--workers",
            "1",
        ]
    )
    assert rc1 == 0, capsys.readouterr().err

    rc2 = main(
        [
            "preview",
            str(rp),
            "--count",
            "8",
            "--output",
            str(parallel_out),
            "--seed",
            "13",
            "--workers",
            "4",
        ]
    )
    assert rc2 == 0, capsys.readouterr().err

    # Same set of PNG filenames (same seed + same indices).
    serial_pngs = {p.name: p for p in (serial_out / "images").glob("*.png")}
    parallel_pngs = {p.name: p for p in (parallel_out / "images").glob("*.png")}
    assert serial_pngs.keys() == parallel_pngs.keys()
    assert len(serial_pngs) == 8

    # Byte-equal images per index.
    for name in serial_pngs:
        assert serial_pngs[name].read_bytes() == parallel_pngs[name].read_bytes(), (
            f"png mismatch at {name} between workers=1 and workers=4"
        )

    # Manifest equality after sorting by index. Workers may finish
    # in any order, but the writer assembles the manifest in sample-
    # index order — so the lines should already be in lockstep.
    serial_records = [
        json.loads(line)
        for line in (serial_out / "manifest.jsonl").read_text().splitlines()
        if line
    ]
    parallel_records = [
        json.loads(line)
        for line in (parallel_out / "manifest.jsonl").read_text().splitlines()
        if line
    ]
    serial_sorted = sorted(serial_records, key=lambda r: r["index"])
    parallel_sorted = sorted(parallel_records, key=lambda r: r["index"])
    assert serial_sorted == parallel_sorted

    # Stats must match too — same rendered/skipped counts.
    serial_stats = json.loads((serial_out / "stats.json").read_text())
    parallel_stats = json.loads((parallel_out / "stats.json").read_text())
    # ``output_dir`` differs by construction; everything else must match.
    serial_stats.pop("output_dir")
    parallel_stats.pop("output_dir")
    assert serial_stats == parallel_stats


# ---------------------------------------------------------------------------
# Degradation integration (M06)
# ---------------------------------------------------------------------------


def test_preview_applies_degradation_by_default(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A recipe with degradation stages must produce different bytes than
    the same recipe re-run with --no-degrade.
    """

    rp = _setup(tmp_path, with_degrade=True)

    out_with = tmp_path / "with"
    out_without = tmp_path / "without"

    rc1 = main(
        [
            "preview",
            str(rp),
            "--count",
            "3",
            "--seed",
            "7",
            "--workers",
            "1",
            "--output",
            str(out_with),
        ]
    )
    assert rc1 == 0, capsys.readouterr().err

    rc2 = main(
        [
            "preview",
            str(rp),
            "--count",
            "3",
            "--seed",
            "7",
            "--workers",
            "1",
            "--no-degrade",
            "--output",
            str(out_without),
        ]
    )
    assert rc2 == 0, capsys.readouterr().err

    with_pngs = sorted((out_with / "images").glob("*.png"))
    without_pngs = sorted((out_without / "images").glob("*.png"))
    assert len(with_pngs) == len(without_pngs) == 3

    # Same filenames (deterministic naming), different bytes (degradation
    # changed at least the JPEG-quantized output) for every sample.
    for w, wo in zip(with_pngs, without_pngs, strict=True):
        assert w.name == wo.name
        assert w.read_bytes() != wo.read_bytes(), f"degradation produced no change for {w.name}"


def test_preview_degradation_is_deterministic_across_workers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """With degradation on, --workers 1 and --workers 4 must still match
    byte-for-byte. The degradation pipeline draws from the same
    per-sample-branched RNG that the renderer reseeds before each call.
    """

    rp = _setup(tmp_path, with_degrade=True)
    serial_out = tmp_path / "serial"
    parallel_out = tmp_path / "parallel"

    rc1 = main(
        [
            "preview",
            str(rp),
            "--count",
            "6",
            "--seed",
            "21",
            "--workers",
            "1",
            "--output",
            str(serial_out),
        ]
    )
    assert rc1 == 0, capsys.readouterr().err
    rc2 = main(
        [
            "preview",
            str(rp),
            "--count",
            "6",
            "--seed",
            "21",
            "--workers",
            "4",
            "--output",
            str(parallel_out),
        ]
    )
    assert rc2 == 0, capsys.readouterr().err

    serial_pngs = {p.name: p for p in (serial_out / "images").glob("*.png")}
    parallel_pngs = {p.name: p for p in (parallel_out / "images").glob("*.png")}
    assert serial_pngs.keys() == parallel_pngs.keys()
    for name in serial_pngs:
        assert serial_pngs[name].read_bytes() == parallel_pngs[name].read_bytes(), (
            f"degraded png mismatch at {name} between workers=1 and workers=4"
        )
