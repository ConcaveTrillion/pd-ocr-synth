"""End-to-end tests for ``pd-ocr-synth render`` (M07).

The full pipeline — corpus → transforms → tokenize → render → degrade
→ writer — exercised against a hermetic recipe built around the
bundled Bunchló GC font.

These tests verify the trainer-consumable layout shape: the spec /
roadmap deliverables that downstream tooling expects.
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
        pytest.skip("Bundled Gaelic font not available; render test skipped.")
    return _BUNDLED_FONT


_RECIPE = """\
schema_version: 1
name: render-smoke
seed: 21
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./trainer-out
  count: 8
corpus:
  - type: local
    path: ./seed-words.txt
fonts:
  - path: {font}
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

_DEGRADE_TAIL = """\
degradation:
  - kind: jpeg
    probability: 1.0
    quality: {min: 70, max: 90}
"""

_SEED_WORDS = "\n".join(["ḃeaḋ", "ċeann", "ḋuine", "ḟear", "ġloine", "ṁaṫair"]) + "\n"


def _setup(tmp_path: Path, *, with_degrade: bool = False) -> Path:
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    body = _RECIPE.format(font=font)
    if with_degrade:
        body += _DEGRADE_TAIL
    rp.write_text(body, encoding="utf-8")
    (tmp_path / "seed-words.txt").write_text(_SEED_WORDS, encoding="utf-8")
    return rp


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_render_writes_trainer_v1_recognition_layout(
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
            "21",
            "--workers",
            "1",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    # Layout per docs/specs/08-output-format.md
    assert (out / "images").is_dir()
    assert (out / LABELS_FILENAME).exists()
    assert (out / MANIFEST_FILENAME).exists()
    assert (out / SNAPSHOT_FILENAME).exists()
    assert (out / STATS_FILENAME).exists()

    # labels.json is a JSON map (the trainer's actual contract).
    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(labels, dict)
    # Every label key matches an image on disk.
    for name in labels:
        assert (out / "images" / name).exists()

    # Manifest lines are JSON, in index order, one per attempted sample.
    manifest_lines = [
        json.loads(line)
        for line in (out / MANIFEST_FILENAME).read_text().splitlines()
        if line.strip()
    ]
    assert len(manifest_lines) == 5
    indices = [r["index"] for r in manifest_lines]
    assert indices == sorted(indices) == list(range(5))

    # Stats reports the actual planned count we asked for, not the
    # recipe's larger default.
    stats = json.loads((out / STATS_FILENAME).read_text(encoding="utf-8"))
    assert stats["samples_planned"] == 5  # planned reflects writer config (recipe.count)
    assert stats["samples_written"] + stats["samples_skipped"] == 5
    assert stats["wall_time_seconds"] >= 0.0


def test_render_refuses_nonempty_destination_without_force_or_resume(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"
    out.mkdir()
    (out / "stale.txt").write_text("from a previous run", encoding="utf-8")

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(out),
            "--seed",
            "21",
            "--workers",
            "1",
        ]
    )
    assert rc == 6  # DESTINATION_EXIT
    err = capsys.readouterr().err
    assert "not empty" in err and "--force" in err and "--resume" in err


def test_render_force_clears_destination(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"
    out.mkdir()
    (out / "leftover.png").write_bytes(b"old")

    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(out),
            "--seed",
            "21",
            "--workers",
            "1",
            "--force",
        ]
    )
    assert rc == 0, capsys.readouterr().err
    assert not (out / "leftover.png").exists()


def test_render_force_and_resume_mutually_exclusive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    rc = main(
        [
            "render",
            str(rp),
            "--output",
            str(tmp_path / "out"),
            "--force",
            "--resume",
        ]
    )
    assert rc == 2  # USAGE_EXIT
    assert "mutually exclusive" in capsys.readouterr().err


def test_render_resume_continues_existing_run(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc1 = main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(out),
            "--seed",
            "21",
            "--workers",
            "1",
        ]
    )
    assert rc1 == 0, capsys.readouterr().err
    initial_labels = set(json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8")).keys())

    # Now resume with a higher count — the prior samples should be
    # left intact and indices 3-7 added.
    rc2 = main(
        [
            "render",
            str(rp),
            "--count",
            "8",
            "--output",
            str(out),
            "--seed",
            "21",
            "--workers",
            "1",
            "--resume",
        ]
    )
    assert rc2 == 0, capsys.readouterr().err

    after_labels = set(json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8")).keys())
    # Count grew, prior labels are still present.
    assert initial_labels.issubset(after_labels)
    assert len(after_labels) >= len(initial_labels)


def test_render_resume_rejects_seed_drift(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc1 = main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(out),
            "--seed",
            "21",
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
            str(out),
            "--seed",
            "999",  # different seed => snapshot mismatch
            "--workers",
            "1",
            "--resume",
        ]
    )
    assert rc2 == 6  # DESTINATION_EXIT (snapshot mismatch family)
    assert "seed" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Dry-run
# ---------------------------------------------------------------------------


def test_render_dry_run_does_not_write(
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
            "3",
            "--output",
            str(out),
            "--seed",
            "21",
            "--dry-run",
        ]
    )
    assert rc == 0, capsys.readouterr().err
    out_capture = capsys.readouterr().out
    # Plan summary printed.
    assert "recipe:" in out_capture
    assert "count:" in out_capture
    assert "fonts present:" in out_capture
    # Nothing on disk.
    assert not out.exists() or not any(out.iterdir())


# ---------------------------------------------------------------------------
# Trainer-contract shape
# ---------------------------------------------------------------------------


def test_render_labels_json_matches_trainer_recognition_contract(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``labels.json`` is what ``pd-ocr-trainer/dataset_store.py`` consumes.

    Schema: a JSON object whose keys are PNG filenames (no path
    components) and whose values are plain strings (no nested objects).
    Every key must point at an image actually present in ``images/``.
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
            "21",
            "--workers",
            "1",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert isinstance(labels, dict)
    images_dir = out / "images"
    for name, text in labels.items():
        assert "/" not in name and "\\" not in name, f"label key has path components: {name!r}"
        assert name.endswith(".png")
        assert isinstance(text, str) and text  # non-empty label
        assert (images_dir / name).exists(), f"label {name} has no image on disk"


# ---------------------------------------------------------------------------
# Determinism: workers parity
# ---------------------------------------------------------------------------


def test_render_serial_and_parallel_produce_same_labels(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same recipe + seed + count → same labels.json regardless of workers.

    PNG byte equality is already covered by the preview tests; here we
    just need the writer-level invariant: the index → text mapping is
    independent of completion order.
    """

    rp = _setup(tmp_path)
    serial_out = tmp_path / "serial"
    parallel_out = tmp_path / "parallel"

    rc1 = main(
        [
            "render",
            str(rp),
            "--count",
            "6",
            "--output",
            str(serial_out),
            "--seed",
            "21",
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
            "6",
            "--output",
            str(parallel_out),
            "--seed",
            "21",
            "--workers",
            "4",
        ]
    )
    assert rc2 == 0, capsys.readouterr().err

    serial_labels = json.loads((serial_out / LABELS_FILENAME).read_text())
    parallel_labels = json.loads((parallel_out / LABELS_FILENAME).read_text())
    assert serial_labels == parallel_labels


# ---------------------------------------------------------------------------
# Spec 07 ``name`` flow: recipe → manifest
# ---------------------------------------------------------------------------


_NAMED_DEGRADE_TAIL = """\
degradation:
  - kind: jpeg
    name: light_jpeg
    probability: 1.0
    quality: {min: 70, max: 90}
"""


def _setup_named_degrade(tmp_path: Path) -> Path:
    """Recipe with a named degradation stage (spec 07 ``name`` key)."""

    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    body = _RECIPE.format(font=font) + _NAMED_DEGRADE_TAIL
    rp.write_text(body, encoding="utf-8")
    (tmp_path / "seed-words.txt").write_text(_SEED_WORDS, encoding="utf-8")
    return rp


def _read_manifest_rendered(out: Path) -> list[dict]:
    from pd_ocr_synth.output.recognition import MANIFEST_FILENAME as _MF

    rows = [json.loads(line) for line in (out / _MF).read_text().splitlines() if line.strip()]
    return [r for r in rows if r.get("status") == "rendered"]


def test_render_flows_stage_name_into_manifest_serial(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec 07 lists ``name`` as "Optional label recorded in the manifest".

    Producer (render/run.py) historically only emitted ``kind`` +
    ``probability`` in ``degradations_applied``; the consumer
    (``_flatten_degradations`` in publish/recognition.py) defensively
    fell back to ``item.get("name")`` because the receiver was authored
    expecting it. This test pins the producer-side fix: ``name`` rides
    through to the manifest when the recipe declares it.
    """

    rp = _setup_named_degrade(tmp_path)
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
            "21",
            "--workers",
            "1",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    rendered = _read_manifest_rendered(out)
    assert rendered, "expected at least one rendered manifest row"
    for row in rendered:
        applied = row["degradations_applied"]
        assert applied, "rendered row should have degradations_applied"
        for stage in applied:
            assert stage["kind"] == "jpeg"
            assert stage["name"] == "light_jpeg"
            assert stage["probability"] == 1.0


def test_render_flows_stage_name_into_manifest_parallel(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The worker-payload round-trip (parallel path) preserves ``name``.

    The parallel pickle/unpickle hop in ``_worker_render`` ships the
    full ``applied`` list verbatim; this test guards against any future
    payload schema that drops keys it doesn't recognise.
    """

    rp = _setup_named_degrade(tmp_path)
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
            "21",
            "--workers",
            "2",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    rendered = _read_manifest_rendered(out)
    assert rendered
    for row in rendered:
        for stage in row["degradations_applied"]:
            assert stage["kind"] == "jpeg"
            assert stage["name"] == "light_jpeg"


def test_render_omits_stage_name_when_recipe_does_not_declare_it(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Stages without ``name`` keep their existing 2-key shape.

    We don't synthesise a fallback name from ``kind`` — the spec calls
    ``name`` "Optional", so absence stays absent. Guards against a
    well-meaning future change that auto-derives ``name=kind`` and
    then drifts the consumer.
    """

    rp = _setup(tmp_path, with_degrade=True)  # _DEGRADE_TAIL has no name
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
            "21",
            "--workers",
            "1",
        ]
    )
    assert rc == 0, capsys.readouterr().err

    rendered = _read_manifest_rendered(out)
    assert rendered
    for row in rendered:
        for stage in row["degradations_applied"]:
            assert stage["kind"] == "jpeg"
            assert "name" not in stage, (
                "stage without recipe-declared ``name`` should not synthesise one"
            )
