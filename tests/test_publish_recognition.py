"""Unit tests for the recognition staging-dir builder (M08).

Covers the format conversion from the local recognition layout (per
``docs/specs/08-output-format.md``) to the HF imagefolder layout (per
``docs/specs/10-publishing.md``). Pure file-IO; no network, no HF SDK.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from pd_ocr_synth.publish import (
    DATA_DIRNAME,
    METADATA_FILENAME,
    build_recognition_staging,
)
from pd_ocr_synth.publish.recognition import StagingError

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_local_output(
    local: Path,
    *,
    samples: list[dict[str, Any]],
    write_snapshot: bool = True,
    write_manifest: bool = True,
) -> None:
    """Materialize a local recognition output dir.

    Each sample dict accepts: ``index``, ``text``, ``font_name``,
    ``font_size_pt``, ``degradations`` (list of dicts or strings),
    ``corpus`` (dict or string), ``status`` ("rendered" or "skipped",
    default "rendered"), ``skip_reason``, ``write_image`` (bool,
    default True for rendered).
    """

    images = local / "images"
    images.mkdir(parents=True, exist_ok=True)

    labels: dict[str, str] = {}
    manifest_lines: list[str] = []

    for sample in samples:
        idx = int(sample["index"])
        name = f"{idx:07d}.png"
        status = sample.get("status", "rendered")
        if status == "rendered":
            if sample.get("write_image", True):
                img = Image.new("RGB", (8, 8), color=(200, 200, 200))
                img.save(images / name, format="PNG")
            labels[name] = sample["text"]
            record: dict[str, Any] = {
                "index": idx,
                "id": Path(name).stem,
                "image": f"images/{name}",
                "text": sample["text"],
                "status": "rendered",
            }
            font_name = sample.get("font_name")
            if font_name is not None:
                record["font"] = {
                    "name": font_name,
                    "path": f"/abs/{font_name}",
                    "size_pt": float(sample.get("font_size_pt", 14.0)),
                }
            degr = sample.get("degradations")
            if degr is not None:
                record["degradations_applied"] = degr
            corpus = sample.get("corpus")
            if corpus is not None:
                record["corpus"] = corpus
            manifest_lines.append(json.dumps(record))
        else:
            record = {
                "index": idx,
                "id": Path(name).stem,
                "status": "skipped",
                "reason": sample.get("skip_reason", "unknown"),
            }
            manifest_lines.append(json.dumps(record))

    (local / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if write_manifest:
        (local / "manifest.jsonl").write_text(
            "\n".join(manifest_lines) + ("\n" if manifest_lines else ""),
            encoding="utf-8",
        )
    if write_snapshot:
        (local / "recipe.snapshot.yaml").write_text(
            "tool_version: 0.0.0\nseed: 5\n",
            encoding="utf-8",
        )
    (local / "stats.json").write_text("{}\n", encoding="utf-8")


def _read_metadata(staging: Path) -> list[dict[str, Any]]:
    text = (staging / METADATA_FILENAME).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_staging_emits_data_dir_and_metadata_jsonl(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[
            {"index": 0, "text": "Séadna", "font_name": "bungc.otf"},
            {"index": 1, "text": "agus", "font_name": "seangc.otf"},
        ],
    )
    staging = tmp_path / "staging"

    result = build_recognition_staging(local, staging)

    assert (staging / DATA_DIRNAME / "0000000.png").is_file()
    assert (staging / DATA_DIRNAME / "0000001.png").is_file()
    assert (staging / METADATA_FILENAME).is_file()
    assert result.images_copied == 2
    assert result.rows_written == 2


def test_staging_metadata_row_carries_flat_provenance(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[
            {
                "index": 0,
                "text": "Séadna",
                "font_name": "bungc.otf",
                "font_size_pt": 14.0,
                "degradations": [
                    {"kind": "skew", "params": {"angle_deg": -1.2}},
                    {"kind": "paper_texture"},
                    {"kind": "jpeg"},
                ],
                "corpus": {"provider": "wikisource", "key": "ga:Séadna", "offset": 1024},
            },
        ],
    )
    staging = tmp_path / "staging"

    build_recognition_staging(local, staging)

    rows = _read_metadata(staging)
    assert len(rows) == 1
    row = rows[0]
    # HF imagefolder convention: file_name is a path inside the staging
    # dir relative to metadata.jsonl, prefixed with data/.
    assert row["file_name"] == "data/0000000.png"
    assert row["text"] == "Séadna"
    assert row["font"] == "bungc.otf"
    assert row["font_size_pt"] == 14.0
    # Degradations are flattened to kind names — the dataset card
    # filters by kind, not by per-sample kwargs.
    assert row["degradations"] == ["skew", "paper_texture", "jpeg"]
    # Corpus collapses to provider:key for HF schema simplicity.
    assert row["corpus"] == "wikisource:ga:Séadna"


def test_staging_metadata_jsonl_is_sorted(tmp_path: Path) -> None:
    """Ordered output matters for the future content-SHA idempotency
    check — non-determinism here would cause spurious "no changes"
    misses on re-staging."""

    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[
            {"index": 2, "text": "gamma", "font_name": "f.otf"},
            {"index": 0, "text": "alpha", "font_name": "f.otf"},
            {"index": 1, "text": "beta", "font_name": "f.otf"},
        ],
    )
    staging = tmp_path / "staging"

    build_recognition_staging(local, staging)
    rows = _read_metadata(staging)

    assert [row["file_name"] for row in rows] == [
        "data/0000000.png",
        "data/0000001.png",
        "data/0000002.png",
    ]


# ---------------------------------------------------------------------------
# Skipped / missing inputs
# ---------------------------------------------------------------------------


def test_staging_drops_skipped_manifest_rows(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[
            {"index": 0, "text": "alpha", "font_name": "f.otf"},
            {"index": 1, "status": "skipped", "skip_reason": "missing_glyph"},
            {"index": 2, "text": "gamma", "font_name": "f.otf"},
        ],
    )
    staging = tmp_path / "staging"

    result = build_recognition_staging(local, staging)

    rows = _read_metadata(staging)
    # Only the rendered samples land in metadata.jsonl: HF doesn't
    # want rows that don't have an image on disk.
    assert [row["text"] for row in rows] == ["alpha", "gamma"]
    assert result.rows_written == 2
    assert result.skipped_manifest_rows == 1
    assert not (staging / DATA_DIRNAME / "0000001.png").exists()


def test_staging_tolerates_missing_manifest(tmp_path: Path) -> None:
    """Older runs / crashed runs without a manifest still stage; the
    metadata.jsonl just lacks the provenance columns."""

    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[{"index": 0, "text": "alpha", "font_name": "f.otf"}],
        write_manifest=False,
    )
    staging = tmp_path / "staging"

    result = build_recognition_staging(local, staging)

    rows = _read_metadata(staging)
    assert rows[0]["text"] == "alpha"
    assert "font" not in rows[0]
    assert result.rows_written == 1


def test_staging_records_missing_images(tmp_path: Path) -> None:
    """If a label has no corresponding image file, surface it in the
    result rather than silently writing a broken row."""

    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[
            {"index": 0, "text": "alpha", "font_name": "f.otf"},
            {
                "index": 1,
                "text": "beta",
                "font_name": "f.otf",
                "write_image": False,
            },
        ],
    )
    staging = tmp_path / "staging"

    result = build_recognition_staging(local, staging)

    rows = _read_metadata(staging)
    assert [row["text"] for row in rows] == ["alpha"]
    assert result.images_copied == 1
    assert result.missing_images == ["0000001.png"]


# ---------------------------------------------------------------------------
# Snapshot copying
# ---------------------------------------------------------------------------


def test_staging_copies_recipe_snapshot(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[{"index": 0, "text": "alpha", "font_name": "f.otf"}],
    )
    staging = tmp_path / "staging"

    result = build_recognition_staging(local, staging)

    assert result.snapshot_copied is True
    snapshot = staging / "recipe.snapshot.yaml"
    assert snapshot.is_file()
    assert "tool_version" in snapshot.read_text(encoding="utf-8")


def test_staging_without_snapshot_still_succeeds(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[{"index": 0, "text": "alpha", "font_name": "f.otf"}],
        write_snapshot=False,
    )
    staging = tmp_path / "staging"

    result = build_recognition_staging(local, staging)
    assert result.snapshot_copied is False
    assert not (staging / "recipe.snapshot.yaml").exists()


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_staging_raises_on_missing_labels(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    (local / "images").mkdir()
    staging = tmp_path / "staging"

    with pytest.raises(StagingError, match="labels.json"):
        build_recognition_staging(local, staging)


def test_staging_raises_on_missing_images_dir(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    (local / "labels.json").write_text("{}\n", encoding="utf-8")
    staging = tmp_path / "staging"

    with pytest.raises(StagingError, match="images"):
        build_recognition_staging(local, staging)


def test_staging_raises_on_corrupt_labels(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    (local / "images").mkdir()
    (local / "labels.json").write_text("not json", encoding="utf-8")
    staging = tmp_path / "staging"

    with pytest.raises(StagingError, match="not valid JSON"):
        build_recognition_staging(local, staging)


def test_staging_refuses_nonempty_destination(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[{"index": 0, "text": "alpha", "font_name": "f.otf"}],
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "leftover.txt").write_text("from before", encoding="utf-8")

    with pytest.raises(StagingError, match="not empty"):
        build_recognition_staging(local, staging)


def test_staging_overwrites_when_requested(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[{"index": 0, "text": "alpha", "font_name": "f.otf"}],
    )
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "leftover.txt").write_text("from before", encoding="utf-8")
    (staging / "stale_dir").mkdir()
    (staging / "stale_dir" / "x").write_bytes(b"x")

    result = build_recognition_staging(local, staging, overwrite=True)

    assert not (staging / "leftover.txt").exists()
    assert not (staging / "stale_dir").exists()
    assert (staging / DATA_DIRNAME / "0000000.png").is_file()
    assert result.images_copied == 1


def test_staging_creates_destination_if_missing(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[{"index": 0, "text": "alpha", "font_name": "f.otf"}],
    )
    staging = tmp_path / "deep" / "nested" / "staging"

    result = build_recognition_staging(local, staging)

    assert staging.is_dir()
    assert result.images_copied == 1


# ---------------------------------------------------------------------------
# Edge cases on metadata fields
# ---------------------------------------------------------------------------


def test_staging_handles_string_corpus(tmp_path: Path) -> None:
    """A flat string ``corpus`` entry (rather than the spec's nested
    dict) should pass through verbatim — keeps the path forward-
    compatible with simpler manifest emitters."""

    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[
            {
                "index": 0,
                "text": "alpha",
                "font_name": "f.otf",
                "corpus": "local:./seed.txt",
            },
        ],
    )
    staging = tmp_path / "staging"

    build_recognition_staging(local, staging)
    rows = _read_metadata(staging)
    assert rows[0]["corpus"] == "local:./seed.txt"


def test_staging_handles_string_degradations(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_output(
        local,
        samples=[
            {
                "index": 0,
                "text": "alpha",
                "font_name": "f.otf",
                "degradations": ["skew", "jpeg"],
            },
        ],
    )
    staging = tmp_path / "staging"

    build_recognition_staging(local, staging)
    rows = _read_metadata(staging)
    assert rows[0]["degradations"] == ["skew", "jpeg"]


# ---------------------------------------------------------------------------
# Integration: round-trip from RecognitionWriter
# ---------------------------------------------------------------------------


def test_staging_round_trips_a_real_recognition_writer_output(tmp_path: Path) -> None:
    """Build a tiny dataset with the actual writer and stage it.

    Locks the contract between M07 (writer) and M08 (staging): any
    change to the writer's manifest shape that breaks staging will
    fail here, not silently in production.
    """

    from types import SimpleNamespace

    from pd_ocr_synth.output import RecognitionWriter
    from pd_ocr_synth.recipe import load_recipe

    font = tmp_path / "fake.otf"
    font.write_bytes(b"\x00fake")
    seed = tmp_path / "seed.txt"
    seed.write_text("alpha\n", encoding="utf-8")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(
        f"""schema_version: 1
name: round-trip
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 2
corpus:
  - type: local
    path: {seed}
fonts:
  - path: {font}
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 240, b: 240 }}
layout:
  mode: word_crops
  padding_px: 4
""",
        encoding="utf-8",
    )
    recipe = load_recipe(rp)
    local = tmp_path / "render"

    def _sample(text: str) -> SimpleNamespace:
        img = Image.new("RGB", (16, 8), color=(240, 240, 240))
        return SimpleNamespace(
            text=text,
            image=img,
            font_path=font,
            font_size_pt=14.0,
            dpi=300,
            ink_color=(10, 10, 10),
            background_color=(240, 240, 240),
            size=(16, 8),
            bbox=(0, 0, 16, 8),
            glyph_runs=(),
        )

    with RecognitionWriter.open(recipe, local, seed=recipe.seed) as writer:
        writer.write_rendered(
            0,
            _sample("alpha"),
            text="alpha",
            applied_degradations=[{"kind": "skew"}],
        )
        writer.write_skipped(1, reason="missing_glyph")

    staging = tmp_path / "staging"
    result = build_recognition_staging(local, staging)

    assert result.images_copied == 1
    assert result.rows_written == 1
    assert result.skipped_manifest_rows == 1
    assert result.snapshot_copied is True

    rows = _read_metadata(staging)
    assert rows[0]["file_name"] == "data/0000000.png"
    assert rows[0]["text"] == "alpha"
    assert rows[0]["font"] == "fake.otf"
    assert rows[0]["font_size_pt"] == 14.0
    assert rows[0]["degradations"] == ["skew"]


def test_staging_omits_provenance_when_manifest_lacks_it(tmp_path: Path) -> None:
    """A manifest with status=rendered but no font/degradations/corpus
    keys should still produce a row — just one with only file_name +
    text."""

    local = tmp_path / "local"
    images = local / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(images / "0000000.png", format="PNG")
    (local / "labels.json").write_text(
        json.dumps({"0000000.png": "alpha"}) + "\n", encoding="utf-8"
    )
    (local / "manifest.jsonl").write_text(
        json.dumps(
            {
                "index": 0,
                "image": "images/0000000.png",
                "status": "rendered",
                "text": "alpha",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    staging = tmp_path / "staging"

    build_recognition_staging(local, staging)
    rows = _read_metadata(staging)
    assert rows[0] == {"file_name": "data/0000000.png", "text": "alpha"}
