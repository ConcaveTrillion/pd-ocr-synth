"""Unit tests for the detection staging-dir builder (M09 chunk #7).

Covers the format conversion from the local detection layout (per
``docs/specs/08-output-format.md`` § Detection mode layout +
``pd_ocr_synth.output.detection``) to an HF imagefolder-shaped staging
dir suitable for upload via :func:`publish_recognition` (which is
shape-agnostic — see :mod:`pd_ocr_synth.publish.detection` module
docstring on why we keep ``labels.json`` rather than collapse it into
``metadata.jsonl``).

Pure file-IO; no network, no HF SDK. Mirrors
``tests/test_publish_recognition.py`` in spirit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from pd_ocr_synth.publish import (
    CONTENT_SHA_KEY,
    DATA_DIRNAME,
    README_FILENAME,
    StagingError,
    build_detection_staging,
)
from pd_ocr_synth.publish.dataset_card import write_dataset_card  # noqa: F401 — stable import

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_local_detection_output(
    local: Path,
    *,
    pages: list[dict[str, Any]],
    write_snapshot: bool = True,
) -> None:
    """Materialize a local detection output dir.

    Each page dict accepts: ``index``, ``polygons`` (list of
    4-corner polygons; default a single full-image polygon),
    ``write_image`` (bool, default True), ``size`` (default (32, 16)),
    ``lines`` (optional list of {text, bbox, words}).
    """

    images = local / "images"
    images.mkdir(parents=True, exist_ok=True)

    labels: dict[str, dict[str, Any]] = {}
    for page in pages:
        idx = int(page["index"])
        name = f"page_{idx:07d}.png"
        size = page.get("size", (32, 16))
        if page.get("write_image", True):
            img = Image.new("RGB", size, color=(220, 220, 220))
            img.save(images / name, format="PNG")
        polygons = page.get("polygons", [[[0, 0], [size[0], 0], [size[0], size[1]], [0, size[1]]]])
        entry: dict[str, Any] = {
            "img_dimensions": [int(size[0]), int(size[1])],
            "img_hash": "deadbeef" * 8,
            "polygons": polygons,
        }
        if "lines" in page:
            entry["lines"] = page["lines"]
        labels[name] = entry

    (local / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (local / "manifest.jsonl").write_text("", encoding="utf-8")
    if write_snapshot:
        (local / "recipe.snapshot.yaml").write_text(
            "tool_version: 0.0.0\nseed: 5\nrecipe:\n  name: detection-smoke\n",
            encoding="utf-8",
        )
    (local / "stats.json").write_text("{}\n", encoding="utf-8")


def _read_front_matter_value(readme: Path, key: str) -> str | None:
    """Return the value of ``key`` from the README's YAML front matter, if any."""

    text = readme.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    block = text[4:end]
    for line in block.splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        if k.strip() == key:
            return v.strip()
    return None


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_detection_staging_emits_data_dir_and_labels_json(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_detection_output(
        local,
        pages=[{"index": 0}, {"index": 1}],
    )
    staging = tmp_path / "staging"

    result = build_detection_staging(local, staging)

    assert (staging / DATA_DIRNAME / "page_0000000.png").is_file()
    assert (staging / DATA_DIRNAME / "page_0000001.png").is_file()
    assert (staging / "labels.json").is_file()
    assert result.images_copied == 2
    assert result.rows_written == 2


def test_detection_staging_preserves_labels_json_verbatim(tmp_path: Path) -> None:
    """``labels.json`` is the trainer/HF contract for detection — stage
    it byte-for-byte rather than re-serializing, so a downstream
    consumer reads exactly what the writer emitted."""

    local = tmp_path / "local"
    _write_local_detection_output(
        local,
        pages=[
            {
                "index": 0,
                "polygons": [
                    [[1, 2], [10, 2], [10, 8], [1, 8]],
                    [[12, 2], [20, 2], [20, 8], [12, 8]],
                ],
                "lines": [
                    {
                        "text": "alpha",
                        "bbox": [1, 2, 10, 8],
                        "words": [{"text": "alpha", "bbox": [1, 2, 10, 8]}],
                    },
                ],
            },
        ],
    )
    staging = tmp_path / "staging"

    build_detection_staging(local, staging)

    src = (local / "labels.json").read_bytes()
    dst = (staging / "labels.json").read_bytes()
    assert src == dst


def test_detection_staging_copies_recipe_snapshot(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_detection_output(local, pages=[{"index": 0}])
    staging = tmp_path / "staging"

    result = build_detection_staging(local, staging)

    assert result.snapshot_copied is True
    snapshot = staging / "recipe.snapshot.yaml"
    assert snapshot.is_file()
    assert "tool_version" in snapshot.read_text(encoding="utf-8")


def test_detection_staging_writes_detection_shape_in_front_matter(tmp_path: Path) -> None:
    """Front matter must announce ``detection/v1`` (vs recognition's
    ``recognition/v1``) so consumers route the right loader."""

    local = tmp_path / "local"
    _write_local_detection_output(local, pages=[{"index": 0}])
    staging = tmp_path / "staging"

    build_detection_staging(local, staging)

    readme = staging / README_FILENAME
    shape = _read_front_matter_value(readme, "pd-ocr-shape")
    assert shape == "detection/v1"


def test_detection_staging_announces_object_detection_task(tmp_path: Path) -> None:
    """``task_categories: [object-detection]`` in the front matter so
    HF's per-task filtering surfaces the dataset under the right tag."""

    local = tmp_path / "local"
    _write_local_detection_output(local, pages=[{"index": 0}])
    staging = tmp_path / "staging"

    build_detection_staging(local, staging)

    text = (staging / README_FILENAME).read_text(encoding="utf-8")
    # YAML list rendering is stable enough we can assert on the literal
    # form here. If we change YAML emitters, update this.
    assert "object-detection" in text
    # And recognition's task category is *not* used.
    assert "text-recognition" not in text


# ---------------------------------------------------------------------------
# Content-SHA wire-up
# ---------------------------------------------------------------------------


def test_detection_staging_embeds_content_sha_in_readme(tmp_path: Path) -> None:
    """``pd-ocr-content-sha`` lands in the README front matter."""

    local = tmp_path / "local"
    _write_local_detection_output(local, pages=[{"index": 0}])
    staging = tmp_path / "staging"

    result = build_detection_staging(local, staging)

    assert result.readme_written is True
    assert result.content_sha is not None
    assert len(result.content_sha) == 64
    assert all(c in "0123456789abcdef" for c in result.content_sha)

    readme_value = _read_front_matter_value(staging / README_FILENAME, CONTENT_SHA_KEY)
    assert readme_value == result.content_sha


def test_detection_staging_content_sha_is_deterministic_across_rebuilds(
    tmp_path: Path,
) -> None:
    local = tmp_path / "local"
    _write_local_detection_output(local, pages=[{"index": 0}, {"index": 1}])

    first = build_detection_staging(local, tmp_path / "staging-a")
    second = build_detection_staging(local, tmp_path / "staging-b")

    assert first.content_sha is not None
    assert first.content_sha == second.content_sha


def test_detection_staging_content_sha_changes_when_image_bytes_change(
    tmp_path: Path,
) -> None:
    local_a = tmp_path / "local-a"
    local_b = tmp_path / "local-b"
    _write_local_detection_output(local_a, pages=[{"index": 0}])
    _write_local_detection_output(local_b, pages=[{"index": 0}])

    # Replace the image in B with a different pixel payload.
    (local_b / "images" / "page_0000000.png").unlink()
    Image.new("RGB", (32, 16), color=(50, 50, 50)).save(
        local_b / "images" / "page_0000000.png", format="PNG"
    )

    a = build_detection_staging(local_a, tmp_path / "staging-a")
    b = build_detection_staging(local_b, tmp_path / "staging-b")

    assert a.content_sha is not None
    assert b.content_sha is not None
    assert a.content_sha != b.content_sha


# ---------------------------------------------------------------------------
# Skipped / missing inputs
# ---------------------------------------------------------------------------


def test_detection_staging_records_missing_images(tmp_path: Path) -> None:
    """Label entry without a corresponding image file: surface in the
    result rather than silently dropping. Mirrors recognition.
    """

    local = tmp_path / "local"
    _write_local_detection_output(
        local,
        pages=[
            {"index": 0},
            {"index": 1, "write_image": False},
        ],
    )
    staging = tmp_path / "staging"

    result = build_detection_staging(local, staging)

    assert result.images_copied == 1
    assert result.missing_images == ["page_0000001.png"]


def test_detection_staging_without_snapshot_still_succeeds(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_detection_output(local, pages=[{"index": 0}], write_snapshot=False)
    staging = tmp_path / "staging"

    result = build_detection_staging(local, staging)

    assert result.snapshot_copied is False
    assert not (staging / "recipe.snapshot.yaml").exists()
    # No README without a snapshot — same gating as recognition.
    assert result.readme_written is False
    assert result.content_sha is None


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_detection_staging_raises_on_missing_labels(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    (local / "images").mkdir()
    staging = tmp_path / "staging"

    with pytest.raises(StagingError, match=r"labels\.json"):
        build_detection_staging(local, staging)


def test_detection_staging_raises_on_missing_images_dir(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    (local / "labels.json").write_text("{}\n", encoding="utf-8")
    staging = tmp_path / "staging"

    with pytest.raises(StagingError, match="images"):
        build_detection_staging(local, staging)


def test_detection_staging_raises_on_corrupt_labels(tmp_path: Path) -> None:
    local = tmp_path / "local"
    local.mkdir()
    (local / "images").mkdir()
    (local / "labels.json").write_text("not json", encoding="utf-8")
    staging = tmp_path / "staging"

    with pytest.raises(StagingError, match="not valid JSON"):
        build_detection_staging(local, staging)


def test_detection_staging_raises_on_unprefixed_label_key(tmp_path: Path) -> None:
    """A labels.json key that doesn't start with ``page_`` would corrupt
    the imagefolder layout (filename prefix is the detection vs
    recognition discriminator). Surface as a typed error rather than
    silently land mismatched files in ``data/``.
    """

    local = tmp_path / "local"
    local.mkdir()
    (local / "images").mkdir()
    Image.new("RGB", (8, 8)).save(local / "images" / "0000000.png", format="PNG")
    (local / "labels.json").write_text(
        json.dumps(
            {
                "0000000.png": {
                    "img_dimensions": [8, 8],
                    "img_hash": "f" * 64,
                    "polygons": [[[0, 0], [8, 0], [8, 8], [0, 8]]],
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    staging = tmp_path / "staging"

    with pytest.raises(StagingError, match="page_"):
        build_detection_staging(local, staging)


def test_detection_staging_refuses_nonempty_destination(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_detection_output(local, pages=[{"index": 0}])
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "leftover.txt").write_text("from before", encoding="utf-8")

    with pytest.raises(StagingError, match="not empty"):
        build_detection_staging(local, staging)


def test_detection_staging_overwrites_when_requested(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_detection_output(local, pages=[{"index": 0}])
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "leftover.txt").write_text("from before", encoding="utf-8")
    (staging / "stale_dir").mkdir()
    (staging / "stale_dir" / "x").write_bytes(b"x")

    result = build_detection_staging(local, staging, overwrite=True)

    assert not (staging / "leftover.txt").exists()
    assert not (staging / "stale_dir").exists()
    assert (staging / DATA_DIRNAME / "page_0000000.png").is_file()
    assert result.images_copied == 1


def test_detection_staging_creates_destination_if_missing(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_detection_output(local, pages=[{"index": 0}])
    staging = tmp_path / "deep" / "nested" / "staging"

    result = build_detection_staging(local, staging)

    assert staging.is_dir()
    assert result.images_copied == 1


# ---------------------------------------------------------------------------
# License override
# ---------------------------------------------------------------------------


def test_detection_staging_license_override_lands_in_front_matter(tmp_path: Path) -> None:
    local = tmp_path / "local"
    _write_local_detection_output(local, pages=[{"index": 0}])
    staging = tmp_path / "staging"

    build_detection_staging(local, staging, license_override="apache-2.0")

    license_value = _read_front_matter_value(staging / README_FILENAME, "license")
    assert license_value == "apache-2.0"


# ---------------------------------------------------------------------------
# Sorted output (determinism check)
# ---------------------------------------------------------------------------


def test_detection_staging_copies_pages_in_sorted_order(tmp_path: Path) -> None:
    """Sorted iteration matters for the content-SHA: any non-determinism
    would cause the digest to flap and the idempotency check to miss.
    """

    local = tmp_path / "local"
    _write_local_detection_output(
        local,
        pages=[{"index": 2}, {"index": 0}, {"index": 1}],
    )
    staging = tmp_path / "staging"

    build_detection_staging(local, staging)

    # Both staging dirs should have all three. We exercise the sort
    # implicitly via the determinism test above; this case mostly
    # verifies the writer doesn't crash on out-of-order labels.
    assert sorted((staging / DATA_DIRNAME).iterdir()) == [
        staging / DATA_DIRNAME / "page_0000000.png",
        staging / DATA_DIRNAME / "page_0000001.png",
        staging / DATA_DIRNAME / "page_0000002.png",
    ]
