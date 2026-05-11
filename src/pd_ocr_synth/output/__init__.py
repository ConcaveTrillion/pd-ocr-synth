"""Output writers — turn rendered samples into trainer-consumable layouts.

Recognition mode (M07) writes the ``pd-ocr-trainer/v1`` recognition
profile: ``images/<NAME>.png`` + ``labels.json`` + ``manifest.jsonl`` +
``recipe.snapshot.yaml`` + ``stats.json``.

Detection mode (M09) writes the parallel detection profile:
``images/page_<NAME>.png`` + ``labels.json`` (with per-line polygons +
rich GT) + the same manifest/snapshot/stats sidecars. The detection
writer is scaffolding only as of M09 — it is exposed here so tests
and future paragraph/page dispatch can drive it directly, but
``run_recipe`` does not yet route to it.
"""

from __future__ import annotations

from pd_ocr_synth.output.detection import (
    DetectionStats,
    DetectionWriter,
    bbox_to_polygon,
    page_filename,
)
from pd_ocr_synth.output.recognition import (
    RecognitionWriter,
    RenderStats,
    image_filename,
    width_for_count,
)
from pd_ocr_synth.output.snapshot import (
    SnapshotMismatchError,
    build_snapshot,
    load_snapshot,
    snapshot_matches,
    write_snapshot,
)

__all__ = [
    "DetectionStats",
    "DetectionWriter",
    "RecognitionWriter",
    "RenderStats",
    "SnapshotMismatchError",
    "bbox_to_polygon",
    "build_snapshot",
    "image_filename",
    "load_snapshot",
    "page_filename",
    "snapshot_matches",
    "width_for_count",
    "write_snapshot",
]
