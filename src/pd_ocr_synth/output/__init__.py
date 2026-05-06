"""Output writers — turn rendered samples into trainer-consumable layouts.

Recognition mode (M07) writes the ``pd-ocr-trainer/v1`` recognition
profile: ``images/<NAME>.png`` + ``labels.json`` + ``manifest.jsonl`` +
``recipe.snapshot.yaml`` + ``stats.json``.

Detection mode is M09 and adds ``pages.json`` instead of the
recognition labels file.
"""

from __future__ import annotations

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
    "RecognitionWriter",
    "RenderStats",
    "SnapshotMismatchError",
    "build_snapshot",
    "image_filename",
    "load_snapshot",
    "snapshot_matches",
    "width_for_count",
    "write_snapshot",
]
