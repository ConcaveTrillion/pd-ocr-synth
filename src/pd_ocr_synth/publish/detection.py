"""Build an HF imagefolder staging directory from a local detection render.

The detection writer (see :mod:`pd_ocr_synth.output.detection`) drops
its files in the layout the trainer reads directly::

    <local>/
    ├── images/page_NNNNNNN.png
    ├── labels.json          # {"page_NNNNNNN.png": {...}, ...}
    ├── manifest.jsonl       # one record per attempted page
    ├── recipe.snapshot.yaml
    └── stats.json

This module is the M09 analog of
:mod:`pd_ocr_synth.publish.recognition`. It converts the local
detection layout into a Hugging Face *imagefolder*-shaped staging dir
so the existing
:func:`pd_ocr_synth.publish.publish_recognition` orchestrator (which
calls ``upload_folder`` on the transport) can ship it. Spec 10 § Format
conversion — detection ultimately calls for parquet sharding via
``datasets.Dataset.from_generator(...).push_to_hub(...)``; that's a
larger separate chunk because (a) ``datasets`` isn't a runtime
dependency and (b) ``push_to_hub`` is a different transport surface
than ``upload_folder``. The imagefolder staging built here is the
prerequisite for either upload strategy and is what the dry-run +
preflight + content-SHA + dataset-card pipelines all consume.

Resulting staging layout::

    <staging>/
    ├── data/page_NNNNNNN.png
    ├── labels.json          # copied verbatim from local; bbox + polygon GT
    ├── recipe.snapshot.yaml
    └── README.md            # detection-flavored dataset card; front
                             # matter carries the ``pd-ocr-content-sha``
                             # idempotency key and ``pd-ocr-shape:
                             # detection/v1``.

Why we keep ``labels.json`` *as-is* rather than collapsing it into
``metadata.jsonl`` like recognition: detection labels are nested
(per-line / per-word bboxes + polygons) and HF imagefolder's
``metadata.jsonl`` schema strongly prefers flat columns. The
trainer's :class:`DetectionDataset` reader takes ``labels.json``
keyed by page filename, so shipping that file unchanged keeps the
trainer's pull path symmetrical to the local pull path. The eventual
parquet path will project ``labels.json`` into the spec-10 detection
schema; until then, ``labels.json`` is the contract.

Pure file-IO — no ``huggingface_hub`` import, no network — same
discipline as :func:`build_recognition_staging`.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from pd_ocr_synth.output.detection import (
    IMAGES_DIRNAME,
    LABELS_FILENAME,
    PAGE_PREFIX,
)
from pd_ocr_synth.output.snapshot import SNAPSHOT_FILENAME
from pd_ocr_synth.publish.content_sha import (
    apply_content_sha_to_readme,
    compute_content_sha,
)
from pd_ocr_synth.publish.dataset_card import (
    load_card_inputs,
    write_dataset_card,
)
from pd_ocr_synth.publish.recognition import (
    DATA_DIRNAME,
    StagingError,
    StagingResult,
)

# Re-exported so this module can be imported in isolation by callers
# that don't want to reach into the recognition module for the
# imagefolder dirname.
__all__ = [
    "DATA_DIRNAME",
    "StagingError",
    "StagingResult",
    "build_detection_staging",
]


def build_detection_staging(
    local_output_dir: Path,
    staging_dir: Path,
    *,
    overwrite: bool = False,
    license_override: str | None = None,
) -> StagingResult:
    """Convert a local detection layout into an HF staging dir.

    Parameters
    ----------
    local_output_dir:
        Directory the detection writer produced. Must contain at
        least ``labels.json`` and ``images/``; ``manifest.jsonl`` is
        not consumed here (detection labels carry their own per-page
        bbox provenance), but ``recipe.snapshot.yaml`` is required for
        the dataset card / front matter.
    staging_dir:
        Destination for the HF-shaped output. Must either not exist,
        be empty, or be passed with ``overwrite=True``. The function
        creates the directory if it doesn't yet exist.
    overwrite:
        When true, an existing ``staging_dir`` is wiped before the
        new build. Default ``False`` so an accidental re-run doesn't
        silently clobber a previously-uploaded staging dir.
    license_override:
        Spec 10's ``--license`` CLI flag value. When set, lands as the
        dataset card's ``license:`` front-matter key, overriding any
        ``recipe.publish.hf_dataset.license``. ``None`` falls back to
        the recipe value (or omits the key when neither is set).

    Returns
    -------
    StagingResult
        Counters describing what landed on disk. ``rows_written`` here
        equals ``images_copied`` (one row == one page); the recognition
        notion of "rows from a flat metadata.jsonl" doesn't apply to
        detection, so we surface the page-count under both fields for
        compatibility with the recognition-shaped result type.

    Raises
    ------
    StagingError
        On missing local output, malformed labels JSON, or a non-empty
        staging dir without ``overwrite``.
    """

    local = Path(local_output_dir)
    staging = Path(staging_dir)

    labels = _read_labels(local)

    _prepare_staging(staging, overwrite=overwrite)

    images_src = local / IMAGES_DIRNAME
    if not images_src.is_dir():
        raise StagingError(f"local output {local} is missing the {IMAGES_DIRNAME!r} directory")
    data_dst = staging / DATA_DIRNAME
    data_dst.mkdir(parents=True, exist_ok=True)

    result = StagingResult(staging_dir=staging)

    # Detection labels are keyed by page filename (e.g.
    # ``page_0000000.png``). We iterate sorted for deterministic
    # output — the content SHA depends on byte-stable file ordering
    # downstream.
    for page_name in sorted(labels):
        if not page_name.startswith(PAGE_PREFIX):
            # Defensive: a hand-edited labels.json with non-page keys
            # would corrupt the imagefolder layout. Surface rather
            # than silently include.
            raise StagingError(
                f"labels.json key {page_name!r} does not start with {PAGE_PREFIX!r}; "
                "the detection writer always emits page-prefixed filenames"
            )
        image_src = images_src / page_name
        if not image_src.is_file():
            result.missing_images.append(page_name)
            continue
        image_dst = data_dst / page_name
        shutil.copy2(image_src, image_dst)
        result.images_copied += 1
        # rows_written tracks "samples that will appear in the
        # uploaded dataset" — for detection that's one per copied
        # page, since labels.json is a single file with one entry per
        # page.
        result.rows_written += 1

    # Copy ``labels.json`` verbatim. The detection writer already
    # writes it sorted + indented, so the staging copy is byte-stable
    # across re-runs over identical local input — important for the
    # content-SHA idempotency loop.
    shutil.copy2(local / LABELS_FILENAME, staging / LABELS_FILENAME)

    snapshot_src = local / SNAPSHOT_FILENAME
    if snapshot_src.is_file():
        shutil.copy2(snapshot_src, staging / SNAPSHOT_FILENAME)
        result.snapshot_copied = True

    # Dataset-card README. Same gating rule as recognition: we need a
    # snapshot to render the card properly (recipe-SHA, tool version,
    # corpus / fonts blocks). Without one, skip the card entirely
    # rather than ship something misleading.
    if result.snapshot_copied:
        card_inputs = load_card_inputs(local, license_override=license_override)
        # Tag the card as detection-shape so the conventional
        # ``pd-ocr-shape`` front-matter key reads ``detection/v1``
        # rather than the recognition default.
        card_inputs.shape = "detection/v1"
        card_inputs.task_categories = ["object-detection"]
        write_dataset_card(staging, card_inputs)
        result.readme_written = True

        # Same idempotency closure as recognition: hash the staging
        # dir as it sits *without* the content-SHA line, then write
        # that digest into the README's front matter. The strip-then-
        # insert in ``apply_content_sha_to_readme`` keeps the cycle
        # idempotent.
        content_sha = compute_content_sha(staging)
        apply_content_sha_to_readme(staging, content_sha)
        result.content_sha = content_sha

    return result


# ---------------------------------------------------------------------------
# Reading local output
# ---------------------------------------------------------------------------


def _read_labels(local: Path) -> dict[str, Any]:
    """Load detection ``labels.json`` into a name → entry dict.

    Detection labels are nested (each value is itself a dict with
    ``polygons`` / ``lines`` / ``img_dimensions`` / ``img_hash``), but
    for staging we only need the *keys* — the file gets copied
    verbatim. We still parse to validate the structure: a corrupt
    labels.json should fail loud here rather than ship to HF.
    """

    path = local / LABELS_FILENAME
    if not path.is_file():
        raise StagingError(
            f"local output {local} is missing {LABELS_FILENAME}; run `pd-ocr-synth render` first"
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StagingError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise StagingError(f"{path} must be a JSON object, got {type(raw).__name__}")
    return raw


# ---------------------------------------------------------------------------
# Writing staging output
# ---------------------------------------------------------------------------


def _prepare_staging(staging: Path, *, overwrite: bool) -> None:
    """Same as recognition's ``_prepare_staging``: create empty, or
    refuse a non-empty dir without ``overwrite``.

    Inlined rather than imported from ``recognition`` to keep the
    detection module a self-contained reading: the layouts diverge
    enough downstream (one ``labels.json`` vs one ``metadata.jsonl``,
    different page-prefix conventions) that sharing this small helper
    isn't worth the cross-module coupling.
    """

    if staging.exists():
        if any(staging.iterdir()):
            if not overwrite:
                raise StagingError(
                    f"staging dir {staging} is not empty; pass overwrite=True to wipe it"
                )
            for child in staging.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
    else:
        staging.mkdir(parents=True, exist_ok=True)
