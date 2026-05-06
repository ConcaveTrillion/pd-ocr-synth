"""Build an HF imagefolder staging directory from a local render output.

The recognition writer (see ``pd_ocr_synth.output.recognition``) drops
its files in the layout the trainer reads directly:

    <local>/
    ├── images/<NNNNNNN>.png
    ├── labels.json          # {"<NNNNNNN>.png": "<text>"}
    ├── manifest.jsonl       # one record per attempted sample
    ├── recipe.snapshot.yaml
    └── stats.json

Hugging Face datasets prefer a different shape (the "imagefolder"
convention): images under ``data/`` next to a single ``metadata.jsonl``
that joins each image with its label and any flat provenance columns.

This module converts the first into the second. It is intentionally
**pure file-IO** — no ``huggingface_hub`` import, no network — so the
conversion step is testable in isolation and the upload step can layer
on top without re-reading the local layout. See ``docs/specs/10-
publishing.md`` and ``docs/roadmap/08-publishing-hf.md``.

Resulting staging layout::

    <staging>/
    ├── data/<NNNNNNN>.png
    ├── metadata.jsonl       # one row per rendered sample
    └── recipe.snapshot.yaml

Skipped manifest entries (no image on disk) are left out of
``metadata.jsonl``: the dataset only wants rows that actually
correspond to an image. Row count therefore matches
``len(labels.json)``, not ``len(manifest.jsonl)``.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pd_ocr_synth.output.recognition import (
    IMAGES_DIRNAME,
    LABELS_FILENAME,
    MANIFEST_FILENAME,
)
from pd_ocr_synth.output.snapshot import SNAPSHOT_FILENAME
from pd_ocr_synth.publish.dataset_card import (
    load_card_inputs,
    write_dataset_card,
)

# HF imagefolder convention: images live under ``data/`` and the
# ``file_name`` column in ``metadata.jsonl`` is the path relative to
# the staging root. Centralized so the writer + tests + future upload
# wrapper don't drift.
DATA_DIRNAME = "data"
METADATA_FILENAME = "metadata.jsonl"


class StagingError(Exception):
    """Raised for unrecoverable problems building the staging dir.

    Distinct exception type so the future CLI can map it to its own
    exit code (per ``docs/specs/10-publishing.md`` "Errors and
    recovery": local output corrupt / missing maps to exit 6 / 5).
    """


@dataclass(slots=True)
class StagingResult:
    """Counters returned from a staging build.

    Captures enough to drive the future ``--dry-run`` summary
    (``Files: N (M .png + metadata.jsonl + ...)``) without having to
    re-stat the staging directory.
    """

    staging_dir: Path
    images_copied: int = 0
    rows_written: int = 0
    snapshot_copied: bool = False
    readme_written: bool = False
    skipped_manifest_rows: int = 0
    # Filenames present in ``labels.json`` whose image file was missing
    # on disk. Surfaced (not silently dropped) so a corrupt local
    # render is auditable.
    missing_images: list[str] = field(default_factory=list)


def build_recognition_staging(
    local_output_dir: Path,
    staging_dir: Path,
    *,
    overwrite: bool = False,
) -> StagingResult:
    """Convert a local recognition layout into an HF staging dir.

    Parameters
    ----------
    local_output_dir:
        Directory the recognition writer produced. Must contain at
        least ``labels.json`` and ``images/``; the ``manifest.jsonl``
        is read for provenance columns but tolerated when missing
        (older runs might not have one).
    staging_dir:
        Destination for the HF-shaped output. Must either not exist,
        be empty, or be passed with ``overwrite=True``. The function
        creates the directory if it doesn't yet exist.
    overwrite:
        When true, an existing ``staging_dir`` is wiped before the
        new build. Default ``False`` so an accidental re-run doesn't
        silently clobber a previously-uploaded staging dir.

    Returns
    -------
    StagingResult
        Counters describing what landed on disk.

    Raises
    ------
    StagingError
        On missing local output, malformed labels/manifest JSON, or a
        non-empty staging dir without ``overwrite``.
    """

    local = Path(local_output_dir)
    staging = Path(staging_dir)

    labels = _read_labels(local)
    manifest_records = _read_manifest_records(local)
    manifest_by_image = _index_manifest_by_image(manifest_records)

    _prepare_staging(staging, overwrite=overwrite)

    images_src = local / IMAGES_DIRNAME
    if not images_src.is_dir():
        raise StagingError(f"local output {local} is missing the {IMAGES_DIRNAME!r} directory")
    data_dst = staging / DATA_DIRNAME
    data_dst.mkdir(parents=True, exist_ok=True)

    result = StagingResult(staging_dir=staging)

    metadata_rows: list[dict[str, Any]] = []
    # Iterate in sorted order so the metadata.jsonl is deterministic
    # — important for the future content-SHA idempotency check.
    for image_name in sorted(labels):
        image_src = images_src / image_name
        if not image_src.is_file():
            result.missing_images.append(image_name)
            continue
        image_dst = data_dst / image_name
        # ``copy2`` preserves mtime; we don't currently rely on it but
        # it makes the staging dir easier to inspect after the fact.
        shutil.copy2(image_src, image_dst)
        result.images_copied += 1

        manifest_record = manifest_by_image.get(image_name, {})
        row = _build_metadata_row(
            image_name=image_name,
            text=labels[image_name],
            manifest_record=manifest_record,
        )
        metadata_rows.append(row)
        result.rows_written += 1

    # Skipped manifest rows (no image, no label) — not an error, just
    # surfaced for the result. The render writer already records the
    # skip reason in ``stats.json``; we only track the count here.
    # Counted from the full manifest, not from the by-image index,
    # because skipped records have no ``image`` field.
    result.skipped_manifest_rows = sum(
        1 for rec in manifest_records if rec.get("status") == "skipped"
    )

    _write_metadata_jsonl(staging / METADATA_FILENAME, metadata_rows)

    snapshot_src = local / SNAPSHOT_FILENAME
    if snapshot_src.is_file():
        shutil.copy2(snapshot_src, staging / SNAPSHOT_FILENAME)
        result.snapshot_copied = True

    # Dataset-card README. Generated only when we have a snapshot to
    # build it from — without the snapshot we'd be missing the recipe
    # block, the tool version, and the recipe-SHA front-matter key, so
    # an empty README would mislead consumers more than its absence.
    if result.snapshot_copied:
        card_inputs = load_card_inputs(local)
        write_dataset_card(staging, card_inputs)
        result.readme_written = True

    return result


# ---------------------------------------------------------------------------
# Reading local output
# ---------------------------------------------------------------------------


def _read_labels(local: Path) -> dict[str, str]:
    """Load ``labels.json`` into a name → text dict.

    Raises ``StagingError`` for missing / malformed files so the
    caller surfaces a single error class regardless of which input
    is broken.
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
    out: dict[str, str] = {}
    for key, value in raw.items():
        out[str(key)] = str(value)
    return out


def _read_manifest_records(local: Path) -> list[dict[str, Any]]:
    """Load every record from ``manifest.jsonl``.

    Tolerant of missing file (returns ``[]``) and malformed lines
    (skipped silently): the writer flushes one record at a time, so a
    partial trailing line from a crash shouldn't block staging.
    """

    path = local / MANIFEST_FILENAME
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def _index_manifest_by_image(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Index records by ``image`` basename for joining with labels.json.

    The ``image`` field carries ``"images/<name>.png"``; we strip the
    directory prefix so the key matches a ``labels.json`` key. Records
    without an ``image`` field (e.g. skipped samples) are not indexed
    here — counts on those come from the raw record list.
    """

    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        image_field = rec.get("image")
        if isinstance(image_field, str):
            out[Path(image_field).name] = rec
    return out


# ---------------------------------------------------------------------------
# Writing staging output
# ---------------------------------------------------------------------------


def _prepare_staging(staging: Path, *, overwrite: bool) -> None:
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


def _build_metadata_row(
    *,
    image_name: str,
    text: str,
    manifest_record: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one ``metadata.jsonl`` row.

    Schema lives in ``docs/specs/10-publishing.md`` — flat columns
    only. Missing provenance keys are simply omitted rather than
    written as ``null``: HF's parquet inference is happier with a
    column that's *absent* on some rows than one that mixes types.
    """

    row: dict[str, Any] = {
        "file_name": f"{DATA_DIRNAME}/{image_name}",
        "text": text,
    }

    font = manifest_record.get("font") or {}
    if isinstance(font, Mapping):
        font_name = font.get("name") or font.get("path")
        if font_name:
            row["font"] = str(font_name)
        size = font.get("size_pt")
        if size is not None:
            row["font_size_pt"] = float(size)

    degradations = _flatten_degradations(manifest_record.get("degradations_applied"))
    if degradations:
        row["degradations"] = degradations

    corpus = manifest_record.get("corpus")
    if isinstance(corpus, Mapping):
        corpus_value = _format_corpus(corpus)
        if corpus_value is not None:
            row["corpus"] = corpus_value
    elif isinstance(corpus, str):
        row["corpus"] = corpus

    return row


def _flatten_degradations(value: Any) -> list[str]:
    """Reduce per-stage degradation records to a list of stage names.

    The manifest carries the configured stages (with their kwargs);
    the dataset card / metadata only needs the *kind* labels for
    filtering and tagging. Mixed-shape inputs (list of strings, list
    of mappings) are tolerated.
    """

    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, Mapping):
            kind = item.get("kind") or item.get("name")
            if kind:
                out.append(str(kind))
    return out


def _format_corpus(corpus: Mapping[str, Any]) -> str | None:
    """Collapse the manifest's nested ``corpus`` dict to a flat string.

    Per spec 08, the manifest may carry
    ``{"provider": ..., "key": ..., "offset": ...}``; the published
    metadata wants a single string column for cleaner HF dataset
    inference. ``provider:key`` is the intuitive shape (matches what
    the recipe's wikisource and HF entries already look like).
    """

    provider = corpus.get("provider")
    key = corpus.get("key")
    if provider and key:
        return f"{provider}:{key}"
    if key:
        return str(key)
    if provider:
        return str(provider)
    return None


def _write_metadata_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Serialize rows as JSON Lines.

    ``ensure_ascii=False`` keeps Gaelic / Irish text readable in the
    on-disk file; HF's parquet conversion handles the encoding either
    way.
    """

    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
