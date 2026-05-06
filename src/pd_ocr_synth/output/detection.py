"""Detection-mode output writer for ``pd-ocr-trainer/v1``.

This is the M09 analog of :mod:`pd_ocr_synth.output.recognition`. It
writes the on-disk layout that ``pd-ocr-trainer``'s detection pipeline
consumes — full pages with bbox annotations.

Layout produced (per ``docs/specs/08-output-format.md`` §"Detection
mode layout", harmonized with ``pd-ocr-trainer/dataset_store.py`` and
``doctr.datasets.DetectionDataset`` which is the actual API contract)::

    <destination>/
    ├── images/
    │   ├── page_0000000.png
    │   └── ...
    ├── labels.json          # {"page_NNNNNNN.png": {meta + polygons}, ...}
    ├── manifest.jsonl       # one JSON record per attempted page
    ├── recipe.snapshot.yaml # written by ``output.snapshot``
    └── stats.json           # run-level counters

``labels.json`` is the file ``doctr.datasets.DetectionDataset`` reads.
Its top-level keys are page filenames (no path components); each value
is an object with at minimum::

    {
      "img_dimensions": [W, H],
      "img_hash": "<sha256 hex>",
      "polygons": [ [[x,y],[x,y],[x,y],[x,y]], ... ]
    }

``polygons`` is the flat list doctr feeds to the detection head — one
4-corner polygon per detected line. We additionally write a richer
``lines`` list (with per-word bboxes + text) so downstream tooling
(labeler, parquet publish, our own CI checks) can recover the full
ground truth without re-rendering. Doctr ignores fields it doesn't
recognize, so the extra payload is free.

Spec 08's earlier draft used ``pages.json`` as the filename. The
trainer's existing reader is the canonical contract (same precedent
as M07 picking ``labels.json`` over ``labels.csv``), so detection
mode also writes ``labels.json``.

The writer is **incremental and resume-safe**, mirroring
:class:`RecognitionWriter`:

- ``open(..., resume=True)`` reads any existing ``labels.json`` /
  ``manifest.jsonl`` entries and arranges to skip indices already
  written.
- ``open(..., force=True)`` clears the destination first.
- The default behavior refuses to write into a non-empty directory.

This module is **scaffolding only** as of M09 — it is not yet wired
into ``run_recipe`` dispatch. Callers can drive it directly today
(useful for building paragraph/page renderer integration tests
ahead of the full dispatch path). The dispatch wire-up lands once
the paragraph/page renderers are ready to feed multi-line samples
through it.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, Any

from pd_ocr_synth.output.snapshot import (
    SNAPSHOT_FILENAME,
    SnapshotMismatchError,
    build_snapshot,
    load_snapshot,
    snapshot_matches,
    write_snapshot,
)
from pd_ocr_synth.recipe import Recipe

LABELS_FILENAME = "labels.json"
MANIFEST_FILENAME = "manifest.jsonl"
STATS_FILENAME = "stats.json"
IMAGES_DIRNAME = "images"

# Filename prefix for page images. The recognition writer uses bare
# ``NNNNNNN.png``; detection mode uses ``page_NNNNNNN.png`` per spec
# 08 so a profile dir holding both tasks (or a downstream tool that
# stems filenames to recover a project ID) can tell the two apart at
# a glance.
PAGE_PREFIX = "page_"

# Lower bound for zero-padding regardless of count. Same rationale as
# recognition: 7 digits handles up to 10M pages, and pinning the floor
# means a smoke run and a full run produce the same naming convention.
_MIN_PAD_WIDTH = 7


def width_for_count(count: int) -> int:
    """Pick the zero-pad width for page filenames.

    ``count`` is the recipe's ``output.count``. Width is large enough
    that every index in ``[0, count)`` lands at the same character
    width, with a sane lower bound for tiny recipes.
    """

    if count <= 0:
        return _MIN_PAD_WIDTH
    return max(_MIN_PAD_WIDTH, len(str(max(count - 1, 0))))


def page_filename(index: int, *, width: int) -> str:
    """Spec-shaped filename for page ``index``."""

    return f"{PAGE_PREFIX}{index:0{width}d}.png"


def bbox_to_polygon(
    bbox: tuple[int, int, int, int] | list[int],
) -> list[list[int]]:
    """Expand an axis-aligned ``(x0, y0, x1, y1)`` bbox to a 4-corner polygon.

    Order is clockwise from top-left::

        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    matching how ``doctr.datasets.DetectionDataset`` flattens bboxes
    via ``poly.min/max(axis=1)`` — any point order works for the
    straight-bbox case, but clockwise-from-TL is the convention used
    elsewhere in the codebase and the most natural reading order.
    """

    x0, y0, x1, y1 = (int(v) for v in bbox)
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _hash_image_bytes(image_bytes: bytes) -> str:
    """SHA-256 of an encoded PNG. Surfaced in ``labels.json`` so a
    downstream consumer can detect bit-rot / accidental re-encoding."""

    return hashlib.sha256(image_bytes).hexdigest()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class DetectionStats:
    """Accumulated counters for a detection-mode render run.

    Mirrors the ``stats.json`` schema in spec 08 with two extra
    detection-specific counters (lines + words) since "samples" at the
    page granularity hides the volume of training signal each page
    actually produces.
    """

    samples_planned: int = 0
    samples_written: int = 0
    samples_skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    fonts_used: dict[str, int] = field(default_factory=dict)
    lines_total: int = 0
    words_total: int = 0
    wall_time_seconds: float = 0.0

    def record_skip(self, reason: str) -> None:
        self.samples_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def record_render(self, *, font_name: str, n_lines: int, n_words: int) -> None:
        self.samples_written += 1
        self.fonts_used[font_name] = self.fonts_used.get(font_name, 0) + 1
        self.lines_total += int(n_lines)
        self.words_total += int(n_words)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class DetectionWriter:
    """Streaming writer for the detection profile layout.

    Use as a context manager so the labels file, manifest, snapshot
    and stats finalize even if the render loop raises::

        with DetectionWriter.open(recipe, output_dir, ...) as writer:
            for index, sample in enumerate(...):
                writer.write_rendered(index, sample, ...)
            for index in skipped_indices:
                writer.write_skipped(index, ...)

    The writer is index-addressed and idempotent on re-write of the
    same index (overwrites the PNG + label entry, replaces the
    manifest line). Idempotency is what makes ``--resume`` safe.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        *,
        recipe: Recipe,
        output_dir: Path,
        seed: int,
        pad_width: int,
        existing_labels: dict[str, dict[str, Any]],
        existing_manifest_lines: list[dict[str, Any]],
        existing_indices: set[int],
        planned_count: int | None = None,
    ) -> None:
        self.recipe = recipe
        self.output_dir = output_dir
        self.images_dir = output_dir / IMAGES_DIRNAME
        self.seed = seed
        self.pad_width = pad_width
        self._labels: dict[str, dict[str, Any]] = dict(existing_labels)
        self._manifest: dict[int, dict[str, Any]] = {
            int(rec["index"]): rec for rec in existing_manifest_lines if "index" in rec
        }
        self._existing_indices: set[int] = set(existing_indices)
        effective_planned = int(planned_count) if planned_count is not None else recipe.output.count
        self.stats = DetectionStats(samples_planned=effective_planned)
        # Pre-seed counters from existing manifest so resume reports
        # the rolled-up totals, not just the new samples.
        for rec in self._manifest.values():
            if rec.get("status") == "rendered":
                font_name = (rec.get("font") or {}).get("name")
                if font_name:
                    self.stats.fonts_used[font_name] = self.stats.fonts_used.get(font_name, 0) + 1
                self.stats.samples_written += 1
                self.stats.lines_total += int(rec.get("n_lines", 0))
                self.stats.words_total += int(rec.get("n_words", 0))
            elif rec.get("status") == "skipped":
                self.stats.record_skip(str(rec.get("reason", "unknown")))
        self._closed = False
        self._manifest_handle: IO[str] | None = None

    @classmethod
    def open(
        cls,
        recipe: Recipe,
        output_dir: Path,
        *,
        seed: int,
        force: bool = False,
        resume: bool = False,
        planned_count: int | None = None,
    ) -> DetectionWriter:
        """Prepare ``output_dir`` for writing and return a ready writer.

        Behavior matrix matches recognition (see
        :meth:`RecognitionWriter.open` for the rationale):

        - empty / nonexistent ``output_dir`` → write fresh.
        - non-empty + ``force`` → wipe everything, write fresh.
        - non-empty + ``resume`` → keep existing samples, validate
          snapshot, accept matching new writes.
        - non-empty + neither → raise ``DestinationNotEmptyError``.
        """

        output_dir = Path(output_dir)
        if force and resume:
            raise ValueError("--force and --resume are mutually exclusive")

        not_empty = output_dir.exists() and any(output_dir.iterdir())

        if not_empty and not (force or resume):
            raise DestinationNotEmptyError(
                f"destination {output_dir} is not empty; pass --force to overwrite "
                "or --resume to continue"
            )

        if force and output_dir.exists():
            for child in output_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / IMAGES_DIRNAME).mkdir(parents=True, exist_ok=True)

        existing_labels: dict[str, dict[str, Any]] = {}
        existing_manifest: list[dict[str, Any]] = []
        existing_indices: set[int] = set()

        if resume:
            current_snapshot = build_snapshot(recipe, seed=seed)
            previous = load_snapshot(output_dir)
            if previous is None:
                raise SnapshotMismatchError(
                    f"--resume requires {SNAPSHOT_FILENAME} in {output_dir}; not found"
                )
            ok, reason = snapshot_matches(previous, current_snapshot)
            if not ok:
                raise SnapshotMismatchError(
                    f"cannot resume: {reason}. Re-run without --resume or with --force."
                )
            labels_path = output_dir / LABELS_FILENAME
            if labels_path.exists():
                try:
                    raw = json.loads(labels_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise SnapshotMismatchError(
                        f"labels.json at {labels_path} is not valid JSON: {exc}"
                    ) from exc
                if isinstance(raw, dict):
                    for key, val in raw.items():
                        if isinstance(val, dict):
                            existing_labels[str(key)] = val
            manifest_path = output_dir / MANIFEST_FILENAME
            if manifest_path.exists():
                for line in manifest_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(rec, dict) and "index" in rec:
                        existing_manifest.append(rec)
                        existing_indices.add(int(rec["index"]))

        pad_width = width_for_count(recipe.output.count)
        writer = cls(
            recipe=recipe,
            output_dir=output_dir,
            seed=seed,
            pad_width=pad_width,
            existing_labels=existing_labels,
            existing_manifest_lines=existing_manifest,
            existing_indices=existing_indices,
            planned_count=planned_count,
        )
        writer._write_snapshot()
        return writer

    # ------------------------------------------------------------------
    # Public write API
    # ------------------------------------------------------------------

    def already_rendered(self, index: int) -> bool:
        """``True`` if ``index`` has a *successful* manifest entry on disk."""

        rec = self._manifest.get(int(index))
        if rec is None:
            return False
        return rec.get("status") == "rendered"

    def filename(self, index: int) -> str:
        return page_filename(index, width=self.pad_width)

    def write_rendered(
        self,
        index: int,
        sample: Any,
        *,
        applied_degradations: Iterable[dict[str, Any]] = (),
    ) -> None:
        """Persist a successfully rendered page sample.

        ``sample`` is a :class:`pd_ocr_synth.render.RenderedSample`
        produced by a paragraph or page layout — it must carry
        ``line_boxes`` (rich per-line ground truth). Per-word ground
        truth (``word_boxes``) is optional but recommended; if the
        renderer attaches words to lines via overlap (the renderer's
        contract for paragraphs/pages), they're lifted into each
        line's ``words`` array.

        We keep the ``sample`` argument loosely typed at this layer
        to avoid a cycle between ``output`` and ``render`` modules.
        """

        if self._closed:
            raise RuntimeError("DetectionWriter is already closed")
        idx = int(index)
        name = self.filename(idx)
        path = self.images_dir / name

        # Encode to PNG once so we can compute the on-disk hash without
        # re-reading the file. Pillow's ``Image.save`` writes bytes
        # identical to what we feed to the hash here (modulo metadata
        # which we don't set), so the on-disk file matches.
        sample.image.save(path, format="PNG")
        img_bytes = path.read_bytes()
        img_hash = _hash_image_bytes(img_bytes)

        line_boxes = tuple(getattr(sample, "line_boxes", ()) or ())
        word_boxes = tuple(getattr(sample, "word_boxes", ()) or ())

        # Build per-line entries (rich GT). Assign each word_box to
        # the line whose bbox contains the word's vertical center;
        # words that don't intersect any line stay attached at the
        # page level so we never silently drop ground truth.
        line_entries: list[dict[str, Any]] = []
        unassigned_words: list[Any] = list(word_boxes)
        for line in line_boxes:
            line_bbox = tuple(int(v) for v in line.bbox)
            ly0, ly1 = line_bbox[1], line_bbox[3]
            line_words: list[dict[str, Any]] = []
            kept: list[Any] = []
            for wb in unassigned_words:
                wbb = tuple(int(v) for v in wb.bbox)
                wcy = (wbb[1] + wbb[3]) // 2
                if ly0 <= wcy <= ly1:
                    line_words.append({"text": wb.text, "bbox": list(wbb)})
                else:
                    kept.append(wb)
            unassigned_words = kept
            line_entries.append(
                {
                    "text": line.text,
                    "bbox": list(line_bbox),
                    "polygon": bbox_to_polygon(line_bbox),
                    "words": line_words,
                }
            )

        # Top-level ``polygons`` is what doctr's DetectionDataset reads.
        # We fall back to per-word polygons when the renderer didn't
        # supply line GT (defensive — real paragraph/page renderers
        # always emit line_boxes).
        if line_entries:
            polygons = [entry["polygon"] for entry in line_entries]
        else:
            polygons = [bbox_to_polygon(wb.bbox) for wb in word_boxes]

        width, height = sample.size
        label_entry: dict[str, Any] = {
            "img_dimensions": [int(width), int(height)],
            "img_hash": img_hash,
            "polygons": polygons,
            "lines": line_entries,
        }
        if unassigned_words:
            # Per the no-silent-drop rule: words without a containing
            # line still appear in the label so trainer/labeler tools
            # can flag them. (Footnotes, drop caps, marginalia could
            # legitimately land here once those layouts exist.)
            label_entry["unassigned_words"] = [
                {"text": wb.text, "bbox": [int(v) for v in wb.bbox]} for wb in unassigned_words
            ]
        self._labels[name] = label_entry

        font_path = Path(sample.font_path)
        n_words = sum(len(entry["words"]) for entry in line_entries) + len(unassigned_words)
        record: dict[str, Any] = {
            "index": idx,
            "id": Path(name).stem,
            "image": f"{IMAGES_DIRNAME}/{name}",
            "status": "rendered",
            "font": {
                "name": font_path.name,
                "path": str(font_path),
                "size_pt": float(sample.font_size_pt),
            },
            "render": {
                "dpi": int(sample.dpi),
                "ink_rgb": list(sample.ink_color),
                "bg_rgb": list(sample.background_color),
            },
            "size": list(sample.size),
            "n_lines": len(line_entries),
            "n_words": n_words,
            "transforms_applied": [t.name for t in self.recipe.text_transforms],
            "degradations_applied": list(applied_degradations),
        }
        self._manifest[idx] = record
        self.stats.record_render(
            font_name=font_path.name,
            n_lines=len(line_entries),
            n_words=n_words,
        )

    def write_skipped(
        self,
        index: int,
        *,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record that ``index`` could not be rendered (no PNG, no label)."""

        if self._closed:
            raise RuntimeError("DetectionWriter is already closed")
        idx = int(index)
        # Drop any stale image/label from a previous successful render
        # at this index — keeps on-disk state consistent with manifest.
        prior = self._manifest.get(idx)
        if prior is not None and prior.get("status") == "rendered":
            stale_image = self.images_dir / self.filename(idx)
            if stale_image.exists():
                stale_image.unlink()
            self._labels.pop(self.filename(idx), None)
        record: dict[str, Any] = {
            "index": idx,
            "id": Path(self.filename(idx)).stem,
            "status": "skipped",
            "reason": reason,
        }
        if details:
            record.update(details)
        self._manifest[idx] = record
        self.stats.record_skip(reason)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Flush ``labels.json``, ``manifest.jsonl``, ``stats.json``."""

        if self._closed:
            return
        self._closed = True
        self._write_labels()
        self._write_manifest()
        self._write_stats()

    def __enter__(self) -> DetectionWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Always finalize on-disk state, even on render failure — the
        # user should be able to inspect a partial run rather than
        # losing the manifest of "what was attempted before the crash."
        self.close()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_snapshot(self) -> None:
        snapshot = build_snapshot(self.recipe, seed=self.seed)
        write_snapshot(snapshot, self.output_dir)

    def _write_labels(self) -> None:
        path = self.output_dir / LABELS_FILENAME
        ordered = {name: self._labels[name] for name in sorted(self._labels)}
        path.write_text(
            json.dumps(ordered, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    def _write_manifest(self) -> None:
        path = self.output_dir / MANIFEST_FILENAME
        with path.open("w", encoding="utf-8") as fh:
            for idx in sorted(self._manifest):
                fh.write(json.dumps(self._manifest[idx], ensure_ascii=False) + "\n")

    def _write_stats(self) -> None:
        path = self.output_dir / STATS_FILENAME
        path.write_text(
            json.dumps(self.stats.as_dict(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


class DestinationNotEmptyError(Exception):
    """Raised when ``open()`` finds a non-empty destination without
    ``--force`` or ``--resume``.

    Distinct from :class:`SnapshotMismatchError` so the CLI can map
    each to its own exit code (per ``docs/specs/01-cli.md``).
    """
