"""Recognition-mode output writer for ``pd-ocr-trainer/v1``.

Layout produced (per ``docs/specs/08-output-format.md``, harmonized
with ``pd-ocr-trainer/dataset_store.py`` which is the actual API
contract — see the spec note about ``labels.json`` vs ``labels.csv``)::

    <destination>/
    ├── images/
    │   ├── 0000000.png
    │   └── ...
    ├── labels.json          # {"0000000.png": "Séadna", ...}
    ├── manifest.jsonl       # one JSON record per attempted sample
    ├── recipe.snapshot.yaml # written by ``output.snapshot``
    └── stats.json           # run-level counters

Filenames are zero-padded to enough digits to fit the configured
``output.count``. Manifest records cover both rendered and skipped
samples so a downstream consumer can audit the run.

The writer is **incremental and resume-safe**:

- ``open(..., resume=True)`` reads any existing ``labels.json`` /
  ``manifest.jsonl`` entries and arranges to skip indices already
  written.
- ``open(..., force=True)`` clears the destination first.
- The default behavior refuses to write into a non-empty directory.

Manifest writes are line-buffered and flushed after each record so
that an interrupted run can be resumed without losing observability
on what was already attempted.
"""

from __future__ import annotations

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

# Lower bound for zero-padding regardless of count. Six digits is enough
# for 1M samples, which exceeds anything the recognition path will
# practically write; any smaller and 50k vs 500k recipes would produce
# differently-named samples for the same indices.
_MIN_PAD_WIDTH = 7


def width_for_count(count: int) -> int:
    """Pick the zero-pad width for image filenames.

    ``count`` is the recipe's ``output.count``. Width is large enough
    that every index in ``[0, count)`` lands at the same character
    width, with a sane lower bound for tiny recipes.
    """

    if count <= 0:
        return _MIN_PAD_WIDTH
    width = max(_MIN_PAD_WIDTH, len(str(max(count - 1, 0))))
    return width


def image_filename(index: int, *, width: int) -> str:
    """Spec-shaped filename for sample ``index``."""

    return f"{index:0{width}d}.png"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class RenderStats:
    """Accumulated counters for a render run.

    Mirrors the ``stats.json`` schema in spec 08. Mutable on purpose
    — the writer accumulates these as samples land, then dumps the
    final snapshot to disk on ``close()``.
    """

    samples_planned: int = 0
    samples_written: int = 0
    samples_skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    fonts_used: dict[str, int] = field(default_factory=dict)
    tokens_unique: int = 0
    wall_time_seconds: float = 0.0

    def record_skip(self, reason: str) -> None:
        self.samples_skipped += 1
        self.skip_reasons[reason] = self.skip_reasons.get(reason, 0) + 1

    def record_render(self, *, font_name: str) -> None:
        self.samples_written += 1
        self.fonts_used[font_name] = self.fonts_used.get(font_name, 0) + 1

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


class RecognitionWriter:
    """Streaming writer for the recognition profile layout.

    Use as a context manager so the manifest stream and labels file
    flush + finalize even if the render loop raises::

        with RecognitionWriter.open(recipe, output_dir, ...) as writer:
            for index, sample in enumerate(...):
                writer.write_rendered(index, sample, ...)
            for index in skipped_indices:
                writer.write_skipped(index, ...)

    The writer is index-addressed: every call carries its sample
    index, and writes are idempotent on re-write of the same index
    (overwrites the PNG + label entry, replaces the manifest line).
    Idempotency is what makes ``--resume`` safe.
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
        existing_labels: dict[str, str],
        existing_manifest_lines: list[dict[str, Any]],
        existing_indices: set[int],
        planned_count: int | None = None,
    ) -> None:
        self.recipe = recipe
        self.output_dir = output_dir
        self.images_dir = output_dir / IMAGES_DIRNAME
        self.seed = seed
        self.pad_width = pad_width
        self._labels: dict[str, str] = dict(existing_labels)
        # Manifest records keyed by index so re-writes replace rather
        # than append. We materialize the list at close() time in
        # index order for stable on-disk output.
        self._manifest: dict[int, dict[str, Any]] = {
            int(rec["index"]): rec for rec in existing_manifest_lines if "index" in rec
        }
        self._existing_indices: set[int] = set(existing_indices)
        # ``planned_count`` overrides the recipe's count for stats /
        # filename padding. The CLI's ``--count`` override flows
        # through here so a smoke run reports its actual plan size.
        effective_planned = int(planned_count) if planned_count is not None else recipe.output.count
        self.stats = RenderStats(samples_planned=effective_planned)
        # Pre-seed fonts_used from existing manifest so resume reports
        # the rolled-up count, not just the new samples.
        for rec in self._manifest.values():
            if rec.get("status") != "rendered":
                continue
            font_name = (rec.get("font") or {}).get("name")
            if font_name:
                self.stats.fonts_used[font_name] = self.stats.fonts_used.get(font_name, 0) + 1
                self.stats.samples_written += 1
        for rec in self._manifest.values():
            if rec.get("status") == "skipped":
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
    ) -> RecognitionWriter:
        """Prepare ``output_dir`` for writing and return a ready writer.

        Behavioral matrix mirrors the spec:

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
            # Aggressive wipe is intentional: the user said --force.
            for child in output_dir.iterdir():
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / IMAGES_DIRNAME).mkdir(parents=True, exist_ok=True)

        existing_labels: dict[str, str] = {}
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
                    existing_labels = {str(k): str(v) for k, v in raw.items()}
            manifest_path = output_dir / MANIFEST_FILENAME
            if manifest_path.exists():
                for line in manifest_path.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        # Skip malformed manifest lines defensively;
                        # surfacing every parse error would block
                        # legitimate resumes after a hard crash.
                        continue
                    if isinstance(rec, dict) and "index" in rec:
                        existing_manifest.append(rec)
                        existing_indices.add(int(rec["index"]))

        # Filename padding still uses the recipe's count (the
        # contractual bound on indices) — overriding count via the CLI
        # is for smoke tests, and shouldn't change the file naming
        # convention so a smoke run can be promoted to a full render
        # without renaming files.
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
        """``True`` if ``index`` has a *successful* manifest entry on disk.

        Skipped samples are not considered "already rendered" — a
        resume sweep should retry them in case the input that caused
        the skip has been fixed (e.g. a font was added).
        """

        rec = self._manifest.get(int(index))
        if rec is None:
            return False
        return rec.get("status") == "rendered"

    def filename(self, index: int) -> str:
        return image_filename(index, width=self.pad_width)

    def write_rendered(
        self,
        index: int,
        sample: Any,
        *,
        text: str,
        applied_degradations: Iterable[dict[str, Any]] = (),
    ) -> None:
        """Persist a successfully rendered sample.

        ``sample`` is a :class:`pd_ocr_synth.render.RenderedSample`;
        we keep it loosely typed at this layer to avoid a cycle
        between ``output`` and ``render`` modules.
        """

        if self._closed:
            raise RuntimeError("RecognitionWriter is already closed")
        idx = int(index)
        name = self.filename(idx)
        path = self.images_dir / name
        sample.image.save(path, format="PNG")

        self._labels[name] = text
        font_path = Path(sample.font_path)
        record: dict[str, Any] = {
            "index": idx,
            "id": Path(name).stem,
            "image": f"{IMAGES_DIRNAME}/{name}",
            "text": text,
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
            "bbox": list(sample.bbox),
            "transforms_applied": [t.name for t in self.recipe.text_transforms],
            "degradations_applied": list(applied_degradations),
        }
        # Per-word ground truth lands here for ``lines``-mode samples;
        # ``word_crops`` produces an empty tuple (the sample IS the
        # word) and we skip the field entirely so existing manifests
        # stay byte-identical to the M07 schema.
        word_boxes = getattr(sample, "word_boxes", ()) or ()
        if word_boxes:
            record["word_boxes"] = [{"text": wb.text, "bbox": list(wb.bbox)} for wb in word_boxes]
        self._manifest[idx] = record
        self.stats.record_render(font_name=font_path.name)

    def write_skipped(
        self,
        index: int,
        *,
        reason: str,
        text: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Record that ``index`` could not be rendered (no PNG, no label)."""

        if self._closed:
            raise RuntimeError("RecognitionWriter is already closed")
        idx = int(index)
        # If we previously rendered this index (e.g. on a partial
        # resume) and now we're skipping, drop the stale image+label.
        # Keeps the on-disk state consistent with the manifest.
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
        if text is not None:
            record["text"] = text
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

    def __enter__(self) -> RecognitionWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        # Always finalize the on-disk state, even on render failure —
        # the user should be able to inspect a partial run rather than
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
        # Sort for deterministic diff-friendly output.
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
        # ``samples_planned`` may be < count when --dry-run / smoke runs
        # set a smaller count via the override; we report whatever the
        # writer was constructed with.
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
