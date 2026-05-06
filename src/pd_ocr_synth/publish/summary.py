"""Summary helper over a staged HF imagefolder ``metadata.jsonl``.

Per ``docs/specs/10-publishing.md`` § Dry run, the upcoming
``pd-ocr-synth publish --dry-run`` surface needs to print a short
recap of what the staging dir holds before any HF call is made:

    Files: 50003 (50000 .png + metadata.jsonl + README.md + ...)
    Total size: 247.3 MB
    Dataset card preview:
      ...
    Content SHA: 2c4f8e... (no existing commit; first publish)

The total-files / total-bytes lines come from a directory walk; the
dataset-card preview comes from the README; the content-SHA is
already plumbed through ``recognition.build_recognition_staging``.

What's still missing — and what this module supplies — is a structured
*summary of the metadata rows themselves* so the dry-run output can
say "5000 rows, 6 fonts, 12 degradation stages, 3 corpus providers"
without re-reading the file. This is also a prerequisite for the
``docs/roadmap/08-publishing-hf.md`` deliverable that asks for a more
detailed dataset-card body (font / size / degradation coverage tables
land in M08's later chunks).

Design choices
--------------
- **Staging input, not local input.** The staging-dir
  ``metadata.jsonl`` is the canonical, flattened, upload-shaped record
  of what's about to ship. Summarizing the local ``manifest.jsonl``
  would double-up provenance work the staging builder already did, and
  would skew counts when the local dir contains rows whose images were
  later dropped.
- **Pure file-IO.** Same constraint as the rest of the publish chunks:
  no ``huggingface_hub`` import, no network, no recipe object. Tests
  run in milliseconds.
- **Tolerant of older / partial files.** A row missing a column
  contributes to the row count but not to the missing column's
  histogram. We never raise on partial rows; we surface the count of
  *unparseable* lines via the report so the caller can decide whether
  to flag a corrupt staging dir.
- **Separate "compute" from "format".** :func:`summarize_metadata`
  returns the structured :class:`ManifestSummary`; :func:`format_summary`
  turns it into the human string the dry-run will print. That split
  lets the upcoming dataset-card refinement reuse the structured
  numbers without re-parsing the formatted text.

Counter ordering
----------------
Counters are returned as ``list[tuple[str, int]]`` rather than
``Counter`` so the order is stable: most-common first, ties broken
alphabetically. The dry-run prints these and human readers benefit
from a deterministic order across runs.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pd_ocr_synth.publish.recognition import METADATA_FILENAME


@dataclass(frozen=True, slots=True)
class ManifestSummary:
    """Aggregate counts derived from a staging ``metadata.jsonl``.

    Every histogram is a list of ``(value, count)`` pairs sorted by
    descending count then ascending value. Empty sequences mean the
    column was absent from every row (which is allowed for the
    optional provenance columns — only ``file_name`` / ``text`` are
    mandatory in the row schema).

    Attributes
    ----------
    rows:
        Number of well-formed JSON rows read from ``metadata.jsonl``.
        A row counts toward this total iff it parsed as a JSON object
        (mapping); malformed lines are tracked separately in
        :attr:`unparseable_lines`.
    fonts:
        Distinct font identifiers (the value of the ``font`` column),
        with frequency. Sorted (count desc, value asc).
    font_sizes_pt:
        Distinct point sizes (``font_size_pt`` column), with frequency.
        Sized are floats but printed as ``f"{size:g}"`` so 14.0 shows
        as ``14`` and 14.5 keeps its decimal — see
        :func:`format_summary`.
    degradations:
        Distinct degradation *stage names* across the union of every
        ``degradations`` list. A row that lists three stages adds three
        increments. This matches the trainer's "did this dataset
        contain stage X?" question, not "how many rows used X
        exclusively?".
    corpora:
        Distinct values of the flat ``corpus`` column.
    rows_missing_text:
        Rows whose ``text`` column was absent or empty after stripping.
        Surfaced rather than raised so the dry-run can warn loudly
        without aborting; an empty label is a render bug, not a
        publish bug.
    unparseable_lines:
        Lines that didn't parse as a JSON object. Tracked so a stray
        BOM / partial flush doesn't get swept under the rug.
    """

    rows: int
    fonts: list[tuple[str, int]] = field(default_factory=list)
    font_sizes_pt: list[tuple[float, int]] = field(default_factory=list)
    degradations: list[tuple[str, int]] = field(default_factory=list)
    corpora: list[tuple[str, int]] = field(default_factory=list)
    rows_missing_text: int = 0
    unparseable_lines: int = 0

    @property
    def font_count(self) -> int:
        """Distinct fonts seen across the manifest."""

        return len(self.fonts)

    @property
    def degradation_count(self) -> int:
        """Distinct degradation stage names seen across the manifest."""

        return len(self.degradations)

    @property
    def corpus_count(self) -> int:
        """Distinct corpus identifiers seen across the manifest."""

        return len(self.corpora)


class SummaryError(Exception):
    """Raised when the staging metadata file is missing or unreadable.

    Distinct from per-row issues (which are counted, not raised) so
    callers can map a hard "no metadata at all" failure to a clean
    exit code without having to parse error strings.
    """


def summarize_metadata(staging_dir: Path) -> ManifestSummary:
    """Aggregate counts over ``<staging_dir>/metadata.jsonl``.

    Parameters
    ----------
    staging_dir:
        Built staging directory. Must contain ``metadata.jsonl``;
        anything else (images, README, snapshot) is ignored here —
        callers that want a *file count* walk the dir themselves.

    Returns
    -------
    ManifestSummary
        Populated counters. Always returned; a metadata file with zero
        rows yields a summary with ``rows=0`` and empty histograms,
        not an exception.

    Raises
    ------
    SummaryError
        If ``metadata.jsonl`` is missing entirely. (A truly empty file
        is fine — that's a legitimate "zero-row staging" outcome the
        dry-run might want to flag, but not abort on.)
    """

    metadata_path = Path(staging_dir) / METADATA_FILENAME
    if not metadata_path.is_file():
        raise SummaryError(
            f"staging dir {staging_dir} is missing {METADATA_FILENAME}; "
            "did `build_recognition_staging` run to completion?"
        )

    rows = 0
    rows_missing_text = 0
    unparseable = 0
    fonts: Counter[str] = Counter()
    sizes: Counter[float] = Counter()
    degradations: Counter[str] = Counter()
    corpora: Counter[str] = Counter()

    for line in metadata_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            unparseable += 1
            continue
        if not isinstance(record, dict):
            unparseable += 1
            continue

        rows += 1

        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            rows_missing_text += 1

        font = record.get("font")
        if isinstance(font, str) and font:
            fonts[font] += 1

        size = record.get("font_size_pt")
        # JSON numerics arrive as int / float — normalize to float so
        # ``14`` and ``14.0`` collapse onto one bucket. Reject other
        # types silently rather than raise; the schema is tolerated to
        # be loose here (see module docstring).
        if isinstance(size, (int, float)) and not isinstance(size, bool):
            sizes[float(size)] += 1

        degs = record.get("degradations")
        if isinstance(degs, list):
            for stage in degs:
                if isinstance(stage, str) and stage:
                    degradations[stage] += 1

        corpus = record.get("corpus")
        if isinstance(corpus, str) and corpus:
            corpora[corpus] += 1

    return ManifestSummary(
        rows=rows,
        fonts=_sorted_counter(fonts),
        font_sizes_pt=_sorted_counter(sizes),
        degradations=_sorted_counter(degradations),
        corpora=_sorted_counter(corpora),
        rows_missing_text=rows_missing_text,
        unparseable_lines=unparseable,
    )


def format_summary(summary: ManifestSummary, *, max_items: int = 10) -> str:
    """Render a human-readable multi-line block of the summary.

    Format mirrors what ``--dry-run`` will print. Each section caps at
    ``max_items`` entries so a 50-font dataset doesn't drown the
    terminal; an "(+N more)" line gets appended when truncation
    happens. ``rows_missing_text`` / ``unparseable_lines`` show up as
    warnings only when non-zero so the common-case output stays
    tight.

    Parameters
    ----------
    summary:
        The structured summary to render.
    max_items:
        Per-histogram truncation cap. Must be positive; a non-positive
        value falls back to ``1`` so the function never silently
        produces an empty section.
    """

    cap = max(1, max_items)
    lines: list[str] = []
    lines.append(f"Rows: {summary.rows}")

    lines.append(
        f"Fonts: {summary.font_count} distinct"
        + (" (none recorded)" if summary.font_count == 0 else "")
    )
    _append_histogram(lines, summary.fonts, cap=cap, formatter=str)

    lines.append(
        f"Font sizes (pt): {len(summary.font_sizes_pt)} distinct"
        + (" (none recorded)" if not summary.font_sizes_pt else "")
    )
    _append_histogram(lines, summary.font_sizes_pt, cap=cap, formatter=_format_size)

    lines.append(
        f"Degradations: {summary.degradation_count} distinct"
        + (" (none recorded)" if summary.degradation_count == 0 else "")
    )
    _append_histogram(lines, summary.degradations, cap=cap, formatter=str)

    lines.append(
        f"Corpora: {summary.corpus_count} distinct"
        + (" (none recorded)" if summary.corpus_count == 0 else "")
    )
    _append_histogram(lines, summary.corpora, cap=cap, formatter=str)

    if summary.rows_missing_text:
        lines.append(f"WARNING: {summary.rows_missing_text} row(s) missing text")
    if summary.unparseable_lines:
        lines.append(f"WARNING: {summary.unparseable_lines} unparseable line(s) in metadata.jsonl")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sorted_counter(counter: Counter[Any]) -> list[tuple[Any, int]]:
    """Stable sort: most-common first, alphabetic-by-value tiebreak.

    ``Counter.most_common`` is stable on insertion order, which would
    leak the input file's row order into the output. We want a
    deterministic sort that humans diffing two summaries can read.
    """

    return sorted(counter.items(), key=lambda kv: (-kv[1], _sort_key(kv[0])))


def _sort_key(value: Any) -> Any:
    """Map heterogeneous histogram keys to a comparable tiebreak key.

    Floats go through unchanged so the size histogram tiebreaks
    numerically; strings tiebreak alphabetically. ``str()`` on every
    other type is the safe fallback — the histograms only ever carry
    strings or floats today, so this branch is defensive.
    """

    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, str(value))


def _append_histogram(
    lines: list[str],
    items: Iterable[tuple[Any, int]],
    *,
    cap: int,
    formatter: Any,
) -> None:
    """Append ``cap`` entries from ``items`` then a "(+N more)" line."""

    materialized = list(items)
    for value, count in materialized[:cap]:
        lines.append(f"  {formatter(value)}: {count}")
    overflow = len(materialized) - cap
    if overflow > 0:
        lines.append(f"  (+{overflow} more)")


def _format_size(size: float) -> str:
    """Render a font size cleanly: ``14.0`` → ``14``, ``14.5`` → ``14.5``."""

    return f"{size:g}"
