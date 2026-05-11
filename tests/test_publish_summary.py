"""Unit tests for the manifest-summary helper (M08).

Covers ``pd_ocr_synth.publish.summary``: a pure file-IO aggregator
over the staging-dir ``metadata.jsonl`` that the upcoming
``pd-ocr-synth publish --dry-run`` will print, and that the eventual
dataset-card refinement will reuse for coverage tables.

These tests treat the public surface (``summarize_metadata``,
``format_summary``, ``ManifestSummary``, ``SummaryError``) as the
contract and lock specific behaviors:

- Histograms are sorted by count desc, value asc (stable across runs).
- Heterogeneous JSON numerics for ``font_size_pt`` collapse onto one
  bucket regardless of int / float on the wire.
- Per-row issues (missing ``text``, malformed lines) are *counted*,
  not raised — the dry-run can warn but must still run.
- A missing ``metadata.jsonl`` raises :class:`SummaryError`; a
  zero-row file does not.

A round-trip test against the real ``build_recognition_staging``
pins the M08-internal contract: whatever the staging writer emits,
the summary helper must agree with on row count.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from pd_ocr_synth.publish import (
    METADATA_FILENAME,
    ManifestSummary,
    SummaryError,
    build_recognition_staging,
    format_summary,
    summarize_metadata,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_metadata(staging: Path, rows: list[dict] | list[str]) -> Path:
    """Materialize a minimal staging dir with a ``metadata.jsonl`` only.

    Most summary tests don't care about the rest of the staging
    layout — the helper reads one file. Mixing structured rows and
    raw strings (for unparseable-line cases) keeps the fixture
    helper compact.
    """

    staging.mkdir(parents=True, exist_ok=True)
    path = staging / METADATA_FILENAME
    lines: list[str] = []
    for row in rows:
        if isinstance(row, str):
            lines.append(row)
        else:
            lines.append(json.dumps(row, ensure_ascii=False))
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Happy path — structural counts and ordering
# ---------------------------------------------------------------------------


def test_summary_counts_rows_and_distincts(tmp_path: Path) -> None:
    """Basic aggregation: row count + distinct cardinalities for each column."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            {
                "file_name": "data/0000000.png",
                "text": "Séadna",
                "font": "bungc/bungc.otf",
                "font_size_pt": 14.0,
                "degradations": ["skew", "jpeg"],
                "corpus": "wikisource:Séadna",
            },
            {
                "file_name": "data/0000001.png",
                "text": "Aithris",
                "font": "bungc/bungc.otf",
                "font_size_pt": 14.0,
                "degradations": ["jpeg"],
                "corpus": "wikisource:Séadna",
            },
            {
                "file_name": "data/0000002.png",
                "text": "Conas",
                "font": "seangc/seangc.otf",
                "font_size_pt": 16.0,
                "degradations": ["paper_texture", "jpeg"],
                "corpus": "celt:G100002",
            },
        ],
    )

    summary = summarize_metadata(staging)
    assert isinstance(summary, ManifestSummary)
    assert summary.rows == 3
    assert summary.font_count == 2
    assert summary.degradation_count == 3
    assert summary.corpus_count == 2
    assert summary.rows_missing_text == 0
    assert summary.unparseable_lines == 0


def test_histograms_sort_by_count_desc_then_value_asc(tmp_path: Path) -> None:
    """Order is deterministic across runs: count desc, value asc on ties."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            # Insertion order deliberately scrambled so a Counter-most-common
            # default (which preserves insertion) would fail this test.
            {"file_name": "data/0.png", "text": "a", "font": "z-font"},
            {"file_name": "data/1.png", "text": "a", "font": "a-font"},
            {"file_name": "data/2.png", "text": "a", "font": "z-font"},
            {"file_name": "data/3.png", "text": "a", "font": "m-font"},
            {"file_name": "data/4.png", "text": "a", "font": "a-font"},
            {"file_name": "data/5.png", "text": "a", "font": "z-font"},
            {"file_name": "data/6.png", "text": "a", "font": "m-font"},
        ],
    )

    summary = summarize_metadata(staging)
    # z-font: 3, a-font: 2, m-font: 2 — m and a tie on count, alphabetical
    # tiebreak puts a-font first.
    assert summary.fonts == [("z-font", 3), ("a-font", 2), ("m-font", 2)]


def test_font_size_int_and_float_collapse_onto_one_bucket(tmp_path: Path) -> None:
    """JSON numerics are normalized to ``float`` so 14 / 14.0 don't split."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            {"file_name": "data/0.png", "text": "a", "font_size_pt": 14},
            {"file_name": "data/1.png", "text": "a", "font_size_pt": 14.0},
            {"file_name": "data/2.png", "text": "a", "font_size_pt": 14.5},
        ],
    )

    summary = summarize_metadata(staging)
    assert summary.font_sizes_pt == [(14.0, 2), (14.5, 1)]


def test_degradations_count_per_stage_not_per_row(tmp_path: Path) -> None:
    """A row that lists three stages adds three increments, one per stage."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            {"file_name": "data/0.png", "text": "a", "degradations": ["skew", "jpeg"]},
            {"file_name": "data/1.png", "text": "a", "degradations": ["jpeg"]},
            {
                "file_name": "data/2.png",
                "text": "a",
                "degradations": ["skew", "jpeg", "paper_texture"],
            },
        ],
    )

    summary = summarize_metadata(staging)
    # jpeg: 3, skew: 2, paper_texture: 1
    assert summary.degradations == [("jpeg", 3), ("skew", 2), ("paper_texture", 1)]


# ---------------------------------------------------------------------------
# Tolerated-shape edge cases — counted, not raised
# ---------------------------------------------------------------------------


def test_rows_missing_text_are_counted_but_not_raised(tmp_path: Path) -> None:
    """Missing / empty ``text`` is a render bug; surface it without aborting."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            {"file_name": "data/0.png", "text": "Séadna"},
            {"file_name": "data/1.png", "text": ""},
            {"file_name": "data/2.png", "text": "   "},
            {"file_name": "data/3.png"},  # text key missing entirely
        ],
    )

    summary = summarize_metadata(staging)
    # All four rows still count toward ``rows`` — we want the dry-run
    # to surface the discrepancy via ``rows_missing_text``, not hide
    # rows from the total.
    assert summary.rows == 4
    assert summary.rows_missing_text == 3


def test_unparseable_lines_are_counted_separately(tmp_path: Path) -> None:
    """Malformed JSON / non-object rows don't count toward ``rows``."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            {"file_name": "data/0.png", "text": "ok"},
            "not even json",
            "[1, 2, 3]",  # parses but isn't a mapping
            "",  # empty lines are stripped, not counted
            {"file_name": "data/1.png", "text": "ok"},
        ],
    )

    summary = summarize_metadata(staging)
    assert summary.rows == 2
    assert summary.unparseable_lines == 2


def test_optional_columns_absent_yield_empty_histograms(tmp_path: Path) -> None:
    """Rows without provenance columns produce empty per-column lists."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            {"file_name": "data/0.png", "text": "a"},
            {"file_name": "data/1.png", "text": "b"},
        ],
    )

    summary = summarize_metadata(staging)
    assert summary.rows == 2
    assert summary.fonts == []
    assert summary.font_sizes_pt == []
    assert summary.degradations == []
    assert summary.corpora == []
    assert summary.font_count == 0
    assert summary.degradation_count == 0
    assert summary.corpus_count == 0


def test_wrong_typed_columns_are_skipped_silently(tmp_path: Path) -> None:
    """Defensive: a future schema drift won't crash the summary helper."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            {
                "file_name": "data/0.png",
                "text": "a",
                "font": 42,  # wrong type — ignored
                "font_size_pt": "fourteen",  # wrong type — ignored
                "degradations": "jpeg",  # wrong type (should be list) — ignored
                "corpus": ["wikisource", "Séadna"],  # wrong type — ignored
            }
        ],
    )

    summary = summarize_metadata(staging)
    assert summary.rows == 1
    assert summary.fonts == []
    assert summary.font_sizes_pt == []
    assert summary.degradations == []
    assert summary.corpora == []


def test_boolean_font_size_is_rejected(tmp_path: Path) -> None:
    """``True``/``False`` are subclasses of ``int``; don't let them slip in."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [{"file_name": "data/0.png", "text": "a", "font_size_pt": True}],
    )

    summary = summarize_metadata(staging)
    assert summary.font_sizes_pt == []


def test_zero_row_metadata_is_a_legit_summary(tmp_path: Path) -> None:
    """Empty metadata file → empty summary, *not* ``SummaryError``."""

    staging = tmp_path / "staging"
    _write_metadata(staging, [])

    summary = summarize_metadata(staging)
    assert summary.rows == 0
    assert summary.fonts == []
    assert summary.unparseable_lines == 0


# ---------------------------------------------------------------------------
# Hard failures
# ---------------------------------------------------------------------------


def test_missing_metadata_file_raises_typed_error(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()

    with pytest.raises(SummaryError) as excinfo:
        summarize_metadata(staging)
    # Error must name the missing filename so it's grep-able from a CI log.
    assert METADATA_FILENAME in str(excinfo.value)


# ---------------------------------------------------------------------------
# format_summary — the human-readable block
# ---------------------------------------------------------------------------


def test_format_summary_lists_headline_counts_and_top_buckets(tmp_path: Path) -> None:
    """Format output covers each section and reports distinct counts."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            {
                "file_name": "data/0.png",
                "text": "a",
                "font": "bungc",
                "font_size_pt": 14.0,
                "degradations": ["jpeg"],
                "corpus": "wikisource:x",
            },
            {
                "file_name": "data/1.png",
                "text": "b",
                "font": "bungc",
                "font_size_pt": 14.0,
                "degradations": ["jpeg", "skew"],
                "corpus": "wikisource:x",
            },
        ],
    )

    summary = summarize_metadata(staging)
    text = format_summary(summary)

    assert "Rows: 2" in text
    assert "Fonts: 1 distinct" in text
    assert "  bungc: 2" in text
    # 14.0 should render as "14" via ``{:g}``.
    assert "  14: 2" in text
    assert "Degradations: 2 distinct" in text
    assert "  jpeg: 2" in text
    assert "  skew: 1" in text
    assert "Corpora: 1 distinct" in text
    assert "  wikisource:x: 2" in text


def test_format_summary_truncates_long_histograms(tmp_path: Path) -> None:
    """``max_items`` caps each section and prints an "(+N more)" line."""

    staging = tmp_path / "staging"
    rows = [{"file_name": f"data/{i}.png", "text": "a", "font": f"font-{i:02d}"} for i in range(15)]
    _write_metadata(staging, rows)

    summary = summarize_metadata(staging)
    text = format_summary(summary, max_items=3)

    # Three entries shown (we can't assume which three without computing
    # the sort, but they all end in `: 1`), one truncation line.
    assert "(+12 more)" in text
    # Sanity: at least three font-* lines.
    assert text.count("font-") >= 3


def test_format_summary_omits_warnings_when_clean(tmp_path: Path) -> None:
    """Common-case output stays tight: no warning lines when counts are zero."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [{"file_name": "data/0.png", "text": "a"}],
    )
    summary = summarize_metadata(staging)
    text = format_summary(summary)
    assert "WARNING" not in text


def test_format_summary_emits_warnings_for_dirty_data(tmp_path: Path) -> None:
    """Both warning lines fire when their respective counters are non-zero."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            {"file_name": "data/0.png", "text": ""},
            "not json",
        ],
    )
    summary = summarize_metadata(staging)
    text = format_summary(summary)
    assert "1 row(s) missing text" in text
    assert "1 unparseable line(s)" in text


def test_format_summary_handles_no_recorded_columns(tmp_path: Path) -> None:
    """``(none recorded)`` annotation makes empty sections explicit."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [{"file_name": "data/0.png", "text": "a"}],
    )
    summary = summarize_metadata(staging)
    text = format_summary(summary)
    # Each optional column reports "(none recorded)".
    assert "Fonts: 0 distinct (none recorded)" in text
    assert "Font sizes (pt): 0 distinct (none recorded)" in text
    assert "Degradations: 0 distinct (none recorded)" in text
    assert "Corpora: 0 distinct (none recorded)" in text


def test_format_summary_clamps_non_positive_max_items(tmp_path: Path) -> None:
    """``max_items=0`` falls back to 1 so the section isn't silently dropped."""

    staging = tmp_path / "staging"
    _write_metadata(
        staging,
        [
            {"file_name": "data/0.png", "text": "a", "font": "f1"},
            {"file_name": "data/1.png", "text": "a", "font": "f2"},
            {"file_name": "data/2.png", "text": "a", "font": "f3"},
        ],
    )
    summary = summarize_metadata(staging)
    text = format_summary(summary, max_items=0)
    # One font row + one truncation line for the remaining two.
    assert "(+2 more)" in text


# ---------------------------------------------------------------------------
# Round-trip: real staging build → summary
# ---------------------------------------------------------------------------


def test_summary_against_real_staging_build(tmp_path: Path) -> None:
    """Lock the M08-internal contract: staging writer + summary agree.

    Produces a tiny but real local recognition layout, runs
    ``build_recognition_staging`` over it, then summarizes the
    resulting ``metadata.jsonl``. The row count must equal the
    images-copied count from the writer.
    """

    local = tmp_path / "local"
    images = local / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(180, 180, 180)).save(images / "0000000.png", format="PNG")
    Image.new("RGB", (8, 8), color=(170, 170, 170)).save(images / "0000001.png", format="PNG")
    (local / "labels.json").write_text(
        '{"0000000.png": "Séadna", "0000001.png": "Aithris"}\n', encoding="utf-8"
    )
    (local / "manifest.jsonl").write_text(
        '{"image": "images/0000000.png", "text": "Séadna", "status": "ok",'
        ' "font": {"name": "bungc.otf", "size_pt": 14.0},'
        ' "degradations_applied": [{"kind": "jpeg"}],'
        ' "corpus": {"provider": "wikisource", "key": "Séadna"}}\n'
        '{"image": "images/0000001.png", "text": "Aithris", "status": "ok",'
        ' "font": {"name": "seangc.otf", "size_pt": 16.0},'
        ' "degradations_applied": [{"kind": "skew"}, {"kind": "jpeg"}],'
        ' "corpus": {"provider": "wikisource", "key": "Aithris"}}\n',
        encoding="utf-8",
    )
    snapshot_yaml = (
        "tool_version: 0.1.2\n"
        "seed: 0\n"
        "recipe:\n"
        "  schema_version: 1\n"
        "  name: gaelic\n"
        "  seed: 0\n"
        "  fonts:\n"
        "    - path: /abs/fonts/bungc.otf\n"
        "  corpus: []\n"
        "  publish:\n"
        "    hf_dataset:\n"
        "      repo: ntw8532/pd-ocr-synth-gaelic\n"
        "      license: cc-by-4.0\n"
        "      language: [ga]\n"
        "      tags: [ocr, gaelic]\n"
        "input_hashes: {}\n"
    )
    (local / "recipe.snapshot.yaml").write_text(snapshot_yaml, encoding="utf-8")

    staging = tmp_path / "staging"
    result = build_recognition_staging(local, staging)
    summary = summarize_metadata(staging)

    # The staging writer's "images_copied" is the upstream row count;
    # the summary must agree exactly.
    assert summary.rows == result.images_copied == 2
    # Font / degradation counts come from the manifest provenance the
    # staging writer flattened into ``metadata.jsonl``.
    assert summary.font_count == 2
    assert summary.degradation_count == 2  # skew + jpeg
    # Corpus column is present and joined to two values.
    assert summary.corpus_count == 2
    assert summary.rows_missing_text == 0
    assert summary.unparseable_lines == 0
