"""CLI tests for ``pd-ocr-synth audit`` (M10 stretch).

Exercises the read-side of the per-render audit log written by
``render`` (see iter 45 / ``src/pd_ocr_synth/audit.py``). The write
path is already covered by ``tests/test_cli_render_audit.py`` and
``tests/test_audit.py``; this file only asserts on the read-back
subcommand:

- table-mode default (header + one row per entry, short SHA);
- ``--json`` mode (machine-readable, schema verbatim);
- ``--limit N`` tails to the most recent N rows;
- usage error for non-positive ``--limit`` (exit 2);
- destination-family errors (exit 6) for missing dir / missing file;
- empty audit file emits header + ``(no audit entries)`` and returns 0.

We construct the audit JSONL by hand using the public
``append_audit_entry`` API rather than driving an end-to-end render —
that's the right grain: the integration tests already cover the
write path, so this layer can stay fast and font-independent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pd_ocr_synth.audit import (
    AUDIT_FILENAME,
    AUDIT_SCHEMA_VERSION,
    AuditEntry,
    append_audit_entry,
)
from pd_ocr_synth.cli import main


def _make_entry(
    *,
    timestamp: str = "2026-05-06T01:23:45Z",
    recipe_name: str = "gaelic",
    recipe_sha: str | None = "abcdef0123456789" * 4,
    output_dir: str = "/tmp/out",
    count: int = 100,
    seed: int = 42,
    workers: int = 4,
    rendered: int = 95,
    skipped: int = 5,
    runtime_seconds: float = 12.34,
) -> AuditEntry:
    return AuditEntry(
        timestamp=timestamp,
        recipe_name=recipe_name,
        recipe_sha=recipe_sha,
        output_dir=output_dir,
        count=count,
        seed=seed,
        workers=workers,
        rendered=rendered,
        skipped=skipped,
        runtime_seconds=runtime_seconds,
    )


def _seed_audit(out: Path, entries: list[AuditEntry]) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    audit_path = out / AUDIT_FILENAME
    for entry in entries:
        append_audit_entry(audit_path, entry)
    return audit_path


def test_audit_table_default_renders_one_row_per_entry(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp="2026-05-06T01:00:00Z", seed=1, count=10, rendered=10, skipped=0),
            _make_entry(timestamp="2026-05-06T02:00:00Z", seed=2, count=20, rendered=18, skipped=2),
        ],
    )

    rc = main(["audit", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    stdout = captured.out
    # Header + separator + 2 data rows = 4 non-empty lines minimum.
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == 4
    assert lines[0].startswith("timestamp")
    # Short SHA should be 8 hex chars; the seeded SHA above is 64 chars
    # of ``abcdef0123456789`` repeated, so the prefix is ``abcdef01``.
    assert "abcdef01" in stdout
    assert "2026-05-06T01:00:00Z" in stdout
    assert "2026-05-06T02:00:00Z" in stdout
    # Recipe name surfaces in each row.
    assert stdout.count("gaelic") == 2


def test_audit_json_mode_emits_full_schema(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp="2026-05-06T01:00:00Z", seed=1),
            _make_entry(timestamp="2026-05-06T02:00:00Z", seed=2),
            _make_entry(timestamp="2026-05-06T03:00:00Z", seed=3),
        ],
    )

    rc = main(["audit", str(out), "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert isinstance(payload, list)
    assert len(payload) == 3
    # Schema-version present on every row → forward-compat readers can
    # branch on it.
    assert all(entry["schema_version"] == AUDIT_SCHEMA_VERSION for entry in payload)
    # Order preserved (oldest first).
    assert [entry["seed"] for entry in payload] == [1, 2, 3]
    # Full schema visible (not just the table-projected subset).
    keys = set(payload[0].keys())
    assert {
        "timestamp",
        "recipe_name",
        "recipe_sha",
        "output_dir",
        "count",
        "seed",
        "workers",
        "rendered",
        "skipped",
        "runtime_seconds",
        "schema_version",
    } <= keys


def test_audit_limit_tails_to_most_recent(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [_make_entry(timestamp=f"2026-05-06T0{i}:00:00Z", seed=i) for i in range(1, 6)],
    )

    rc = main(["audit", str(out), "--json", "--limit", "2"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # Tail: keeps the *last* two, drops the older three.
    assert [entry["seed"] for entry in payload] == [4, 5]


def test_audit_limit_non_positive_is_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(out, [_make_entry()])

    rc = main(["audit", str(out), "--limit", "0"])
    captured = capsys.readouterr()
    assert rc == 2  # USAGE_EXIT per docs/specs/01-cli.md
    assert "limit" in captured.err.lower()


def test_audit_missing_output_dir_is_destination_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["audit", str(tmp_path / "does-not-exist")])
    captured = capsys.readouterr()
    assert rc == 6  # DESTINATION_EXIT per docs/specs/01-cli.md
    assert "does not exist" in captured.err


def test_audit_missing_file_is_destination_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Output dir exists but no audit file → exit 6 with hint."""

    out = tmp_path / "render-out"
    out.mkdir()

    rc = main(["audit", str(out)])
    captured = capsys.readouterr()
    assert rc == 6
    assert "no audit file" in captured.err
    # Hint at the most likely cause (the user disabled audit).
    assert "--no-audit" in captured.err or "PD_OCR_SYNTH_NO_AUDIT" in captured.err


def test_audit_empty_file_emits_header_and_returns_zero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An audit file that exists but is empty isn't an error.

    A render that legitimately produced zero entries is rare but
    legal; the user's mental model is "there are no rows to show",
    not "your dir is broken". Returning 0 lets ``audit`` compose
    cleanly with shell pipelines.
    """

    out = tmp_path / "render-out"
    out.mkdir()
    (out / AUDIT_FILENAME).write_text("", encoding="utf-8")

    rc = main(["audit", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "timestamp" in captured.out  # header still printed
    assert "(no audit entries)" in captured.out


def test_audit_empty_file_json_mode_emits_empty_array(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    out.mkdir()
    (out / AUDIT_FILENAME).write_text("", encoding="utf-8")

    rc = main(["audit", str(out), "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert json.loads(captured.out) == []


def test_audit_handles_null_recipe_sha(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In-memory recipes record ``recipe_sha=None`` — table must not crash.

    Pins the contract that the renderer (``Recipe(...)`` constructed
    in tests / future programmatic callers) lands as ``-`` in the
    table column rather than crashing on a slice of ``None``.
    """

    out = tmp_path / "render-out"
    _seed_audit(out, [_make_entry(recipe_sha=None)])

    rc = main(["audit", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    # The data row has ``-`` in the sha column.
    data_lines = [
        line
        for line in captured.out.splitlines()
        if line.strip() and not line.startswith("timestamp") and not line.startswith("-")
    ]
    assert len(data_lines) == 1
    assert " -        " in data_lines[0] or data_lines[0].split()[1] == "-"


def test_audit_truncates_long_recipe_names(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Recipe names longer than the column width get an ellipsis.

    Otherwise the row shifts and the table is unreadable. We pin the
    exact-fit / truncate behaviour so a rename to a longer name
    can't silently break the column layout.
    """

    out = tmp_path / "render-out"
    long_name = "this-is-a-very-long-recipe-name-that-overflows"
    _seed_audit(out, [_make_entry(recipe_name=long_name)])

    rc = main(["audit", str(out)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert long_name not in captured.out  # full name shouldn't fit
    assert "this-is-a-very-long-recipe" in captured.out or "…" in captured.out


# ---------------------------------------------------------------------------
# Filter flags: --since / --until / --recipe-sha (M10 stretch follow-up)
# ---------------------------------------------------------------------------
#
# These exercise the filtering layer that sits between the JSONL read and
# the ``--limit`` tail. The contract under test:
#
# - ``--since`` / ``--until`` are inclusive on both ends (second-precision
#   ISO-8601 timestamps make an exclusive bound user-hostile);
# - ``--recipe-sha`` is a case-insensitive prefix match and excludes rows
#   with a ``null`` SHA (an in-memory recipe shouldn't satisfy a "find
#   runs of recipe X" query);
# - filters apply *before* ``--limit`` so "last N matching" composes;
# - filter parse errors are usage errors (exit 2);
# - an empty filter result is still exit 0 (a successful query that
#   matched nothing isn't an error).


def _ts(hour: int) -> str:
    """ISO-8601 timestamp for ``2026-05-06T<hour>:00:00Z`` (test convenience)."""

    return f"2026-05-06T{hour:02d}:00:00Z"


def test_audit_since_filters_inclusive_lower_bound(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp=_ts(1), seed=1),
            _make_entry(timestamp=_ts(2), seed=2),
            _make_entry(timestamp=_ts(3), seed=3),
        ],
    )

    rc = main(["audit", str(out), "--json", "--since", _ts(2)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # ``--since`` is inclusive: the 02:00 row is kept.
    assert [entry["seed"] for entry in payload] == [2, 3]


def test_audit_until_filters_inclusive_upper_bound(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp=_ts(1), seed=1),
            _make_entry(timestamp=_ts(2), seed=2),
            _make_entry(timestamp=_ts(3), seed=3),
        ],
    )

    rc = main(["audit", str(out), "--json", "--until", _ts(2)])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # ``--until`` is inclusive: the 02:00 row is kept.
    assert [entry["seed"] for entry in payload] == [1, 2]


def test_audit_since_and_until_compose(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [_make_entry(timestamp=_ts(h), seed=h) for h in range(1, 6)],
    )

    rc = main(
        ["audit", str(out), "--json", "--since", _ts(2), "--until", _ts(4)],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # Closed range [02:00, 04:00].
    assert [entry["seed"] for entry in payload] == [2, 3, 4]


def test_audit_since_accepts_date_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--since 2026-05-06`` means ``2026-05-06T00:00:00Z`` (start-of-day).

    The convenience date-only form lets a user say "everything from
    today" without typing a time component.
    """

    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp="2026-05-05T23:59:59Z", seed=1),
            _make_entry(timestamp="2026-05-06T00:00:00Z", seed=2),
            _make_entry(timestamp="2026-05-06T12:00:00Z", seed=3),
        ],
    )

    rc = main(["audit", str(out), "--json", "--since", "2026-05-06"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # The midnight row is on the inclusive boundary; the prior-day row
    # is dropped.
    assert [entry["seed"] for entry in payload] == [2, 3]


def test_audit_since_invalid_format_is_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(out, [_make_entry()])

    rc = main(["audit", str(out), "--since", "yesterday"])
    captured = capsys.readouterr()
    assert rc == 2  # USAGE_EXIT
    assert "since" in captured.err.lower()


def test_audit_until_invalid_format_is_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(out, [_make_entry()])

    rc = main(["audit", str(out), "--until", "not-a-date"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "until" in captured.err.lower()


def test_audit_recipe_sha_prefix_filters(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp=_ts(1), seed=1, recipe_sha="aaaa" + "0" * 60),
            _make_entry(timestamp=_ts(2), seed=2, recipe_sha="bbbb" + "0" * 60),
            _make_entry(timestamp=_ts(3), seed=3, recipe_sha="aaaa" + "1" * 60),
        ],
    )

    rc = main(["audit", str(out), "--json", "--recipe-sha", "aaaa"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert [entry["seed"] for entry in payload] == [1, 3]


def test_audit_recipe_sha_is_case_insensitive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [_make_entry(recipe_sha="abcdef" + "0" * 58)],
    )

    rc = main(["audit", str(out), "--json", "--recipe-sha", "ABCDEF"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert len(json.loads(captured.out)) == 1


def test_audit_recipe_sha_excludes_null_sha_entries(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A SHA filter is read as "find runs of recipe X" — a null SHA
    means "no recipe identity recorded" and shouldn't satisfy that."""

    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp=_ts(1), seed=1, recipe_sha=None),
            _make_entry(timestamp=_ts(2), seed=2, recipe_sha="aa" + "0" * 62),
        ],
    )

    rc = main(["audit", str(out), "--json", "--recipe-sha", "aa"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert [entry["seed"] for entry in json.loads(captured.out)] == [2]


def test_audit_recipe_sha_empty_string_is_usage_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A flag value of ``""`` would otherwise match every entry — that
    silent surprise is more user-hostile than asking for a real prefix."""

    out = tmp_path / "render-out"
    _seed_audit(out, [_make_entry()])

    rc = main(["audit", str(out), "--recipe-sha", ""])
    captured = capsys.readouterr()
    assert rc == 2
    assert "recipe-sha" in captured.err.lower() or "recipe_sha" in captured.err.lower()


def test_audit_filter_then_limit_composes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--limit`` applies *after* the filters: "last 2 from today"."""

    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [_make_entry(timestamp=_ts(h), seed=h) for h in range(1, 6)],
    )

    rc = main(
        [
            "audit",
            str(out),
            "--json",
            "--since",
            _ts(2),
            "--until",
            _ts(4),
            "--limit",
            "2",
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # Filter window is [02:00, 04:00] → seeds [2, 3, 4]; tail-2 → [3, 4].
    assert [entry["seed"] for entry in payload] == [3, 4]


def test_audit_empty_filter_result_returns_zero_in_json_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A filter that matches nothing isn't an error — it's a successful
    query of an empty result set. JSON mode emits ``[]``."""

    out = tmp_path / "render-out"
    _seed_audit(out, [_make_entry(timestamp=_ts(1))])

    rc = main(
        ["audit", str(out), "--json", "--since", "2099-01-01"],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert json.loads(captured.out) == []


def test_audit_empty_filter_result_returns_zero_in_table_mode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Same contract in table mode: header + ``(no audit entries)``."""

    out = tmp_path / "render-out"
    _seed_audit(out, [_make_entry(timestamp=_ts(1))])

    rc = main(
        ["audit", str(out), "--recipe-sha", "deadbeef"],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "(no audit entries)" in captured.out


def test_audit_since_with_z_suffix_matches_stored_format(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The audit writer stores timestamps with a ``Z`` suffix; passing
    a ``+00:00`` form must still produce identical filter behaviour."""

    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp=_ts(1), seed=1),
            _make_entry(timestamp=_ts(3), seed=3),
        ],
    )

    rc = main(
        ["audit", str(out), "--json", "--since", "2026-05-06T02:00:00+00:00"],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    # Both forms (Z, +00:00) should normalize to the same comparison
    # string and keep only the 03:00 row.
    payload = json.loads(captured.out)
    assert [entry["seed"] for entry in payload] == [3]


def test_audit_since_with_positive_offset_normalizes_to_utc(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--since 2026-05-06T20:00:00+05:00`` is the same instant as
    ``2026-05-06T15:00:00Z`` and must filter against that — not against
    the lex-larger ``+05:00`` string. A stored row at 18:00Z is *after*
    that instant and must be kept; a stored row at 14:00Z is *before*
    and must be dropped.

    Regression for a latent bug: the old implementation called
    ``isoformat().replace('+00:00', 'Z')``, which left non-UTC offsets
    untouched. Lex comparison ``'2026-05-06T18:00:00Z' >=
    '2026-05-06T20:00:00+05:00'`` is false because ``T18`` < ``T20``,
    even though semantically 18:00Z is later than 15:00Z. Result: the
    18:00Z row was incorrectly excluded from a ``--since
    2026-05-06T20:00:00+05:00`` filter.
    """

    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp="2026-05-06T14:00:00Z", seed=1),  # before 15:00Z — drop
            _make_entry(timestamp="2026-05-06T18:00:00Z", seed=2),  # after 15:00Z  — keep
        ],
    )

    rc = main(
        ["audit", str(out), "--json", "--since", "2026-05-06T20:00:00+05:00"],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # Only the 18:00Z row survives the (UTC-equivalent 15:00Z) lower
    # bound.
    assert [entry["seed"] for entry in payload] == [2]


def test_audit_until_with_negative_offset_normalizes_to_utc(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Mirror of the +05:00 case for the upper bound + a negative
    offset. ``--until 2026-05-06T10:00:00-05:00`` is 15:00Z, so a row at
    14:00Z survives and a row at 16:00Z is dropped."""

    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp="2026-05-06T14:00:00Z", seed=1),  # before 15:00Z — keep
            _make_entry(timestamp="2026-05-06T16:00:00Z", seed=2),  # after 15:00Z  — drop
        ],
    )

    rc = main(
        ["audit", str(out), "--json", "--until", "2026-05-06T10:00:00-05:00"],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert [entry["seed"] for entry in payload] == [1]


# ---------------------------------------------------------------------------
# --summary aggregate mode (M10 stretch follow-up)
# ---------------------------------------------------------------------------
#
# Contract under test:
#
# - ``--summary`` returns a fixed-shape aggregate over the matched
#   window (post-filter, post-limit) — never the per-row table;
# - text mode prints a labelled block; JSON mode prints a single
#   object with every documented key present (zero / null / empty list
#   for the empty window so consumers can rely on the shape);
# - composes with ``--since``, ``--until``, ``--recipe-sha``, and
#   ``--limit`` because the summary fires after those filters in
#   ``_cmd_audit``;
# - top_recipe_shas reflects run frequency, with the SHA truncated for
#   readability and a deterministic tie-break.


def test_audit_summary_empty_window_returns_zero_stats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty audit file → all-zero / null summary in JSON mode."""

    out = tmp_path / "render-out"
    out.mkdir()
    (out / AUDIT_FILENAME).write_text("", encoding="utf-8")

    rc = main(["audit", str(out), "--summary", "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert payload == {
        "entry_count": 0,
        "total_count": 0,
        "total_rendered": 0,
        "total_skipped": 0,
        "total_runtime_seconds": 0.0,
        "distinct_recipe_names": 0,
        "distinct_recipe_shas": 0,
        "oldest_timestamp": None,
        "newest_timestamp": None,
        "top_recipe_shas": [],
    }


def test_audit_summary_sums_counts_and_runtime(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Aggregates over multiple entries are summed correctly."""

    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(
                timestamp=_ts(1),
                count=10,
                rendered=10,
                skipped=0,
                runtime_seconds=1.5,
            ),
            _make_entry(
                timestamp=_ts(2),
                count=20,
                rendered=18,
                skipped=2,
                runtime_seconds=2.25,
            ),
            _make_entry(
                timestamp=_ts(3),
                count=30,
                rendered=25,
                skipped=5,
                runtime_seconds=4.0,
            ),
        ],
    )

    rc = main(["audit", str(out), "--summary", "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert payload["entry_count"] == 3
    assert payload["total_count"] == 60
    assert payload["total_rendered"] == 53
    assert payload["total_skipped"] == 7
    assert payload["total_runtime_seconds"] == 7.75
    # All three entries share the same recipe_name / recipe_sha (the
    # _make_entry default), so the distinct counts collapse to 1.
    assert payload["distinct_recipe_names"] == 1
    assert payload["distinct_recipe_shas"] == 1
    assert payload["oldest_timestamp"] == _ts(1)
    assert payload["newest_timestamp"] == _ts(3)


def test_audit_summary_top_recipe_shas_ranks_by_run_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The top-3 SHAs list is sorted descending by run count.

    Builds five distinct SHAs with frequencies 5, 4, 3, 2, 1 so the
    ranking and the truncation are both observable.
    """

    out = tmp_path / "render-out"
    sha_a = "a" * 64
    sha_b = "b" * 64
    sha_c = "c" * 64
    sha_d = "d" * 64
    sha_e = "e" * 64

    entries: list[AuditEntry] = []
    hour = 1
    for sha, count in [(sha_a, 5), (sha_b, 4), (sha_c, 3), (sha_d, 2), (sha_e, 1)]:
        for _ in range(count):
            entries.append(
                _make_entry(
                    timestamp=_ts(hour % 24),
                    recipe_sha=sha,
                    recipe_name=f"r-{sha[0]}",
                )
            )
            hour += 1
    _seed_audit(out, entries)

    rc = main(["audit", str(out), "--summary", "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert payload["entry_count"] == 15
    assert payload["distinct_recipe_shas"] == 5
    assert payload["distinct_recipe_names"] == 5
    # Top-3 by run count, truncated to 12 hex chars.
    assert payload["top_recipe_shas"] == [
        {"recipe_sha": "a" * 12, "count": 5},
        {"recipe_sha": "b" * 12, "count": 4},
        {"recipe_sha": "c" * 12, "count": 3},
    ]


def test_audit_summary_respects_filters(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--summary`` aggregates over the *filtered* window, not the file.

    Seeds two SHAs and filters to one of them; the summary should only
    see that subset.
    """

    out = tmp_path / "render-out"
    sha_keep = "1" * 64
    sha_drop = "2" * 64
    _seed_audit(
        out,
        [
            _make_entry(timestamp=_ts(1), recipe_sha=sha_keep, count=10, rendered=10, skipped=0),
            _make_entry(timestamp=_ts(2), recipe_sha=sha_drop, count=99, rendered=99, skipped=0),
            _make_entry(timestamp=_ts(3), recipe_sha=sha_keep, count=20, rendered=18, skipped=2),
        ],
    )

    rc = main(
        [
            "audit",
            str(out),
            "--summary",
            "--json",
            "--recipe-sha",
            sha_keep[:8],
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # Only the two ``sha_keep`` entries should be in the window.
    assert payload["entry_count"] == 2
    assert payload["total_count"] == 30
    assert payload["total_rendered"] == 28
    assert payload["total_skipped"] == 2
    assert payload["distinct_recipe_shas"] == 1
    assert payload["top_recipe_shas"] == [{"recipe_sha": "1" * 12, "count": 2}]


def test_audit_summary_respects_limit_tail(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--summary --limit N`` summarizes only the most recent N entries.

    The summary fires after ``--limit`` clamps the window, so a user
    asking "what did the last 2 runs do?" gets a real answer.
    """

    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp=_ts(1), count=10, rendered=10, skipped=0),
            _make_entry(timestamp=_ts(2), count=20, rendered=20, skipped=0),
            _make_entry(timestamp=_ts(3), count=30, rendered=30, skipped=0),
        ],
    )

    rc = main(["audit", str(out), "--summary", "--json", "--limit", "2"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert payload["entry_count"] == 2
    # The last two rows have count 20 + 30 = 50, not the full 60.
    assert payload["total_count"] == 50
    assert payload["oldest_timestamp"] == _ts(2)
    assert payload["newest_timestamp"] == _ts(3)


def test_audit_summary_text_mode_renders_labelled_block(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default (no ``--json``) emits a human-readable block.

    Asserts each documented label appears on its own line and the
    top-recipe-shas section surfaces with the truncated SHA + run
    count.
    """

    out = tmp_path / "render-out"
    sha = "f" * 64
    _seed_audit(
        out,
        [
            _make_entry(
                timestamp=_ts(1),
                recipe_sha=sha,
                count=5,
                rendered=4,
                skipped=1,
                runtime_seconds=2.5,
            ),
            _make_entry(
                timestamp=_ts(2),
                recipe_sha=sha,
                count=7,
                rendered=7,
                skipped=0,
                runtime_seconds=3.5,
            ),
        ],
    )

    rc = main(["audit", str(out), "--summary"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    stdout = captured.out
    assert "entry_count:" in stdout
    assert "total_count:" in stdout
    assert "total_rendered:" in stdout
    assert "total_skipped:" in stdout
    assert "total_runtime_seconds:" in stdout
    assert "distinct_recipe_names:" in stdout
    assert "distinct_recipe_shas:" in stdout
    assert "oldest_timestamp:" in stdout
    assert "newest_timestamp:" in stdout
    assert "top_recipe_shas:" in stdout
    # Truncated SHA (12 chars) and the run count appear in the block.
    assert ("f" * 12) in stdout
    # Runtime is the summed 2.5 + 3.5 = 6.00, formatted with 2 decimals.
    assert "6.00" in stdout


def test_audit_summary_text_mode_omits_top_section_when_empty(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty window has no SHAs to rank, so the top block is suppressed."""

    out = tmp_path / "render-out"
    out.mkdir()
    (out / AUDIT_FILENAME).write_text("", encoding="utf-8")

    rc = main(["audit", str(out), "--summary"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    stdout = captured.out
    assert "entry_count:           0" in stdout
    # No "top_recipe_shas:" header when the window has no SHAs.
    assert "top_recipe_shas:" not in stdout


def test_audit_summary_skips_null_recipe_sha_for_distinct_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """In-memory recipes (null SHA) don't count toward distinct-SHA tally.

    They still count toward ``entry_count`` and the summed metrics —
    we just can't fingerprint them. This matches the audit reader
    docstring: a null SHA means "no stable identifier".
    """

    out = tmp_path / "render-out"
    _seed_audit(
        out,
        [
            _make_entry(timestamp=_ts(1), recipe_sha=None, count=10, rendered=10, skipped=0),
            _make_entry(timestamp=_ts(2), recipe_sha="a" * 64, count=20, rendered=20, skipped=0),
        ],
    )

    rc = main(["audit", str(out), "--summary", "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert payload["entry_count"] == 2
    assert payload["total_count"] == 30
    # Only the SHA-bearing entry contributes to the distinct count.
    assert payload["distinct_recipe_shas"] == 1
    # And only that entry shows up in top_recipe_shas.
    assert payload["top_recipe_shas"] == [{"recipe_sha": "a" * 12, "count": 1}]


# ---------------------------------------------------------------------------
# --audit-file override (M10 stretch QoL): read from a non-default path
# ---------------------------------------------------------------------------
#
# The default lookup is ``<output_dir>/_audit.jsonl``. ``--audit-file PATH``
# overrides that so users can replay archived staging-dir logs or aggregate
# files without copying them into a render dir. Contract under test:
#
# - The flag-pointed file is read; the canonical file under output_dir is
#   not opened (and need not exist).
# - The output_dir positional is still required; argparse rejects a missing
#   one as a usage error — covered implicitly by other audit tests.
# - Missing flag-pointed file maps to exit 6 (destination-family) with a
#   clear "--audit-file does not exist" error.
# - The flag composes with --json, --since, --recipe-sha, --limit, --summary.


def test_audit_audit_file_reads_from_override_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--audit-file PATH`` reads from PATH instead of <output_dir>/_audit.jsonl.

    Pins the override resolution: when the flag is set, the canonical
    location under ``output_dir`` is not touched. We seed the override
    path with one set of entries and the default location with a
    different set; the resulting JSON must be the override's, not the
    default's.
    """

    # Default location (would be read without the flag).
    out = tmp_path / "render-out"
    _seed_audit(out, [_make_entry(seed=999, recipe_name="ignored-default")])

    # Override location (this is what the flag points at).
    archive = tmp_path / "archived" / "audit.jsonl"
    archive.parent.mkdir(parents=True)
    append_audit_entry(archive, _make_entry(seed=42, recipe_name="archived-recipe"))
    append_audit_entry(archive, _make_entry(seed=43, recipe_name="archived-recipe"))

    rc = main(["audit", str(out), "--audit-file", str(archive), "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # Two entries from the override path; the default-location row
    # (``seed=999``) must not appear.
    assert [entry["seed"] for entry in payload] == [42, 43]
    assert all(entry["recipe_name"] == "archived-recipe" for entry in payload)


def test_audit_audit_file_does_not_require_default_file(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When ``--audit-file`` is set, the default file is not required.

    The output_dir positional is still passed (argparse demands it) and
    must exist (it's a real dir on disk in this test), but the canonical
    ``_audit.jsonl`` need not be present. Pins that the override truly
    overrides — not augments.
    """

    out = tmp_path / "render-out"
    out.mkdir()
    # Note: NO ``_audit.jsonl`` here. The default-only case (no flag)
    # would exit 6 with "no audit file at ..." — see
    # ``test_audit_missing_file_is_destination_error``.

    archive = tmp_path / "elsewhere.jsonl"
    append_audit_entry(archive, _make_entry(seed=7))

    rc = main(["audit", str(out), "--audit-file", str(archive), "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert [entry["seed"] for entry in payload] == [7]


def test_audit_audit_file_missing_path_is_destination_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A non-existent ``--audit-file`` path → exit 6 with a clear message.

    Same family as a missing default file: the consumer pointed us at
    something that isn't there. The error message names the flag so
    the user knows which input was wrong.
    """

    out = tmp_path / "render-out"
    out.mkdir()
    bogus = tmp_path / "nope.jsonl"
    assert not bogus.exists()

    rc = main(["audit", str(out), "--audit-file", str(bogus)])
    captured = capsys.readouterr()
    assert rc == 6
    assert "--audit-file does not exist" in captured.err
    assert str(bogus) in captured.err


def test_audit_audit_file_composes_with_filters(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--audit-file`` composes with ``--since`` / ``--recipe-sha`` / ``--limit``.

    A single end-to-end smoke test of filter composition is enough —
    the per-filter behaviour is already covered by the default-path
    tests above, and the override path uses the same filter pipeline.
    """

    out = tmp_path / "render-out"
    out.mkdir()
    archive = tmp_path / "archive.jsonl"
    sha_a = "a" * 64
    sha_b = "b" * 64
    for entry in [
        _make_entry(timestamp=_ts(1), seed=1, recipe_sha=sha_a),
        _make_entry(timestamp=_ts(2), seed=2, recipe_sha=sha_b),
        _make_entry(timestamp=_ts(3), seed=3, recipe_sha=sha_a),
        _make_entry(timestamp=_ts(4), seed=4, recipe_sha=sha_a),
    ]:
        append_audit_entry(archive, entry)

    rc = main(
        [
            "audit",
            str(out),
            "--audit-file",
            str(archive),
            "--json",
            "--since",
            _ts(2),
            "--recipe-sha",
            "a" * 8,
            "--limit",
            "1",
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # ``--since 2`` keeps rows 2, 3, 4. ``--recipe-sha aaaa...`` drops row 2
    # (which is sha_b). ``--limit 1`` tails to the most recent (seed 4).
    assert [entry["seed"] for entry in payload] == [4]


def test_audit_audit_file_with_summary_aggregates_override_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--audit-file`` + ``--summary`` aggregates the override file's rows.

    Locks that the summary path uses the same resolution: rows from
    the default location are not silently mixed in with the override.
    """

    out = tmp_path / "render-out"
    # Seed the default location with rows that *would* skew the summary
    # if the override resolution were broken.
    _seed_audit(
        out,
        [
            _make_entry(count=999, rendered=999, skipped=0),
            _make_entry(count=999, rendered=999, skipped=0),
            _make_entry(count=999, rendered=999, skipped=0),
        ],
    )

    archive = tmp_path / "small.jsonl"
    append_audit_entry(archive, _make_entry(count=10, rendered=10, skipped=0))
    append_audit_entry(archive, _make_entry(count=20, rendered=18, skipped=2))

    rc = main(
        ["audit", str(out), "--audit-file", str(archive), "--summary", "--json"],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # If the default file were leaking in, entry_count would be 5 and
    # total_count would dwarf 30.
    assert payload["entry_count"] == 2
    assert payload["total_count"] == 30
    assert payload["total_rendered"] == 28
    assert payload["total_skipped"] == 2


# ---------------------------------------------------------------------------
# --global flag (M10 stretch QoL): read from the cross-recipe aggregate
# ---------------------------------------------------------------------------
#
# The render pipeline mirrors every audit row to a global aggregate at
# ``<cache_root>/audit.jsonl`` so cross-recipe forensics are a one-liner
# rather than "point the audit subcommand at each output dir separately".
# ``--global`` is the read-side companion: it short-circuits the
# positional ``output_dir`` (no per-output-dir context needed) and reads
# from the aggregate path.
#
# Contract under test:
#
# - ``--global`` reads from ``$PD_OCR_SYNTH_CACHE/audit.jsonl``.
# - ``output_dir`` is optional in ``--global`` mode.
# - ``--global`` and ``--audit-file`` are mutually exclusive (exit 2).
# - A missing global path is exit 0 with empty result (the canonical
#   path is allowed to be absent on a fresh machine — that's a valid
#   "I have run zero renders" answer).
# - ``--global`` composes with --json, --since, --recipe-sha, --limit,
#   --summary.


def test_audit_global_flag_reads_aggregate_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--global`` reads from ``<cache_root>/audit.jsonl``.

    Seeds the global aggregate with a couple of rows under an
    isolated cache root, then asserts ``audit --global --json``
    surfaces them. ``output_dir`` is intentionally omitted to lock
    that the flag short-circuits the positional.
    """

    from pd_ocr_synth.audit import GLOBAL_AUDIT_FILENAME

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))

    aggregate = cache_root / GLOBAL_AUDIT_FILENAME
    append_audit_entry(aggregate, _make_entry(seed=1, recipe_name="recipe-a"))
    append_audit_entry(aggregate, _make_entry(seed=2, recipe_name="recipe-b"))

    rc = main(["audit", "--global", "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert [r["seed"] for r in payload] == [1, 2]
    assert {r["recipe_name"] for r in payload} == {"recipe-a", "recipe-b"}


def test_audit_global_flag_does_not_require_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Positional ``output_dir`` is optional under ``--global``.

    The aggregate has its own canonical path; per-output-dir context
    is meaningless. ``argparse``'s ``nargs="?"`` makes the positional
    optional at the parse layer; this test asserts the runtime
    handler accepts the missing positional in ``--global`` mode.
    """

    from pd_ocr_synth.audit import GLOBAL_AUDIT_FILENAME

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))

    aggregate = cache_root / GLOBAL_AUDIT_FILENAME
    append_audit_entry(aggregate, _make_entry(seed=99))

    rc = main(["audit", "--global"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    # Table mode header + one data row.
    assert "timestamp" in captured.out  # header
    # No "no audit entries" line when there is one row.
    assert "(no audit entries)" not in captured.out


def test_audit_global_with_missing_aggregate_returns_empty_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No aggregate file yet → exit 0 with an empty result set.

    Distinct from ``--audit-file <missing>`` (which is exit 6: the
    consumer asserted the file exists). The global path is canonical
    and may legitimately be absent on a fresh machine — "no runs
    yet" is a valid answer to "show me my audit history".
    """

    cache_root = tmp_path / "cache-empty"
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))
    # Note: cache_root not created; the aggregate is absent.

    # JSON mode for an unambiguous empty signal.
    rc = main(["audit", "--global", "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert json.loads(captured.out) == []


def test_audit_global_with_missing_aggregate_table_mode_renders_empty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Table mode on a missing aggregate prints header + the empty marker.

    Same shape as the empty-default-file case (exit 0 with header +
    ``(no audit entries)``); locks that ``--global`` reuses the
    same renderer and doesn't introduce a different empty-set
    output path.
    """

    cache_root = tmp_path / "cache-empty"
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))

    rc = main(["audit", "--global"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "timestamp" in captured.out  # header
    assert "(no audit entries)" in captured.out


def test_audit_global_and_audit_file_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Passing both ``--global`` and ``--audit-file`` is a usage error.

    The two flags select competing input sources; rejecting the
    combination at exit 2 prevents the implementation from having to
    pick a precedence rule (and prevents the user from being
    confused about which file got read).
    """

    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(tmp_path / "cache"))
    archive = tmp_path / "elsewhere.jsonl"
    append_audit_entry(archive, _make_entry(seed=1))

    rc = main(["audit", "--global", "--audit-file", str(archive)])
    captured = capsys.readouterr()
    assert rc == 2
    assert "mutually exclusive" in captured.err


def test_audit_global_composes_with_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--global`` + ``--since`` + ``--recipe-sha`` + ``--limit`` compose.

    Same composition contract as ``--audit-file``: one smoke test
    end-to-end is enough since the filter pipeline is shared.
    """

    from pd_ocr_synth.audit import GLOBAL_AUDIT_FILENAME

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))
    aggregate = cache_root / GLOBAL_AUDIT_FILENAME
    sha_a = "a" * 64
    sha_b = "b" * 64
    for entry in [
        _make_entry(timestamp=_ts(1), seed=1, recipe_sha=sha_a),
        _make_entry(timestamp=_ts(2), seed=2, recipe_sha=sha_b),
        _make_entry(timestamp=_ts(3), seed=3, recipe_sha=sha_a),
        _make_entry(timestamp=_ts(4), seed=4, recipe_sha=sha_a),
    ]:
        append_audit_entry(aggregate, entry)

    rc = main(
        [
            "audit",
            "--global",
            "--json",
            "--since",
            _ts(2),
            "--recipe-sha",
            "a" * 8,
            "--limit",
            "1",
        ],
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    # ``--since 2`` keeps rows 2, 3, 4. ``--recipe-sha aaaa...`` drops row 2.
    # ``--limit 1`` tails to seed=4.
    assert [entry["seed"] for entry in payload] == [4]


def test_audit_global_with_summary_aggregates_aggregate_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--global`` + ``--summary`` aggregates the aggregate file's rows.

    Locks that the global path doesn't accidentally pull in any
    per-output-dir audit files — the aggregate stands alone.
    """

    from pd_ocr_synth.audit import GLOBAL_AUDIT_FILENAME

    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))
    aggregate = cache_root / GLOBAL_AUDIT_FILENAME
    append_audit_entry(aggregate, _make_entry(count=10, rendered=10, skipped=0))
    append_audit_entry(aggregate, _make_entry(count=20, rendered=18, skipped=2))

    rc = main(["audit", "--global", "--summary", "--json"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    payload = json.loads(captured.out)
    assert payload["entry_count"] == 2
    assert payload["total_count"] == 30
    assert payload["total_rendered"] == 28
    assert payload["total_skipped"] == 2


def test_audit_without_args_is_usage_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No positional + no ``--global`` + no ``--audit-file`` → exit 2.

    Pins the help-text message: the user gets a single line naming
    the alternatives, not a stack trace.
    """

    rc = main(["audit"])
    captured = capsys.readouterr()
    assert rc == 2
    assert "output_dir is required" in captured.err
    assert "--global" in captured.err
    assert "--audit-file" in captured.err
