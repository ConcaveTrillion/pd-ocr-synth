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
