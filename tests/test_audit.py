"""Unit tests for ``pd_ocr_synth.audit`` (M10 stretch).

Per-render audit JSONL log: one line per ``run_recipe`` invocation,
recording timestamp, recipe identity (name + source SHA), seed, and
outcome counts. Tests here cover the small surface of the audit
module in isolation; the run-through-CLI integration test lives in
``test_cli_render_audit.py``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from pd_ocr_synth.audit import (
    AUDIT_DISABLE_ENV,
    AUDIT_FILENAME,
    AUDIT_SCHEMA_VERSION,
    AuditEntry,
    AuditSchemaVersionWarning,
    append_audit_entry,
    compute_recipe_sha,
    now_timestamp,
    read_audit_entries,
    should_emit_audit,
)

# ---------------------------------------------------------------------------
# AuditEntry shape
# ---------------------------------------------------------------------------


def _make_entry(**overrides) -> AuditEntry:
    """Helper: build an AuditEntry with sensible defaults."""

    defaults: dict = {
        "timestamp": "2026-05-06T00:00:00Z",
        "recipe_name": "render-smoke",
        "recipe_sha": "0" * 64,
        "output_dir": "/tmp/out",
        "count": 8,
        "seed": 21,
        "workers": 1,
        "rendered": 8,
        "skipped": 0,
        "runtime_seconds": 1.5,
    }
    defaults.update(overrides)
    return AuditEntry(**defaults)


def test_audit_entry_serializes_to_jsonl_with_all_fields() -> None:
    entry = _make_entry()
    payload = json.loads(entry.to_jsonl())

    # Every documented field present, no extras (forward-compat
    # readers must round-trip cleanly).
    assert set(payload.keys()) == {
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
    }
    assert payload["schema_version"] == AUDIT_SCHEMA_VERSION
    assert payload["recipe_name"] == "render-smoke"


def test_audit_entry_jsonl_is_single_line_no_trailing_newline() -> None:
    line = _make_entry().to_jsonl()
    assert "\n" not in line  # writer adds the newline
    assert line.startswith("{")
    assert line.endswith("}")


def test_audit_entry_recipe_sha_can_be_none() -> None:
    """In-memory recipes have no source_path → SHA is ``None``."""

    entry = _make_entry(recipe_sha=None)
    payload = json.loads(entry.to_jsonl())
    assert payload["recipe_sha"] is None


# ---------------------------------------------------------------------------
# should_emit_audit
# ---------------------------------------------------------------------------


def test_should_emit_audit_returns_true_when_flag_set_and_env_clean() -> None:
    assert should_emit_audit(audit=True, env={}) is True


def test_should_emit_audit_returns_false_when_flag_clear() -> None:
    assert should_emit_audit(audit=False, env={}) is False


def test_should_emit_audit_env_var_overrides_flag() -> None:
    """``PD_OCR_SYNTH_NO_AUDIT=1`` suppresses even when audit=True."""

    for truthy in ("1", "true", "TRUE", "yes", "YES", "on"):
        assert should_emit_audit(audit=True, env={AUDIT_DISABLE_ENV: truthy}) is False


def test_should_emit_audit_env_falsy_does_not_suppress() -> None:
    """Empty / "0" / "false" env values must NOT suppress audit."""

    for falsy in ("", "0", "false", "no", "off"):
        assert should_emit_audit(audit=True, env={AUDIT_DISABLE_ENV: falsy}) is True


# ---------------------------------------------------------------------------
# compute_recipe_sha
# ---------------------------------------------------------------------------


def test_compute_recipe_sha_is_none_when_path_is_none() -> None:
    assert compute_recipe_sha(None) is None


def test_compute_recipe_sha_is_none_when_file_missing(tmp_path: Path) -> None:
    assert compute_recipe_sha(tmp_path / "does-not-exist.yaml") is None


def test_compute_recipe_sha_hashes_file_bytes(tmp_path: Path) -> None:
    p = tmp_path / "r.yaml"
    p.write_text("name: foo\n", encoding="utf-8")
    sha = compute_recipe_sha(p)
    assert isinstance(sha, str)
    assert len(sha) == 64
    # Expected SHA-256 of the literal bytes; ground truth in the test
    # so a regression in the hash function fails loudly.
    import hashlib

    assert sha == hashlib.sha256(b"name: foo\n").hexdigest()


def test_compute_recipe_sha_changes_on_content_edit(tmp_path: Path) -> None:
    p = tmp_path / "r.yaml"
    p.write_text("name: a\n", encoding="utf-8")
    sha_a = compute_recipe_sha(p)
    p.write_text("name: b\n", encoding="utf-8")
    sha_b = compute_recipe_sha(p)
    assert sha_a != sha_b


# ---------------------------------------------------------------------------
# now_timestamp
# ---------------------------------------------------------------------------


def test_now_timestamp_returns_iso8601_z_suffix() -> None:
    ts = now_timestamp()
    # YYYY-MM-DDTHH:MM:SSZ — 20 chars
    assert len(ts) == 20
    assert ts.endswith("Z")
    assert ts[4] == "-" and ts[7] == "-" and ts[10] == "T"
    assert ts[13] == ":" and ts[16] == ":"


# ---------------------------------------------------------------------------
# append_audit_entry / read_audit_entries
# ---------------------------------------------------------------------------


def test_append_audit_entry_creates_file_and_writes_one_line(tmp_path: Path) -> None:
    audit_path = tmp_path / "out" / AUDIT_FILENAME
    append_audit_entry(audit_path, _make_entry())
    assert audit_path.is_file()
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["recipe_name"] == "render-smoke"


def test_append_audit_entry_appends_without_clobbering(tmp_path: Path) -> None:
    """Two appends → two distinct lines, in order."""

    audit_path = tmp_path / AUDIT_FILENAME
    append_audit_entry(audit_path, _make_entry(seed=1))
    append_audit_entry(audit_path, _make_entry(seed=2))
    entries = read_audit_entries(audit_path)
    assert len(entries) == 2
    assert entries[0]["seed"] == 1
    assert entries[1]["seed"] == 2


def test_read_audit_entries_handles_missing_file(tmp_path: Path) -> None:
    assert read_audit_entries(tmp_path / "nope.jsonl") == []


def test_read_audit_entries_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / AUDIT_FILENAME
    entry = _make_entry()
    p.write_text(
        "\n" + entry.to_jsonl() + "\n\n" + entry.to_jsonl() + "\n   \n",
        encoding="utf-8",
    )
    assert len(read_audit_entries(p)) == 2


def test_append_audit_entry_creates_nested_parent_dirs(tmp_path: Path) -> None:
    """Audit file's parent dir is created on demand (the writer also
    creates ``output_dir`` so this is normally a no-op, but the
    helper must not assume that)."""

    audit_path = tmp_path / "deeply" / "nested" / "out" / AUDIT_FILENAME
    append_audit_entry(audit_path, _make_entry())
    assert audit_path.is_file()


def test_append_audit_entry_closes_file_when_serialization_raises(
    tmp_path: Path,
) -> None:
    """Regression guard: a mid-write failure must not leak the FD.

    ``append_audit_entry`` opens the audit file in append mode and
    writes one JSONL line. If serialization (``entry.to_jsonl()``)
    raises *inside* the open block, the implementation must still
    close the underlying file handle — i.e. it must use a context
    manager (``with audit_path.open(...) as fh``) rather than a
    bare ``open()`` followed by ``write()``.

    The check: trigger a write-time failure by making
    ``to_jsonl`` raise, capture warnings as errors so an unclosed
    file at GC time would surface as a ``ResourceWarning``, force
    a GC pass, and assert no ``ResourceWarning`` fires. This is
    Python's standard mechanism for catching FD leaks and matches
    how ``pytest -W error::ResourceWarning`` would catch them in a
    stricter CI mode.

    Why this matters: the M10 audit emit-failure path in
    ``render/run.py`` catches ``OSError`` and downgrades it to a
    stderr warning. If the underlying open file isn't closed, the
    "render succeeded but audit failed" path silently leaks a FD
    per render attempt — invisible under happy-path testing,
    discoverable under fault-injection.
    """

    import gc
    import warnings as _warnings

    audit_path = tmp_path / AUDIT_FILENAME

    class _Boom(Exception):  # noqa: N818
        pass

    class _BadEntry:
        """AuditEntry-shaped stand-in whose serializer raises."""

        def to_jsonl(self) -> str:
            raise _Boom("serialization failure")

    with _warnings.catch_warnings():
        _warnings.simplefilter("error", ResourceWarning)
        with pytest.raises(_Boom):
            append_audit_entry(audit_path, _BadEntry())  # type: ignore[arg-type]
        # Force the GC: an unclosed file would trip ResourceWarning
        # only on finalization, which CPython runs eagerly for
        # refcount-zero objects but not always immediately under
        # PyPy / debug builds. ``gc.collect()`` is the portable
        # belt-and-braces.
        gc.collect()

    # Belt-and-braces: the file may or may not exist (depending on
    # whether OS-level append created the inode before write
    # raised), but if it does exist it must be closeable from a
    # fresh open — i.e. nothing else is holding a write lock to it.
    if audit_path.exists():
        with audit_path.open("a", encoding="utf-8") as fh:
            fh.write("")  # smoke: append-mode reopen succeeds


# ---------------------------------------------------------------------------
# Forward-compat: schema_version skip-with-warning policy
# ---------------------------------------------------------------------------


def _v2_line(**overrides) -> str:
    """Build a synthetic ``schema_version=2`` audit line.

    Future-version rows must be tolerated by today's reader (skipped,
    not crashed-on). The shape here mirrors v1 minus a hypothetical
    field rename so the test exercises the version branch even if
    field-set parity happens to hold.
    """

    payload = {
        "timestamp": "2099-01-01T00:00:00Z",
        "recipe_name": "future-recipe",
        "recipe_sha": "f" * 64,
        "output_dir": "/tmp/v2",
        "planned_count": 16,  # hypothetical v2 rename of ``count``
        "seed": 99,
        "workers": 2,
        "rendered": 16,
        "skipped": 0,
        "runtime_milliseconds": 2500,  # hypothetical v2 unit change
        "schema_version": 2,
    }
    payload.update(overrides)
    return json.dumps(payload)


def test_read_audit_entries_skips_future_schema_version_with_warning(
    tmp_path: Path,
) -> None:
    """v2 row + v1 row → only v1 returned, one aggregated warning."""

    p = tmp_path / AUDIT_FILENAME
    v1_line = _make_entry(seed=1).to_jsonl()
    p.write_text(_v2_line() + "\n" + v1_line + "\n", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AuditSchemaVersionWarning)
        entries = read_audit_entries(p)

    assert len(entries) == 1
    assert entries[0]["seed"] == 1  # the v1 row
    # Exactly one warning even though we could have fired per-row;
    # the aggregated form keeps long mixed-version files quiet.
    schema_warnings = [w for w in caught if issubclass(w.category, AuditSchemaVersionWarning)]
    assert len(schema_warnings) == 1
    msg = str(schema_warnings[0].message)
    assert "schema_version" in msg
    assert "2" in msg  # the unknown version mentioned


def test_read_audit_entries_no_warning_when_all_rows_current_version(
    tmp_path: Path,
) -> None:
    """All-v1 file → no schema-version warning emitted."""

    p = tmp_path / AUDIT_FILENAME
    append_audit_entry(p, _make_entry(seed=1))
    append_audit_entry(p, _make_entry(seed=2))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AuditSchemaVersionWarning)
        entries = read_audit_entries(p)

    assert len(entries) == 2
    schema_warnings = [w for w in caught if issubclass(w.category, AuditSchemaVersionWarning)]
    assert schema_warnings == []


def test_read_audit_entries_aggregates_multiple_unknown_versions(
    tmp_path: Path,
) -> None:
    """Multiple distinct unknown versions → still one warning, naming each."""

    p = tmp_path / AUDIT_FILENAME
    p.write_text(
        _v2_line()
        + "\n"
        + _v2_line()
        + "\n"
        + _v2_line(schema_version=3)
        + "\n"
        + _make_entry().to_jsonl()
        + "\n",
        encoding="utf-8",
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AuditSchemaVersionWarning)
        entries = read_audit_entries(p)

    assert len(entries) == 1
    schema_warnings = [w for w in caught if issubclass(w.category, AuditSchemaVersionWarning)]
    assert len(schema_warnings) == 1
    msg = str(schema_warnings[0].message)
    # Both encountered versions surfaced with their counts so an
    # operator skimming the warning sees the full picture.
    assert "2" in msg and "3" in msg
    assert "2x" in msg  # two v2 rows


def test_read_audit_entries_treats_missing_schema_version_as_v1(
    tmp_path: Path,
) -> None:
    """Hand-edited / legacy lines without ``schema_version`` are kept.

    Forward-compat applies to *unknown* versions; absence of the field
    is interpreted as legacy v1 to keep minimal fixtures and pre-v1
    hand-rolled audit files readable.
    """

    p = tmp_path / AUDIT_FILENAME
    legacy = json.dumps({"recipe_name": "legacy", "seed": 7})  # no schema_version
    p.write_text(legacy + "\n", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AuditSchemaVersionWarning)
        entries = read_audit_entries(p)

    assert len(entries) == 1
    assert entries[0]["recipe_name"] == "legacy"
    schema_warnings = [w for w in caught if issubclass(w.category, AuditSchemaVersionWarning)]
    assert schema_warnings == []


def test_read_audit_entries_treats_non_int_schema_version_as_unknown(
    tmp_path: Path,
) -> None:
    """``"1"`` (string) and ``1.0`` (float) are NOT v1 — type-check is strict.

    Prevents silent acceptance of malformed-but-truthy version values
    that could mask a real schema migration bug. ``True`` is also
    excluded (Python ``True == 1`` is True at the value level but the
    intent in an audit row is clearly bogus).
    """

    p = tmp_path / AUDIT_FILENAME
    bogus = json.dumps({"recipe_name": "bogus", "schema_version": "1"})
    p.write_text(bogus + "\n", encoding="utf-8")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", AuditSchemaVersionWarning)
        entries = read_audit_entries(p)

    assert entries == []
    schema_warnings = [w for w in caught if issubclass(w.category, AuditSchemaVersionWarning)]
    assert len(schema_warnings) == 1


def test_read_audit_entries_warning_can_be_promoted_to_error(
    tmp_path: Path,
) -> None:
    """Strict CI tooling can treat unknown versions as a hard error."""

    p = tmp_path / AUDIT_FILENAME
    p.write_text(_v2_line() + "\n", encoding="utf-8")

    with warnings.catch_warnings():
        warnings.simplefilter("error", AuditSchemaVersionWarning)
        with pytest.raises(AuditSchemaVersionWarning):
            read_audit_entries(p)
