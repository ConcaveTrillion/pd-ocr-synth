"""Unit tests for ``pd_ocr_synth.audit`` (M10 stretch).

Per-render audit JSONL log: one line per ``run_recipe`` invocation,
recording timestamp, recipe identity (name + source SHA), seed, and
outcome counts. Tests here cover the small surface of the audit
module in isolation; the run-through-CLI integration test lives in
``test_cli_render_audit.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from pd_ocr_synth.audit import (
    AUDIT_DISABLE_ENV,
    AUDIT_FILENAME,
    AUDIT_SCHEMA_VERSION,
    AuditEntry,
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
