"""Unit tests for the global aggregate audit log (M10 stretch QoL).

The render pipeline writes a per-output-dir audit row to
``<output_dir>/_audit.jsonl`` (covered in ``test_audit.py`` /
``test_cli_render_audit.py``). This file covers the *global*
aggregate mirror at ``<cache_root>/audit.jsonl`` introduced as a
QoL chunk on top of M10 to make cross-recipe forensics a one-liner
rather than "point the audit subcommand at each output dir
separately".

Coverage matrix here:

- :func:`default_global_audit_path` resolution
  (``$PD_OCR_SYNTH_CACHE`` / default home).
- :func:`should_emit_global_audit` matrix
  (``audit=False`` short-circuits, base disable env, dedicated
  ``PD_OCR_SYNTH_NO_GLOBAL_AUDIT``).
- End-to-end: ``run_recipe`` mirrors the row to the global path
  alongside the per-output-dir log, and the rows match.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pd_ocr_synth.audit import (
    AUDIT_DISABLE_ENV,
    AUDIT_FILENAME,
    AUDIT_SCHEMA_VERSION,
    GLOBAL_AUDIT_DISABLE_ENV,
    GLOBAL_AUDIT_FILENAME,
    AuditEntry,
    append_audit_entry,
    default_global_audit_path,
    read_audit_entries,
    should_emit_audit,
    should_emit_global_audit,
)

# ---------------------------------------------------------------------------
# default_global_audit_path
# ---------------------------------------------------------------------------


def test_default_global_audit_path_uses_cache_env_var(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``$PD_OCR_SYNTH_CACHE`` (the corpus cache var) anchors the path."""

    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(tmp_path))
    resolved = default_global_audit_path()
    assert resolved == tmp_path / GLOBAL_AUDIT_FILENAME


def test_default_global_audit_path_defaults_to_user_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the env var the path falls under ``~/.cache/pd-ocr-synth``."""

    monkeypatch.delenv("PD_OCR_SYNTH_CACHE", raising=False)
    resolved = default_global_audit_path()
    assert resolved == Path.home() / ".cache" / "pd-ocr-synth" / GLOBAL_AUDIT_FILENAME


def test_default_global_audit_path_does_not_create_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Resolution is pure: no side-effects on the filesystem."""

    cache_root = tmp_path / "fresh-root"
    assert not cache_root.exists()
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))
    resolved = default_global_audit_path()
    assert resolved == cache_root / GLOBAL_AUDIT_FILENAME
    # Calling again does not materialize the directory.
    _ = default_global_audit_path()
    assert not cache_root.exists()


def test_default_global_audit_path_accepts_env_dict() -> None:
    """``env`` injection makes the helper testable without monkeypatch."""

    fake_env = {"PD_OCR_SYNTH_CACHE": "/var/tmp/fixed"}
    resolved = default_global_audit_path(env=fake_env)
    assert resolved == Path("/var/tmp/fixed") / GLOBAL_AUDIT_FILENAME


# ---------------------------------------------------------------------------
# should_emit_global_audit
# ---------------------------------------------------------------------------


def test_should_emit_global_audit_short_circuits_when_audit_off() -> None:
    """``audit=False`` disables the global mirror unconditionally.

    The per-render audit gate is the parent switch — turning the audit
    off entirely (CLI ``--no-audit``) must turn off both files.
    """

    assert should_emit_global_audit(audit=False, env={}) is False


def test_should_emit_global_audit_respects_base_disable_env() -> None:
    """``PD_OCR_SYNTH_NO_AUDIT=1`` disables the global mirror too."""

    assert (
        should_emit_global_audit(
            audit=True,
            env={AUDIT_DISABLE_ENV: "1"},
        )
        is False
    )


def test_should_emit_global_audit_respects_dedicated_disable_env() -> None:
    """The dedicated env var disables *only* the global mirror.

    The per-output-dir audit (gated by ``should_emit_audit``) still
    fires; this fixture asserts the per-render audit stays on while
    the global mirror is off — the documented use-case.
    """

    env = {GLOBAL_AUDIT_DISABLE_ENV: "1"}
    assert should_emit_audit(audit=True, env=env) is True
    assert should_emit_global_audit(audit=True, env=env) is False


@pytest.mark.parametrize("truthy", ["1", "true", "yes", "on", "TRUE", "Yes"])
def test_should_emit_global_audit_accepts_common_truthy_values(truthy: str) -> None:
    """Same truthy-set as ``should_emit_audit`` (case-insensitive)."""

    env = {GLOBAL_AUDIT_DISABLE_ENV: truthy}
    assert should_emit_global_audit(audit=True, env=env) is False


@pytest.mark.parametrize("falsy", ["", "0", "false", "no", "off"])
def test_should_emit_global_audit_falsy_keeps_mirror_on(falsy: str) -> None:
    """Falsy / empty values do not disable; matches ``should_emit_audit``."""

    env = {GLOBAL_AUDIT_DISABLE_ENV: falsy}
    assert should_emit_global_audit(audit=True, env=env) is True


def test_should_emit_global_audit_default_env_is_on() -> None:
    """Empty env → both gates open."""

    assert should_emit_global_audit(audit=True, env={}) is True


# ---------------------------------------------------------------------------
# End-to-end: ``run_recipe`` mirrors the audit row into the aggregate.
# ---------------------------------------------------------------------------


_RECIPE = """\
schema_version: 1
name: global-audit-smoke
seed: 21
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./trainer-out
  count: 3
corpus:
  - type: local
    path: ./seed-words.txt
fonts:
  - path: {font}
    weight: 1.0
rendering:
  font_size_pt: {{ min: 14, max: 18 }}
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: word_crops
  padding_px: 4
"""

_SEED_WORDS = "\n".join(["ḃeaḋ", "ċeann", "ḋuine", "ḟear", "ġloine", "ṁaṫair"]) + "\n"


def _setup_recipe(tmp_path: Path, font_bytes: bytes) -> Path:
    """Materialize a tiny on-disk recipe + font + corpus.

    Mirrors ``test_cli_render_audit.py``'s setup so the end-to-end
    test exercises the production CLI path (``main([...])``) rather
    than constructing a Recipe in memory. That keeps the test
    realistic: the CLI is what users actually run.
    """

    font_path = tmp_path / "fonts" / "bundled.otf"
    font_path.parent.mkdir(parents=True, exist_ok=True)
    font_path.write_bytes(font_bytes)
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE.format(font=font_path), encoding="utf-8")
    (tmp_path / "seed-words.txt").write_text(_SEED_WORDS, encoding="utf-8")
    return rp


def _drive_render(rp: Path, out: Path, *extra: str) -> int:
    from pd_ocr_synth.cli import main

    return main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(out),
            "--seed",
            "21",
            "--workers",
            "1",
            *extra,
        ]
    )


def test_global_audit_mirror_round_trips_rendered_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writable_font_bytes: bytes,
) -> None:
    """A real ``render`` writes one row to *both* audit files.

    Asserts the global-mirror payload matches the per-output-dir
    payload byte-for-byte (same JSON shape, same SHA, same counts) so
    a downstream consumer of the aggregate gets identical forensics
    to a per-output-dir reader.
    """

    cache_root = tmp_path / "cache"
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))
    # The autouse ``_isolate_global_audit`` fixture in conftest sets
    # ``PD_OCR_SYNTH_NO_GLOBAL_AUDIT=1`` to keep tests from polluting
    # the real cache. For this test we want to exercise the mirror
    # end-to-end inside our isolated cache root, so unset it.
    monkeypatch.delenv(GLOBAL_AUDIT_DISABLE_ENV, raising=False)

    rp = _setup_recipe(tmp_path, writable_font_bytes)
    out = tmp_path / "trainer-out"
    rc = _drive_render(rp, out)
    assert rc == 0

    # Per-output-dir row.
    local_audit = out / AUDIT_FILENAME
    assert local_audit.is_file(), "per-output audit log missing"
    local_rows = read_audit_entries(local_audit)
    assert len(local_rows) == 1

    # Global-mirror row.
    global_audit = cache_root / GLOBAL_AUDIT_FILENAME
    assert global_audit.is_file(), "global audit aggregate missing"
    global_rows = read_audit_entries(global_audit)
    assert len(global_rows) == 1

    # Bytewise equality: the same AuditEntry was appended to both.
    assert local_rows[0] == global_rows[0]
    # Sanity: schema_version matches the constant so a future bump
    # surfaces the test as a forced update site.
    assert local_rows[0]["schema_version"] == AUDIT_SCHEMA_VERSION
    # Identity fields populated as expected.
    assert local_rows[0]["recipe_name"] == "global-audit-smoke"
    assert local_rows[0]["count"] == 3


def test_global_audit_mirror_off_when_no_audit_env_set(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writable_font_bytes: bytes,
) -> None:
    """``PD_OCR_SYNTH_NO_AUDIT=1`` suppresses the global mirror too.

    Documents that the broader audit-disable switch covers both
    files, so an operator turning audit off doesn't have to remember
    to also unset the global mirror — "off" means off everywhere.
    """

    cache_root = tmp_path / "cache"
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))
    monkeypatch.delenv(GLOBAL_AUDIT_DISABLE_ENV, raising=False)
    monkeypatch.setenv(AUDIT_DISABLE_ENV, "1")  # <- broad off

    rp = _setup_recipe(tmp_path, writable_font_bytes)
    out = tmp_path / "trainer-out"
    rc = _drive_render(rp, out)
    assert rc == 0

    # Neither file present.
    assert not (out / AUDIT_FILENAME).exists()
    assert not (cache_root / GLOBAL_AUDIT_FILENAME).exists()


def test_global_audit_mirror_off_when_only_global_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    writable_font_bytes: bytes,
) -> None:
    """``PD_OCR_SYNTH_NO_GLOBAL_AUDIT=1`` keeps per-output, drops mirror.

    The dedicated switch is the documented "I don't want my home dir
    to grow forever" escape hatch.
    """

    cache_root = tmp_path / "cache"
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))
    monkeypatch.setenv(GLOBAL_AUDIT_DISABLE_ENV, "1")
    monkeypatch.delenv(AUDIT_DISABLE_ENV, raising=False)

    rp = _setup_recipe(tmp_path, writable_font_bytes)
    out = tmp_path / "trainer-out"
    rc = _drive_render(rp, out)
    assert rc == 0

    # Per-output present, global absent.
    assert (out / AUDIT_FILENAME).is_file()
    assert not (cache_root / GLOBAL_AUDIT_FILENAME).exists()


def test_global_audit_mirror_appends_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The aggregate accumulates rows across multiple runs.

    Uses ``append_audit_entry`` directly (no font-fixture dependency)
    to keep this test fast and to lock the append-only contract that
    the runner relies on.
    """

    cache_root = tmp_path / "cache"
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_root))
    monkeypatch.delenv(GLOBAL_AUDIT_DISABLE_ENV, raising=False)

    audit_path = default_global_audit_path()

    def _entry(name: str, count: int) -> AuditEntry:
        return AuditEntry(
            timestamp=f"2026-05-06T0{count}:00:00Z",
            recipe_name=name,
            recipe_sha=None,
            output_dir=str(tmp_path / "out" / name),
            count=count,
            seed=count,
            workers=1,
            rendered=count,
            skipped=0,
            runtime_seconds=0.1 * count,
        )

    append_audit_entry(audit_path, _entry("first", 1))
    append_audit_entry(audit_path, _entry("second", 2))
    append_audit_entry(audit_path, _entry("third", 3))

    rows = read_audit_entries(audit_path)
    assert [r["recipe_name"] for r in rows] == ["first", "second", "third"]
    assert [r["count"] for r in rows] == [1, 2, 3]


def test_global_audit_round_trips_through_jsonl(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """One-line-per-row JSONL contract holds for the global file.

    Mirror path uses the same ``append_audit_entry`` writer as the
    per-output-dir path, so both share the "atomic line, trailing
    newline" guarantee. This locks that the global-aggregate file
    parses with the same ``read_audit_entries`` reader without any
    aggregate-specific shape divergence.
    """

    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv(GLOBAL_AUDIT_DISABLE_ENV, raising=False)

    path = default_global_audit_path()
    entry = AuditEntry(
        timestamp="2026-05-06T01:23:45Z",
        recipe_name="round-trip",
        recipe_sha="ab" * 32,
        output_dir="/tmp/out",
        count=10,
        seed=42,
        workers=2,
        rendered=10,
        skipped=0,
        runtime_seconds=1.5,
    )
    append_audit_entry(path, entry)

    raw = path.read_text(encoding="utf-8")
    # Single line + trailing newline.
    assert raw.count("\n") == 1
    assert raw.endswith("\n")
    # Parses back with all fields intact.
    parsed = json.loads(raw.rstrip("\n"))
    assert parsed["recipe_name"] == "round-trip"
    assert parsed["recipe_sha"] == "ab" * 32
    assert parsed["schema_version"] == AUDIT_SCHEMA_VERSION
