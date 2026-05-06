"""Per-render audit log (M10 stretch).

Each ``run_recipe`` invocation appends one JSONL line to a per-output
audit file at ``<output_dir>/_audit.jsonl`` so a downstream consumer
of a rendered dataset can trace it back to the recipe + seed + runtime
that produced it. Useful when:

- a model trained from a specific dataset version is later debugged
  ("which recipe was this trained on?");
- two render runs produced suspiciously different outputs and you
  want to diff their seeds / SHAs;
- a corrupted dataset needs the original render's wall-time to match
  it against logs from other tools.

The format is intentionally append-only JSONL — one render → one line
— so concurrent runs into the same destination (uncommon, but legal
during ``--resume``) don't clobber each other and a tail-friendly
text-format means ``less`` / ``jq`` work without extra tooling.

## What lands in each entry

- ``timestamp``: ISO-8601 UTC string at the moment the entry is
  finalized (post-render).
- ``recipe_name``: ``recipe.name`` (free-text identifier).
- ``recipe_sha``: SHA-256 of the recipe source file bytes when
  ``recipe.source_path`` is set, else ``None``. We hash the on-disk
  YAML so a recipe that was edited mid-run is captured at its
  rendered-version snapshot. (The publish step uses a separate
  ``content_sha`` over the staging dir; that one is dataset-bytes-
  invariant. The render audit is recipe-bytes-invariant.)
- ``output_dir``: absolute path the writer wrote into.
- ``count``: effective sample count (post ``--count`` override).
- ``seed``: effective seed (post ``--seed`` override).
- ``workers``: worker pool size as the runner saw it (the
  resolution-from-flags step happens earlier in the CLI).
- ``rendered`` / ``skipped``: counts from the ``RunResult``.
- ``runtime_seconds``: wall time as the writer's stats recorded it.
- ``schema_version``: ``1`` — bumps on shape changes so a future
  reader can branch.

## Disabling

- Pass ``audit=False`` to the public ``append_audit_entry`` /
  ``write_audit_for_run`` calls.
- Set the env var ``PD_OCR_SYNTH_NO_AUDIT=1`` in the environment to
  globally suppress audit emission. Useful in tests / sensitive runs
  where filesystem state must be reproducible byte-for-byte without
  the timestamped audit drift.

## Determinism note

The ``timestamp`` field is wall-clock and therefore non-deterministic
across runs. A test that asserts byte-for-byte equality on the audit
file should either freeze time or assert on the parsed-back JSON
shape — see ``tests/test_audit.py`` for the parse-back pattern.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

# Per-output-dir filename. Underscore-prefix mirrors the project's
# convention of writer-internal sidecars (``_snapshot.json``,
# ``_stats.json``) — keeps it from looking like a labels file to the
# trainer's loader.
AUDIT_FILENAME = "_audit.jsonl"

# Env var that globally disables audit emission. Honored by
# :func:`should_emit_audit`. Set to any truthy value (``1``, ``true``,
# ``yes``) to suppress.
AUDIT_DISABLE_ENV = "PD_OCR_SYNTH_NO_AUDIT"

# Schema-version constant for the JSONL row. Bump when the on-disk
# shape changes; readers should round-trip an unknown version into a
# best-effort dict and warn rather than crash.
AUDIT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class AuditEntry:
    """One line in the audit log.

    Fields are ordered so a manual ``cat _audit.jsonl`` is readable
    left-to-right: identity (timestamp, recipe), provenance (SHA,
    seed), then the run outcome. ``schema_version`` lives at the
    bottom because a forward-compatible reader keys off it but a human
    reading the line cares about the content first.
    """

    timestamp: str
    recipe_name: str
    recipe_sha: str | None
    output_dir: str
    count: int
    seed: int
    workers: int
    rendered: int
    skipped: int
    runtime_seconds: float
    schema_version: int = AUDIT_SCHEMA_VERSION

    def to_jsonl(self) -> str:
        """Serialize to a single JSONL line (no trailing newline)."""

        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=False)


def should_emit_audit(*, audit: bool, env: dict[str, str] | None = None) -> bool:
    """Resolve whether to emit an audit entry.

    The caller passes ``audit=False`` for a CLI ``--no-audit`` or a
    test that wants determinism. The env var
    :data:`AUDIT_DISABLE_ENV` overrides ``audit=True`` so an operator
    can blanket-suppress audit globally without touching any callsite.

    ``env`` defaults to ``os.environ``; tests can pass a dict for
    isolation.
    """

    if not audit:
        return False
    environ = os.environ if env is None else env
    raw = environ.get(AUDIT_DISABLE_ENV, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return False
    return True


def compute_recipe_sha(recipe_source_path: Path | None) -> str | None:
    """SHA-256 of the recipe YAML's bytes, or ``None`` if unknown.

    The audit entry needs a stable identifier for "which version of
    the recipe ran". ``recipe.name`` is free-text and rotates under
    the author's hand; the source bytes are a tighter fingerprint.

    We hash the file bytes verbatim — the same convention as
    :mod:`pd_ocr_synth.publish.content_sha`'s per-file step, but
    one-level (the audit doesn't need a tree-level digest because a
    recipe is a single YAML file). Includes any whitespace or comment
    edits the author made, so two runs of "the same recipe" with one
    comment fixed produce different audit SHAs — this is intentional;
    the audit is for traceability, not for change-detection.

    Returns ``None`` when ``recipe_source_path`` is missing (an
    in-memory recipe constructed via ``Recipe(...)`` rather than
    ``load_recipe``); callers should record the SHA as ``null`` in
    that case rather than fabricating a value.
    """

    if recipe_source_path is None:
        return None
    if not recipe_source_path.is_file():
        return None
    digest = hashlib.sha256()
    with recipe_source_path.open("rb") as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def now_timestamp() -> str:
    """ISO-8601 UTC timestamp (second precision, ``Z`` suffix).

    Centralized so tests can monkeypatch one symbol and the audit
    surface stays decoupled from any specific clock implementation.
    """

    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def append_audit_entry(audit_path: Path, entry: AuditEntry) -> None:
    """Append ``entry`` to ``audit_path`` as one JSONL line.

    Creates the parent directory if missing (the writer creates
    ``output_dir`` on its own; the audit file lives alongside the
    other sidecars there). Opens in text-mode append so concurrent
    writes don't truncate, with explicit UTF-8 to keep non-ASCII
    recipe names readable.

    The trailing newline is part of the JSONL contract: a partial
    write that terminates mid-line is recoverable by ``jq -c .`` /
    ``jsonlines`` readers, which skip the trailing junk.
    """

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("a", encoding="utf-8") as fh:
        fh.write(entry.to_jsonl())
        fh.write("\n")


def read_audit_entries(audit_path: Path) -> list[dict]:
    """Parse the audit JSONL back into a list of dicts.

    Convenience for tests + future ``pd-ocr-synth audit`` subcommand
    (out of scope for this chunk). Skips empty / whitespace-only
    lines; raises :class:`json.JSONDecodeError` on malformed JSON so a
    corrupted audit surfaces loudly rather than silently dropping
    rows.
    """

    if not audit_path.is_file():
        return []
    out: list[dict] = []
    for raw in audit_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        out.append(json.loads(raw))
    return out
