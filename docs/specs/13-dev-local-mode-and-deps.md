# 13 — Dev-local mode and dependency upgrades

This spec captures a workspace-wide hazard around `uv sync` and how
`pd-ocr-synth` must behave once the Makefile gains an `upgrade-deps`
target (already stubbed in M01) so it does not silently undo a
contributor's local development setup.

Status: **spec only**. Implementation is **deferred until the Makefile
`dev-local` recipe lands**; the requirement is hooked into M01 (see
[`../roadmap/01-dev-tooling.md`](../roadmap/01-dev-tooling.md)) so that
whichever milestone introduces the dev-local recipe across the
workspace also lands the detection logic here.

## Background

Across the workspace, several `pd-*` repos can be put into a
**dev-local** mode where the project's `.venv` resolves sibling `pd-*`
projects as **editable installs from the local checkout** (instead of
the canonical published wheels), pins to **GPU extras**, or pulls
`python-doctr` from a git ref. This is how cross-repo work happens —
e.g. exercising an unreleased `pd-book-tools` change from
`pd-ocr-trainer`, or running a synth recipe against a local
`pd-ocr-trainer` profile writer.

The hazard: a plain `uv sync --group <something>` (the last step of
`make upgrade-deps`) will resolve against `pyproject.toml` /
`uv.lock` and **silently revert** that environment back to the
canonical, published, CPU-only configuration. Editable sibling installs
disappear, GPU extras drop out, doctr-from-git collapses to the pinned
release. The contributor only finds out when their next test run
unexpectedly fails to pick up changes in a sibling repo.

`pd-ocr-synth` does not (today) install editable sibling pd-* repos —
its only direct workspace coupling is the **output contract** with
`pd-ocr-trainer`'s `dataset_store.py`, not a runtime dependency.
However:

1. Future milestones may add an editable `pd-book-tools` or
   `pd-ocr-trainer` install for integration tests (e.g. round-tripping
   a synth recipe through the trainer's dataset reader).
2. The standardized fix should be uniform across all `pd-*` Makefiles
   so contributors see the same UX everywhere. Diverging here would
   itself be the bug.

So this spec captures the contract now, even though the synth repo
won't exercise the dev-local code paths until a later milestone needs
editable siblings.

## Required behavior

The behavior below applies to **whatever `upgrade-deps`-style target
runs `uv sync`** in this repo's Makefile. It is written so the same
text drops cleanly into peer repos (`pd-book-tools`, `pd-ocr-cli`,
`pd-ocr-trainer`, `pd-ocr-labeler`, `pd-ocr-labeler-spa`,
`pd-prep-for-pgdp`, `pd-png-optimizer`).

### 1. Detect dev-local vs canonical before `uv sync`

Before the implicit `uv sync` step in `upgrade-deps` (or any other
target that calls `uv sync` on an existing `.venv`), the Makefile MUST
detect whether the current `.venv` is in **dev-local** or **canonical**
mode and branch on the result. No `uv sync` may run unconditionally
against an existing `.venv` from `upgrade-deps`.

### 2. Detection mechanism (ordered fallbacks)

Detection probes in this order, taking the first signal found:

1. **Primary — sibling editable probe.** Run
   `uv pip show pd-book-tools` and look for an
   `Editable project location:` line whose value is a path under the
   workspace root (i.e. a sibling `pd-book-tools/` checkout).
   `pd-book-tools` is the cross-repo contract anchor: every other pd-*
   that participates in dev-local mode pins it editable, so its
   editable status is a reliable proxy for the whole environment.
   If `uv pip show pd-book-tools` fails or returns no
   `Editable project location:` line, fall through to (2).
2. **Fallback — venv marker file.** If `.venv/pd-dev-local` exists
   (a zero-byte sentinel written by `make upgrade-deps-local` /
   the dev-local setup recipe), treat as dev-local. This covers
   environments where `pd-book-tools` legitimately is not installed
   (e.g. a synth-only contributor) but the developer has explicitly
   opted into dev-local extras (GPU, doctr-from-git).
3. **Last resort — env var.** If `PD_DEV_LOCAL=1` is set in the
   shell, treat as dev-local. Useful for CI runs that
   programmatically opt into dev-local without pre-creating the
   marker.

If none of (1)/(2)/(3) signal dev-local, the venv is **canonical**.

### 3. UX contract

- **Default `make upgrade-deps`:** when dev-local is detected, the
  target MUST refuse with a clear message and exit non-zero **before**
  running `uv sync`. Recommended message:

  > Detected dev-local venv (editable pd-book-tools / marker /
  > PD_DEV_LOCAL). `make upgrade-deps` would revert this environment
  > to canonical published deps. Use `make upgrade-deps-local`
  > instead, or remove the marker / unset PD_DEV_LOCAL to opt into a
  > canonical refresh.

  The message MUST name the detection signal that triggered the
  refusal so the contributor can remediate it.

- **Sibling target `make upgrade-deps-local`:** runs `uv lock
  --upgrade` then re-applies the dev-local installation recipe
  (editable siblings, GPU extras, doctr-from-git as applicable to
  this repo). For pd-ocr-synth specifically the dev-local recipe is
  TBD until a milestone needs it; until then `upgrade-deps-local`
  may simply alias the canonical sync with a banner noting that
  no dev-local extras apply yet.

- **Canonical mode:** behavior of `make upgrade-deps` is unchanged
  from today — `uv lock --upgrade` followed by
  `uv sync --group all-dev`.

### 4. Cross-platform

Detection and refusal logic MUST work on Linux, macOS, and any
Windows shell currently supported by the Makefile (treat `bash` as
the lowest common denominator; do not depend on GNU-only utilities
beyond `grep` / `test`). The marker file path uses POSIX-style
`.venv/pd-dev-local`; on Windows uv venvs the equivalent path under
`.venv\` is acceptable.

## Why pd-book-tools as the anchor

The probe deliberately targets `pd-book-tools` rather than this
repo's own metadata because:

- `pd-book-tools` is the foundation library every other `pd-*` depends
  on, so its editable install is the single most reliable signal that
  the venv is wired for cross-repo work.
- Anchoring on a sibling means the same Makefile snippet works
  identically across the workspace — no per-repo customization of the
  primary probe, which is the whole point of standardizing.
- Probing `pd-ocr-synth` itself is uninformative: a contributor doing
  pure synth work may still install `pd-ocr-synth` editable inside
  its own venv without that implying any dev-local intent.

The marker file and env var exist to cover legitimate cases where
`pd-book-tools` is absent but dev-local intent is real.

## Out of scope for this spec

- The exact shell snippet for the detection (left to the Makefile
  PR that lands the implementation; the contract above is the spec).
- The `dev-local` recipe contents for pd-ocr-synth (no editable
  siblings yet — added when a future milestone introduces them).
- Migrating other workspace repos; this spec describes pd-ocr-synth's
  share of the standardized fix.
