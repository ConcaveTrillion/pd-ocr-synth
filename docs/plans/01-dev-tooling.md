# M01 — Dev tooling parity

**Goal:** every developer command another workspace project supports
works in this one. After this milestone, the project has zero behavior
but full development ergonomics — a contributor can `make setup &&
make test && make lint` and get green on the empty package.

This milestone is intentionally heavy-weighted. Every later milestone
benefits from skipping ahead until tooling is solid.

## Deliverables

### `Makefile` parity with `pd-ocr-trainer`

Adopt the same target catalog, with the same emoji + help-text style:

- [ ] `help` — auto-generated from `##` comments
- [ ] `setup` — `uv sync --group all-dev` then
      `uv run pre-commit install` (the dev-env target)
- [ ] `install` / `uninstall` — `uv tool install --reinstall .` /
      `uv tool uninstall pd-ocr-synth` (puts the CLI on PATH for
      end-users, separate from dev setup)
- [ ] `remove-venv`, `reset`, `reset-full`, `upgrade-deps`
- [ ] `upgrade-deps` MUST honor the dev-local detection contract in
      [13 — Dev-local mode and dependency upgrades](../specs/13-dev-local-mode-and-deps.md):
      probe `uv pip show pd-book-tools` for `Editable project location`,
      fall back to a `.venv/pd-dev-local` marker, last-resort
      `PD_DEV_LOCAL=1` env var; refuse-with-message when dev-local is
      detected, leave canonical-mode behavior unchanged, and ship a
      sibling `upgrade-deps-local` target. Deferred until the
      Makefile / `dev-local` recipe lands; this milestone owns it.
- [ ] `test`, `test-verbose`, `test-single` (parameterized)
- [ ] `lint`, `lint-fix`, `format`, `pre-commit-check`
- [ ] `ci` — what GitHub Actions runs (lint + test)
- [ ] `build`, `clean`, `clean-cache`
- [ ] `release-patch`, `release-minor`, `release-major`, `_do-release`
- [ ] **Project-specific:** `gaelic-preview` (already in stub),
      `fetch-fonts` (wraps `scripts/fetch-fonts-gaelic.sh`)

Reference: `pd-ocr-trainer/Makefile`. Keep the help-text formatting
identical so the look is consistent across the workspace.

### `pyproject.toml` dependency groups

Match the peer pattern of multiple optional groups:

- [ ] `[dependency-groups]` (uv-style) or
      `[project.optional-dependencies]`:
  - `testing` — pytest, pytest-cov, pytest-xdist
  - `linting` — ruff, pre-commit
  - `all-dev` — superset; what `make setup` uses
- [ ] Pinned `python-doctr`-equivalent: pin `huggingface_hub`,
      `datasets`, `pillow`, `uharfbuzz`, `freetype-py`, `httpx`,
      `pyyaml`, `pydantic`, `beautifulsoup4`, `lxml`, `numpy`,
      `opencv-python`, `tqdm` in `dependencies`.
- [ ] Add `[tool.coverage]` config (mirror peer).

### `.pre-commit-config.yaml`

- [ ] Hooks matching peer projects:
  - `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`,
    `check-added-large-files`
  - `ruff` (lint + format)
- [ ] `make pre-commit-check` wires it to CI.

### `src/pd_ocr_synth/` skeleton

- [ ] `__init__.py` with `__version__`
- [ ] `__main__.py` for `python -m pd_ocr_synth`
- [ ] `cli.py` — argparse / typer stub with `--version`, `--help`,
      and unimplemented subcommands that print "not implemented yet"
      and exit 2.
- [ ] Console script registered in `pyproject.toml`:
      `pd-ocr-synth = "pd_ocr_synth.cli:main"`.

### `tests/`

- [ ] `tests/conftest.py` with shared fixtures (`tmp_path`-style helpers)
- [ ] `tests/test_smoke.py` — `pd-ocr-synth --version` returns 0
- [ ] `tests/test_cli.py` — every subcommand prints help with `--help`
- [ ] Markers config in `pyproject.toml`: `unit`, `integration`, `slow`,
      `gpu`

### CI: `.github/workflows/ci.yml`

- [ ] Mirror `pd-ocr-trainer`'s workflow exactly:
  - Trigger: PRs and pushes to `main`
  - Setup uv, sync deps
  - Run `make ci`
- [ ] Cache uv downloads.
- [ ] Single Linux runner (no GPU, no Mac/Windows for now).

### Setup is `make setup`

We deliberately do **not** ship a `dev-env-setup.sh`. The peer
`pd-ocr-trainer/dev-env-setup.sh` is a legacy artifact from before
the uv-based Makefile (it uses `python3 -m venv` + `pip install
-r requirements.txt`). For pd-ocr-synth, `make setup` is the
complete dev environment setup, and `make fetch-fonts` is the
optional follow-up. `make install` is reserved for actually
installing the CLI as a uv tool — semantic separation that matches
`pd-ocr-labeler`'s convention. A single source of truth (the
Makefile) keeps developer expectations from drifting.

### Devcontainer integration (optional but matches workspace)

- [ ] If the workspace devcontainer auto-discovers projects, ensure
      `pd-ocr-synth` is included in any post-create steps. Otherwise
      defer to a workspace-level change.

## Validation criteria

```bash
git clone .../pd-ocr-synth
cd pd-ocr-synth
make setup        # green; venv built; pre-commit installed
make test         # green; smoke + cli help tests pass
make lint         # green
make build        # green; produces dist/
pd-ocr-synth --version   # prints "pd-ocr-synth 0.0.1"
pd-ocr-synth render gaelic
# → "render: not implemented yet" (exit 2)
```

CI on a freshly opened PR runs `make ci` and is green.

## Out of scope

- Recipe loading, validation, rendering — all deferred to M02+.
- Any actual subcommand behavior beyond stubs.
- GPU CI — irrelevant for synth (CPU rendering).

## Risks / open items

- **`uv` group syntax drift.** If peer projects already migrated from
  `[project.optional-dependencies]` to `[dependency-groups]` (uv 0.5+),
  match the more recent style.
- **Pre-commit hook drift.** Lock to the same hook versions as peers
  to keep formatting consistent across the workspace.
- **Codecov / coverage upload.** If peers do this in CI, mirror it;
  if not, skip.
