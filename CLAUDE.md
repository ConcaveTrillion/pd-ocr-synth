# CLAUDE — pd-ocr-synth

Recipe-driven synthetic OCR training-data generator; produces labeled
image+text pairs for historical and specialty typography (first target:
Cló Gaelach / early Irish). Status: spec-only — `src/` is nearly empty;
implement milestone-by-milestone against `docs/specs/`.
Architecture: `docs/specs/00-overview.md`.

## Commands

| target | does |
|---|---|
| `make setup AI=1` | `uv sync` + pre-commit hooks |
| `make test AI=1` | `uv run pytest -n auto` |
| `make lint AI=1` / `make lint-fix AI=1` | ruff + markdownlint via pre-commit (with auto-fix) |
| `make format AI=1` | ruff format then lint |
| `make ci AI=1` | setup → pre-commit → test → build |
| `make schema AI=1` | regenerate `docs/specs/recipe.schema.json` from pydantic models |
| `make fetch-fonts` | download Gaelic fonts (interactive — do not run non-interactively) |
| `make gaelic-preview AI=1` | render 50 preview samples for the Gaelic recipe (requires M07) |
| `make build AI=1` | wheel + sdist into `dist/` |

`AI=1` captures verbose output to `.ci-ai.log`; stdout shows `✅` on pass or
filtered failure sections on error. Remove `AI=1` only if you need full verbose
output for debugging.

## Rules

- Always run `make ci AI=1` before committing.
- Make targets first; fall back to `uv run …` only when no target exists.
- Never `python -m pytest`. Always `uv run pytest -n auto` or `make test`.
  Bare `python`/`python3`/`.venv/bin/python` miss the venv.
- Milestone-driven spec-first repo: implement against `docs/specs/` milestones
  in order; do not add scope beyond the current milestone.
- Before writing any code, read the relevant milestone spec end-to-end and propose a plan.
- If pydantic recipe models change, run `make schema` to regenerate `docs/specs/recipe.schema.json`.
- Never commit fonts — they are user-provided and license-sensitive; `make fetch-fonts` is interactive by design.
- Recipes live in `recipes/` as YAML; schema is in `docs/specs/recipe.schema.json`.
- Output contract is `pd-ocr-trainer`'s profile layout — confirm from
  `../pd-ocr-trainer/` before changing the output adapter.

## Specs

Full spec set in `docs/specs/00-N.md` (read in order). Roadmap milestones in `docs/plans/`.

## Sibling repos

- `../pd-ocr-trainer/` — consumes synth output; defines the profile directory layout this repo must match.
- `../pd-book-tools/` — shared OCR/image primitives (potential future dependency).

## GH issues

Cross-cut work tasks are tracked as GH issues in
**`ConcaveTrillion/ocr-container-meta`** (not in this repo's own tracker).
Plans under `docs/plans/` in the workspace root are synced there
via `/decompose-spec --sync`. Milestone naming: `spec: <plan-basename> (#N)`.

When shipping a plan task:

- Before starting: `gh issue view <N> --repo ConcaveTrillion/ocr-container-meta`
- After completing: `gh issue close <N> --repo ConcaveTrillion/ocr-container-meta`
- List open tasks:
  `gh issue list --repo ConcaveTrillion/ocr-container-meta --milestone "spec: <name> (#N)" --state open`

## docs/ folder

This repo follows the workspace docs/ template — see [`docs/README.md`](docs/README.md). Active
folders: `architecture/`, `decisions/`, `plans/`, `process/`, `research/`,
`runbooks/`, `specs/`, `templates/`, `usage/`, plus parallel `archive/`
subfolders.

**Superpowers redirect.** When a superpowers skill (e.g. `brainstorming`,
`writing-plans`) instructs you to save to `docs/superpowers/specs/<file>.md`
or `docs/superpowers/plans/<file>.md`, save to `docs/specs/<file>.md` or
`docs/plans/<file>.md` instead. There is no `docs/superpowers/` subdirectory
in this repo.
