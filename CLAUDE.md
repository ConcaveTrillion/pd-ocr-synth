# CLAUDE — pd-ocr-synth

Recipe-driven synthetic OCR training-data generator; produces labeled image+text pairs for historical and specialty
typography (first target: Cló Gaelach / early Irish). Status: spec-only — `src/` is nearly empty; implement
milestone-by-milestone against `docs/specs/`.
Architecture: `docs/specs/00-overview.md`.

## Commands

| target | does |
|---|---|
| `make setup` | uv sync + pre-commit hooks |
| `make test` | pytest -n auto (parallelized) |
| `make lint` | ruff + markdownlint via pre-commit |
| `make lint-fix` | auto-fix Python + Markdown lint |
| `make format` | ruff format then lint |
| `make ci` | setup → pre-commit → test → build |
| `make schema` | regenerate `docs/specs/recipe.schema.json` from pydantic models |
| `make fetch-fonts` | download Gaelic fonts (interactive — do not run non-interactively) |
| `make gaelic-preview` | render 50 preview samples for the Gaelic recipe (requires M07) |
| `make build` | build wheel + sdist into `dist/` |

## Rules

- Make targets first; fall back to `uv run …` only when no target exists.
- Never `python -m pytest`. Always `make test` or `uv run pytest -n auto`.
- Milestone-driven spec-first repo: implement against `docs/specs/` milestones in order; do not add scope beyond the
  current milestone.
- Before writing any code, read the relevant milestone spec end-to-end and propose a plan.
- If pydantic recipe models change, run `make schema` to regenerate `docs/specs/recipe.schema.json`.
- Never commit fonts — they are user-provided and license-sensitive; `make fetch-fonts` is interactive by design.
- Recipes live in `recipes/` as YAML; the schema is in `docs/specs/recipe.schema.json`.
- Output contract is `pd-ocr-trainer`'s profile layout — confirm from `../pd-ocr-trainer/` before changing the output adapter.

## Specs

Full spec set in `docs/specs/00-N.md` (read in order). Roadmap milestones in `docs/roadmap/`.

## Sibling repos

- `../pd-ocr-trainer/` — consumes synth output; defines the profile directory layout this repo must match.
- `../pd-book-tools/` — shared OCR/image primitives (potential future dependency).

## Spec lifecycle

Design spec files (`docs/specs/<date>-<topic>-design.md`) live in `docs/specs/` while the
milestone's chore issues are open. When the last chore closes and the implementation ships,
move the file to `docs/architecture/` and commit. See workspace `docs/conventions.md`.
