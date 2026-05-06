# 01 — CLI

The CLI is the only supported entry point. Recipes are the only configuration
surface.

## Invocation

```
pd-ocr-synth <subcommand> [options]
```

Installed as a console script via `pyproject.toml`:
```
[project.scripts]
pd-ocr-synth = "pd_ocr_synth.cli:main"
```

## Subcommands

| Command | Purpose |
|---------|---------|
| `init <name>` | Scaffold a new recipe directory with a commented template |
| `list` | List recipes discovered on the recipe search path |
| `validate <recipe>` | Schema-check a recipe; verify fonts and corpus paths |
| `describe <recipe>` | Print resolved config + corpus stats (word count, etc.) |
| `fetch <recipe>` | Pre-fetch and cache all web/HF corpora for a recipe |
| `preview <recipe>` | Render N samples to a preview directory for visual review |
| `render <recipe>` | Full run; writes the dataset to the output destination |
| `publish <recipe>` | Upload rendered output to a Hugging Face dataset repo (see [10 — Publishing](10-publishing.md)) |
| `clean <recipe>` | Remove cached corpora (and optionally rendered output) |
| `audit <output-dir>` | Read back the per-render audit JSONL log written by `render` (M10 stretch) |

## Common options

These apply to most subcommands:

| Flag | Meaning |
|------|---------|
| `-c, --count N` | Override sample count from the recipe |
| `-o, --output PATH` | Override output destination |
| `-s, --seed N` | Override random seed (default from recipe, then 0) |
| `-w, --workers N` | Parallel render workers (default: CPU count) |
| `--cache-dir PATH` | Corpus cache root (default: `~/.cache/pd-ocr-synth/`) |
| `--no-cache` | Bypass corpus cache (force re-fetch) |
| `--dry-run` | Validate + plan only; no fetch, no render |
| `-v, --verbose` | More logging |
| `-q, --quiet` | Errors only |

## Recipe resolution

A recipe argument may be:

1. A path to a YAML file (`./my-recipe.yaml`)
2. A name resolvable on the recipe search path (`gaelic` → `recipes/gaelic.yaml`)

The search path is, in order:

1. `$PD_OCR_SYNTH_RECIPES` (colon-separated)
2. `./recipes/` relative to CWD
3. `<package>/recipes/` shipped with the install

`list` walks this path and prints `name → path`.

## Examples

```bash
# Scaffold a new recipe
pd-ocr-synth init fraktur
# → creates ./recipes/fraktur/recipe.yaml + README

# Validate the Gaelic recipe ships
pd-ocr-synth validate gaelic

# Pre-fetch corpora once (writes to ~/.cache/pd-ocr-synth/)
pd-ocr-synth fetch gaelic

# Render 200 samples to a temp dir for visual inspection
pd-ocr-synth preview gaelic --count 200 --output /tmp/gaelic-preview

# Full render (50k samples per the recipe) into the trainer profile dir
pd-ocr-synth render gaelic

# Override count and output for a quick sanity run
pd-ocr-synth render gaelic -c 500 -o /tmp/gaelic-500

# Publish the rendered output to a Hugging Face dataset repo
pd-ocr-synth publish gaelic                            # uses recipe defaults
pd-ocr-synth publish gaelic --repo me/pd-ocr-synth-ga  # explicit repo
pd-ocr-synth publish gaelic --tag v2026.05.05          # pin a release
pd-ocr-synth publish gaelic --dry-run                  # preview only
```

## Exit codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Generic error (also: `lint --strict` with warnings present) |
| 2 | CLI usage error (bad flags, unknown subcommand) |
| 3 | Recipe validation failed |
| 4 | Corpus fetch failed |
| 5 | Render failed (partial output may exist) |
| 6 | Output destination invalid or unwritable |
| 7 | Publish failed (auth, network, or repo-state error) |

`lint --strict` upgrades a clean run with warnings (validate or lint)
from 0 to 1 so it can be used as a CI / pre-commit gate. Validation
errors keep their stricter code 3 either way — strict never
_downgrades_ a stricter exit code.

## Logging

- Default: human-readable progress to stderr; results summary to stdout.
- `--log-format json` emits one JSON record per line (for piping into
  `jq`, log collectors, or wrapping scripts).
- Render reports rate (samples/sec), ETA, and a final manifest path.
