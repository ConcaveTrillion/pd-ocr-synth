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
| `lint <recipe>` | Run `validate` + heuristic lint checks (M10) |
| `describe <recipe>` | Print resolved config + corpus stats (word count, etc.) |
| `schema` | Emit the recipe JSON Schema (default: write `docs/specs/recipe.schema.json`) |
| `fetch <recipe>` | Pre-fetch and cache all web/HF corpora for a recipe |
| `preview <recipe>` | Render N samples to a preview directory for visual review |
| `render <recipe>` | Full run; writes the dataset to the output destination |
| `publish <recipe>` | Upload rendered output to a Hugging Face dataset repo (see [10 — Publishing](10-publishing.md)) |
| `clean <recipe>` | Remove cached corpora (and optionally rendered output) |
| `audit [output-dir]` | Read back the per-render audit JSONL log written by `render` (M10) |

## Render-family options

Apply to `preview` and `render` (and to `publish` via
`--render-first`):

| Flag | Meaning |
|------|---------|
| `-c, --count N` | Override sample count from the recipe |
| `-o, --output PATH` | Override output destination |
| `-s, --seed N` | Override random seed (default from recipe, then 0) |
| `-w, --workers N` | Parallel render workers (default: CPU count) |
| `--cache-dir PATH` | Corpus cache root (default: `~/.cache/pd-ocr-synth/`) |
| `--no-cache` | Bypass corpus cache (force re-fetch) |
| `--dry-run` | Validate + plan only; no fetch, no render |

`fetch` is corpus-only — it walks `recipe.corpus` and warms the cache
— so it accepts only the two cache-related flags (`--cache-dir`,
`--no-cache`). The other render-family flags above (`--count`,
`--output`, `--seed`, `--workers`, `--dry-run`) have no meaning at
fetch time and are not accepted on the `fetch` subparser.

## Per-subcommand flags

Flags that aren't part of the render-family set above. Each table is
the canonical surface for that subcommand — a flag listed here must
exist in the parser, and a flag in the parser that isn't listed here
must be added (a meta-test in `tests/test_spec_docs.py` enforces both
directions).

### `init <name>`

| Flag | Meaning |
|------|---------|
| `--dir PATH` | Directory to scaffold under (default: `./recipes`) |
| `--force` | Overwrite an existing recipe with the same name |

### `validate <recipe>`

| Flag | Meaning |
|------|---------|
| `--offline` | Skip network-touching checks |

### `lint <recipe>`

| Flag | Meaning |
|------|---------|
| `--offline` | Skip network-touching checks (forwarded to validate) |
| `--json` | Emit a JSON object of validation + lint issues (machine-readable) |
| `--strict` | Treat lint warnings as failures: exit 1 if any warning is present (validation errors still take precedence with exit 3); use as a CI / pre-commit gate |

### `describe <recipe>`

| Flag | Meaning |
|------|---------|
| `--format {text,json}` | Output format (default: text) |

### `schema`

| Flag | Meaning |
|------|---------|
| `-o, --output PATH` | Write the schema to this path instead of stdout |

### `preview <recipe>`

| Flag | Meaning |
|------|---------|
| `--no-degrade` | Skip the recipe's degradation pipeline; output raw render only |

### `render <recipe>`

| Flag | Meaning |
|------|---------|
| `--force` | Clear destination before render |
| `--resume` | Resume an interrupted render (mutually exclusive with `--force`) |
| `--no-audit` | Suppress the per-run audit JSONL line under `<output>/_audit.jsonl` |

### `publish <recipe>`

| Flag | Meaning |
|------|---------|
| `--repo OWNER/NAME` | Override the recipe's `publish.hf_dataset.repo` |
| `--private` / `--public` | Force visibility (overrides recipe default) |
| `--license SPDX` | Override the recipe's dataset license |
| `--tag NAME` | Pin the upload to a release tag |
| `--message TEXT` | Custom commit message for the dataset upload |
| `--token TOKEN` | HF auth token (overrides env / cached login) |
| `-o, --output PATH` | Override the local render output path being uploaded |
| `--render-first` | Run `render` before publishing (chained step) |
| `--no-create` | Refuse to create the repo if it doesn't exist (default: create on first publish) |
| `--dry-run` | Preview the publish plan without uploading |

### `clean <recipe>`

| Flag | Meaning |
|------|---------|
| `--cache-dir PATH` | Cache root (default: `$PD_OCR_SYNTH_CACHE` or `~/.cache/pd-ocr-synth`) |

### `audit [output-dir]`

The positional `output_dir` is required *unless* `--global` or
`--audit-file` is passed.

| Flag | Meaning |
|------|---------|
| `--audit-file PATH` | Read audit entries from this JSONL path instead of `<output_dir>/_audit.jsonl`; useful for archived or aggregated audit logs |
| `--global` | Read entries from the global aggregate at `<cache_root>/audit.jsonl` (default `~/.cache/pd-ocr-synth/`); mutually exclusive with `--audit-file` |
| `--json` | Emit a JSON array of entries (machine-readable) instead of the table |
| `--limit N` | Only show the most recent N entries (tail behaviour) |
| `--since ISO` | Only show entries with timestamp >= this ISO-8601 value (e.g. `2026-05-06` or `2026-05-06T10:30:00Z`); applied before `--limit` |
| `--until ISO` | Only show entries with timestamp <= this ISO-8601 value (same parser as `--since`); applied before `--limit` |
| `--recipe-sha PREFIX` | Only show entries whose `recipe_sha` starts with this hex prefix (case-insensitive); entries with a null sha are excluded |
| `--summary` | Print aggregate statistics over the matched entries instead of the per-row table; combine with `--json` for a single JSON object |

## Audit log schema

`render` appends one JSONL line to `<output_dir>/_audit.jsonl` (and,
unless `PD_OCR_SYNTH_NO_GLOBAL_AUDIT=1` is set, mirrors the same line
to `<cache_root>/audit.jsonl`) per invocation. The shape below is the
contract `audit` reads back; it is also what tools like `jq` /
`pandas.read_json(lines=True)` will see.

The on-disk fields appear in the order shown — left-to-right
readability for `cat _audit.jsonl` is intentional: identity first,
provenance second, run outcome last, schema-version anchor at the
end.

A meta-test in `tests/test_spec_docs.py` enforces that this table and
the `AuditEntry` dataclass in `src/pd_ocr_synth/audit.py` stay in
sync: adding a field to the dataclass without listing it here (or
vice-versa) is a hard test failure.

| Field | Type | Since | Description |
|-------|------|-------|-------------|
| `timestamp` | string | v1 | ISO-8601 UTC, second precision, `Z` suffix (e.g. `2026-05-06T10:30:00Z`); finalized after the render completes |
| `recipe_name` | string | v1 | `recipe.name` verbatim (free-text identifier) |
| `recipe_sha` | string \| null | v1 | SHA-256 hex of the on-disk recipe YAML bytes, or `null` when the recipe was constructed in-memory (no `source_path`) |
| `output_dir` | string | v1 | Absolute path the writer wrote into |
| `count` | integer | v1 | Effective sample count (post `--count` override) |
| `seed` | integer | v1 | Effective seed (post `--seed` override) |
| `workers` | integer | v1 | Worker pool size as the runner saw it |
| `rendered` | integer | v1 | Samples actually rendered (from `RunResult`) |
| `skipped` | integer | v1 | Samples skipped, e.g. by `--resume` (from `RunResult`) |
| `runtime_seconds` | float | v1 | Wall time of the render, from the writer's stats |
| `schema_version` | integer | v1 | On-disk shape version; bumps on shape changes (current: `1`) |

### Forward-compatibility policy

The `audit` reader skips rows whose `schema_version` does not match
the version it understands and emits an `AuditSchemaVersionWarning`
naming the encountered version. A v1 reader cannot trust v2 field
semantics (a future bump might rename `count` to `planned_count` or
change `runtime_seconds` from float-seconds to integer-milliseconds)
so silently summing them would produce wrong totals. Rows missing
`schema_version` entirely are treated as legacy v1 — the field was
introduced in v1, so absence implies pre-versioning or hand-edited
input.

## Lint codes

Every issue surfaced by `lint <recipe>` (beyond the validation errors
forwarded from `validate`) carries a stable `code` field. The full
catalog is below — these codes appear verbatim in the human-readable
output, the `--json` payload, and structured logs, so they're safe to
grep / filter on.

A meta-test in `tests/test_spec_docs.py` enforces that this table and
the `LINT_CODES` constant in `src/pd_ocr_synth/lint.py` stay in sync:
adding a new lint helper without listing it here (or vice-versa) is a
hard test failure.

| Code | Trigger |
|------|---------|
| `lint_degradation_always_certain` | Every degradation stage has `probability=1.0`; every sample receives identical augmentation, defeating the point of randomized augmentation. |
| `lint_single_font` | Recipe declares one or zero fonts; models trained on a single typeface tend to overfit to its rasterization quirks. |
| `lint_no_text_transforms` | Recipe declares no `text_transforms`; suspicious for historical-typography targets where the corpus is in modern spelling. |
| `lint_low_sample_count` | `output.count` is below 100, the suggested minimum for a useful training run; usually a forgotten `--count` flag or misconfigured recipe. |
| `lint_seed_default` | `seed` is left at the schema default of `0`; every render of every fork produces bit-identical samples. Set an explicit seed in the recipe. |
| `lint_zero_weight_font` | A declared font has `weight=0.0` and will never be sampled; almost always a typo or a leftover from a temporarily disabled font. |
| `lint_all_optional_fonts` | Every font is marked `optional=True`; if no font files are present on disk the loader ends up with an empty font set and rendering fails. |

All lint issues use `severity="warning"`. A recipe that flunks every
lint check still renders correctly; lint alone never exits non-zero
unless `--strict` is passed (see exit codes below).

## Validation codes

Every issue surfaced by `validate <recipe>` (and the validation phase
of `lint <recipe>`, `render`, `preview`, `fetch`, `publish`) carries a
stable `code` field in addition to `severity` and a free-form
`message`. The codes appear verbatim in the human-readable
`[severity] [code] location: message` output, the `--json` payload,
and structured logs, so they're safe to grep / filter on.

A meta-test in `tests/test_spec_docs.py` enforces that this table and
the `VALIDATION_CODES` constant in `src/pd_ocr_synth/validation.py`
stay in sync: adding a new emission site without listing it here (or
vice-versa) is a hard test failure. A second test in
`tests/test_validation.py` asserts every code emitted by
`validate_recipe` belongs to `VALIDATION_CODES`, so a new code can't
ship undocumented.

| Code | Severity | Trigger |
|------|----------|---------|
| `schema_version_unsupported` | error | `schema_version` is not in `SUPPORTED_SCHEMA_VERSIONS`. Defensive — pydantic normally rejects this on load. |
| `output_destination_unresolved` | error | `output.destination` still contains `${VAR}` or starts with `~`; the env var isn't set, or the recipe was passed unresolved. |
| `output_destination_unwritable` | error | No writable ancestor exists for `output.destination` — typically a permission problem or a path under a non-existent mount. |
| `output_layout_mode_mismatch` | error | `output.mode` and `layout.mode` aren't a valid pairing (recognition needs `word_crops`/`lines`; detection needs `paragraphs`/`pages`). See [spec 08](08-output-format.md). |
| `optional_font_missing` | warning | A font marked `optional: true` doesn't exist on disk; it will be skipped at render time. |
| `font_missing` | error | A required (non-optional) font path doesn't exist. |
| `font_unreadable` | error | The font file exists but `open_font` couldn't parse it (corrupt header, unsupported format). |
| `font_empty` | error | The font opened but reports zero glyphs or an empty cmap; almost always a placeholder or stub file. |
| `local_corpus_missing` | error | A `type: local` corpus references a `path:` that doesn't exist on disk. |
| `corpus_provider_not_implemented` | error | A `corpus[].type` is structurally valid but no runtime provider is registered for it (typically a future provider type listed in spec 04 that hasn't shipped yet). |
| `corpus_max_chars_not_implemented` | error | A corpus entry sets `max_chars` to a value other than the schema default; the option is reserved by spec 04 but not yet honored. |
| `corpus_min_word_length_not_implemented` | error | A corpus entry sets `min_word_length` to a value > 1; reserved by spec 04, not yet honored. |
| `text_transform_not_implemented` | error | A `text_transforms[].kind` is structurally valid but isn't yet implemented by `pd_ocr_synth.text_transforms` (e.g., a future kind from spec 05). |
| `shaping_engine_not_implemented` | error | `rendering.shaping_engine` is set to a value other than the implemented engine; reserved for future engines per spec 06. |
| `antialiasing_disable_not_implemented` | error | `rendering.antialiasing` is `false`; the renderer doesn't yet support disabling AA per spec 06. |
| `layout_key_unused` | warning | A layout key is set but doesn't apply to the active `layout.mode` (e.g., `paragraph_spacing` under `paragraphs`). It will be ignored. |
| `degradation_kind_unknown` | error | A `degradation[].kind` isn't in `KNOWN_DEGRADATION_KINDS` — typically a typo or a kind from a future spec bump. |
| `degradation_kind_not_implemented` | error | The `kind` is in the spec catalog but no runtime stage class is registered; the recipe declares a stage the renderer can't apply. |
| `degradation_stage_unknown_option` | error | A degradation stage carries an option key the registered stage class doesn't accept (typo or stale field name). |
| `paper_texture_missing_directory` | error | A `kind: paper_texture` stage doesn't set `directory:`; required by spec 07. |
| `paper_texture_directory_missing` | error | The configured `paper_texture` directory doesn't exist on disk. |
| `paper_texture_directory_not_dir` | error | The configured `paper_texture` directory path exists but isn't a directory. |
| `publish_description_file_missing` | warning | `publish.hf_dataset.description_file` is set but the file doesn't exist; publish will skip it. |
| `publish_repo_placeholder` | warning | `publish.hf_dataset.repo` looks like the scaffold placeholder (`CHANGE-ME/...` or no `/`); edit it before publishing or pass `--repo`. |

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
*downgrades* a stricter exit code.

## Logging

- Default: human-readable progress to stderr; results summary to stdout.
- `--log-format json` emits one JSON record per line (for piping into
  `jq`, log collectors, or wrapping scripts).
- Render reports rate (samples/sec), ETA, and a final manifest path.
