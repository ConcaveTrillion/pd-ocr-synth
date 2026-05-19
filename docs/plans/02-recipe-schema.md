# M02 — Recipe schema + validator (complete)

**Status:** ✅ landed in commits 14bdbde…1f3bd61 on `main`.

**Goal:** recipes load, validate, and surface errors. Users can
iterate on `gaelic.yaml` even before render works.

Spec: [`02-recipe-format.md`](../specs/02-recipe-format.md).

## Closeout notes

- `pyyaml` only — `ruamel.yaml` line-accurate errors deferred per the
  risks section ("upgrade only if errors are too vague"). pydantic's
  built-in error locations are good enough for now.
- `--offline` flag exists on `validate` but is a no-op until M03 wires
  network checks.
- `pd-ocr-synth validate gaelic` exits 3 in a fresh checkout (missing
  ``$PD_ML_MODELS``, missing seed-words.txt, missing aged-paper
  textures). Setting up those inputs is part of using the recipe, not
  a M02 deficiency. Run `pd-ocr-synth describe ./recipes/gaelic.yaml`
  for a full resolved-config dump.

## Deliverables

### `pd_ocr_synth.recipe`

- [x] Pydantic v2 models for every block: `Recipe`, `OutputBlock`,
      `CorpusEntry` (discriminated union by `type`), `TextTransform`,
      `Font`, `Rendering`, `Layout`, `DegradationStage`, `PublishBlock`.
- [x] `RangeOrChoice[T]` union type for the scalar/range/weighted-choice
      pattern from spec 02.
- [x] YAML loader that:
  - Resolves `~` and `${ENV_VAR}` in path-like strings.
  - Resolves relative paths against the recipe file's directory.
  - Returns a frozen Recipe (post-validation, immutable).

### `pd_ocr_synth.validation`

- [x] `validate_recipe(recipe: Recipe) -> ValidationReport` returning:
  - Missing files (font paths, local corpus paths)
  - Unknown degradation `kind`s
  - Conflicting `layout.mode` keys
  - Output destination unwritable
  - Schema-version mismatch
- [x] Errors report file path + YAML line number when possible
      (use `ruamel.yaml` round-tripping or pydantic's location info).
- [x] `--offline` mode: skip network-touching checks.

### CLI surface

Implement these subcommands (already stubbed in M01):

- [x] `pd-ocr-synth list` — walk recipe search path, print `name → path`.
- [x] `pd-ocr-synth validate <recipe>` — run validation, exit 0/3.
- [x] `pd-ocr-synth describe <recipe>` — dump resolved config + corpus
      stats placeholder ("corpora: 3 (not fetched)" until M03).
- [x] `pd-ocr-synth init <name>` — scaffold `recipes/<name>/recipe.yaml`
      from a template using questions from the spec's "Decide what
      you're targeting" tutorial.

### JSON Schema export

- [x] `pd-ocr-synth schema` (or a Make target) emits
      `docs/specs/recipe.schema.json` from the pydantic models. This
      enables the YAML language server to give recipe authors live
      validation in editors.

### Tests

- [x] Round-trip: load `recipes/gaelic.yaml` → validate → no errors.
- [x] Each validation rule has a positive and negative test using
      tiny in-memory recipes.
- [x] Path-resolution edge cases: `~`, env var expansion, relative-to-
      recipe, absolute paths, missing files.
- [x] Range vs scalar vs weighted-choice in every applicable field.

## Validation criteria

```bash
pd-ocr-synth validate gaelic        # exit 0, prints "OK"
pd-ocr-synth validate broken-yaml   # exit 3, line-accurate error
pd-ocr-synth list                   # shows: gaelic → recipes/gaelic.yaml
pd-ocr-synth describe gaelic        # prints resolved config block
pd-ocr-synth init fraktur           # creates recipes/fraktur/recipe.yaml
```

The exported `recipe.schema.json` opens cleanly in VS Code with the
YAML extension and provides completion + inline errors on
`recipes/gaelic.yaml`.

## Out of scope

- Network fetches (M03).
- Anything that requires reading fonts or rendering (M05).
- The `python:` inline extension loader (M04).

## Risks / open items

- **Pydantic vs dataclass + manual validation.** Pydantic v2 is the
  fastest path; only switch if peer projects have a strong
  convention against it.
- **YAML library choice.** `pyyaml` for parsing, `ruamel.yaml` if line
  numbers in errors are critical. Start with `pyyaml`; upgrade only if
  errors are too vague.
