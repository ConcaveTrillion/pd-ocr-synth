# 02 — Recipe format

This is the reference for the YAML schema. For a tutorial walkthrough, see
[03 — Tutorial](03-tutorial-writing-a-recipe.md).

## Top-level structure

```yaml
schema_version: 1            # required; bump when breaking changes land
name: gaelic                 # required; recipe identifier (matches output profile by default)
description: |               # optional; free-form
  Cló Gaelach + pre-reform Irish orthography
seed: 42                     # optional; default 0
output:                      # required
  ...
corpus:                      # required; one or more providers
  - ...
text_transforms:             # optional; ordered list
  - ...
fonts:                       # required; one or more font specs
  - ...
rendering:                   # required
  ...
layout:                      # required
  ...
degradation_presets:         # optional; named groups expanded into `degradation`
  ...
degradation:                 # optional; ordered pipeline
  - ...
publish:                     # optional; defaults for `pd-ocr-synth publish`
  ...
```

## Path resolution

Paths in a recipe are resolved in this order:

1. Absolute paths used as-is.
2. `~` and `${ENV_VAR}` are expanded.
3. Relative paths are resolved against **the directory containing the
   recipe file** — not CWD.

This makes recipes portable: a recipe folder with its own fonts and textures
can be moved without breaking.

## Variable types — scalars, ranges, choices

The schema supports three forms wherever a "varying" value makes sense
(font size, blur sigma, etc.):

```yaml
# Fixed scalar
font_size_pt: 14

# Uniform range
font_size_pt: { min: 10, max: 20 }

# Discrete weighted choice
font_size_pt:
  - { value: 12, weight: 0.6 }
  - { value: 14, weight: 0.3 }
  - { value: 18, weight: 0.1 }
```

Validators reject mixing forms inside a single key. Integer vs float is
inferred from the `min`/`max` types.

## `output`

```yaml
output:
  format: pd-ocr-trainer/v1     # required; see spec 08
  mode: recognition             # required; recognition | detection
  destination: ${PD_ML_MODELS}/ml-recognition/gaelic/recognition
  count: 50000                  # required; total samples
  manifest: manifest.jsonl      # optional; default per spec 08
```

`destination` is created if missing. If non-empty and not the same recipe's
prior output, render aborts unless `--force` is passed.

## `corpus`

A list. Order matters only for deterministic seeding. Each entry has a
`type` that selects a provider:

```yaml
corpus:
  - type: web
    url: https://example.com/text.html
    parser: html-text
    cache: true
  - type: local
    path: ./corpora/seed-words.txt
  - type: hf_dataset
    name: example/dataset
    split: train
    field: text
```

See [04 — Corpus providers](04-corpus-providers.md) for the full provider
catalog.

After all providers run, their outputs are concatenated and tokenized into
the unit demanded by the chosen layout (words, lines, paragraphs).

## `text_transforms`

An ordered list of named transforms applied to corpus text **before**
tokenization. Each is either a bare name (uses defaults) or a name with
options:

```yaml
text_transforms:
  - normalize_whitespace
  - tironian_et:
      replace_words: ["agus", "and"]
  - apply_lenition_dots
  - long_s_medial:
      probability: 0.85
```

See [05 — Text transforms](05-text-transforms.md) for built-ins.

## `fonts`

```yaml
fonts:
  - path: ./fonts/bungc/bungc.otf
    weight: 0.4
    license: gaelchlo-celtic-free    # informational; surfaced in manifest
    source: https://www.gaelchlo.com/bungc.zip
  - path: ./fonts/seangc/seangc.otf
    weight: 0.4
    license: gaelchlo-celtic-free
    source: https://www.gaelchlo.com/seangc.zip
  - path: ./fonts/Gaedhilge.otf
    weight: 0.2
    license: OFL-1.1
    optional: true                   # skipped if missing rather than failing
```

Weights are normalized. A missing weight defaults to `1.0`. Validation
checks that each font file exists and exposes the codepoints needed by the
post-transform corpus (a sample report is produced by `validate`).

## `rendering`

```yaml
rendering:
  shaping_engine: harfbuzz       # harfbuzz | pillow (pillow is a fallback)
  font_size_pt: { min: 10, max: 20 }
  dpi: { min: 200, max: 400 }
  ink_color:
    r: { min: 5, max: 60 }
    g: { min: 5, max: 60 }
    b: { min: 5, max: 60 }
  background_color:
    r: { min: 220, max: 250 }
    g: { min: 215, max: 245 }
    b: { min: 200, max: 235 }
  antialiasing: true
```

See [06 — Rendering](06-rendering.md).

## `layout`

```yaml
layout:
  mode: word_crops               # word_crops | lines | paragraphs | pages
  padding_px: { min: 4, max: 12 }
  max_width_px: 800              # only for lines/paragraphs
  line_spacing: { min: 1.1, max: 1.4 }
```

Different `mode` values gate different keys. Validators surface unused
keys as warnings.

## `degradation`

An ordered pipeline. Each entry has a stage `kind`, a `probability`, and
stage-specific options:

```yaml
degradation:
  - kind: blur
    probability: 0.6
    sigma: { min: 0.0, max: 1.5 }
  - kind: paper_texture
    probability: 0.5
    directory: ./textures/aged-paper/
    blend: multiply
    opacity: { min: 0.2, max: 0.6 }
  - kind: jpeg
    probability: 0.4
    quality: { min: 60, max: 95 }
```

See [07 — Degradation](07-degradation.md).

## `publish`

Optional defaults consumed by `pd-ocr-synth publish` (see
[10 — Publishing](10-publishing.md)).

```yaml
publish:
  hf_dataset:
    repo: ntw8532/pd-ocr-synth-gaelic   # required when block present
    private: false
    license: cc-by-4.0
    tags: [ocr, gaelic, irish, pd-ocr, synthetic]
    language: [ga]
    description_file: ./gaelic/README.md.template   # optional
```

If absent, `pd-ocr-synth publish <recipe>` requires `--repo` on the
command line. CLI flags always override recipe values.

## Validation rules (summary)

`pd-ocr-synth validate <recipe>` enforces:

1. Required top-level keys are present.
2. `schema_version` matches a supported version.
3. All paths resolve and exist.
4. Each font opens and reports the glyphs it covers.
5. Corpus providers can be reached or have a cached copy when `--offline`.
6. Layout mode keys are consistent.
7. Degradation `kind` values are known.
8. Output destination is writable.

Any failure exits with code 3 and a structured error report.
