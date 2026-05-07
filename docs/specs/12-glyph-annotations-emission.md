# 12 — Glyph-level annotation emission

This spec defines how the synth pipeline emits **per-word glyph
annotations** as a side channel alongside ground-truth text. It is the
producer side of a workspace-wide data model whose canonical definition
lives in `pd-book-tools` (`GlyphAnnotations`, `LigatureMark`). Synth has
*perfect* knowledge of which glyphs it placed for each word, so it is
the gold-standard source for any classifier learning to detect
ligatures, long-s, or swash forms from word crops.

Status: spec only. Implementation lands in M12 (see
[`../roadmap/12-glyph-annotations.md`](../roadmap/12-glyph-annotations.md)).

## Hard invariant: GT text stays semantic

Ground-truth text (the `text` field in recognition `labels.json`,
detection per-word records, manifest entries, and any new annotation
file) **must never** contain Unicode presentation-form ligatures
(U+FB00–U+FB06: `ﬀ ﬁ ﬂ ﬃ ﬄ ﬅ ﬆ`). Long-s presentation (`ſ`, U+017F) is
likewise *not* the GT form: when a recipe rolls long-s for a particular
`s`, the GT text retains `s` and the annotation records the position.

This invariant flips the relationship between presentation and GT:
typographic events live in the side channel, not in the codepoint
string. Validators (M02 lint, M07 writer, M09 detection writer) must
reject any sample whose `text` carries U+FB00–U+FB06 or where U+017F
appears without a corresponding `long_s_positions` entry.

(Spec 00 §"Open questions" originally said "we record the codepoint
string as ground truth and let the trainer's vocab handle the rest."
M12 supersedes that for ligatures and long-s specifically: the
codepoint string is now the **semantic** form, and the typographic
events move to `glyph_annotations`. Spec 00 is updated when M12 lands.)

## Where annotations are produced

Annotations are populated **during glyph placement** in the renderer
(spec 06 §"Ground truth captured per sample"), not after. The HarfBuzz
shaper already emits `glyph_runs` with cluster indices that map back
into the input codepoint string; that mapping is exactly the data
needed to compute `LigatureMark.char_span`. Adding annotations here
costs one extra dict per word and avoids any post-hoc OCR-the-output
inference.

Pipeline position:

```
corpus → transforms → tokenize → SHAPE+RENDER (← annotations attached here)
       → degrade → write
```

The `long_s_medial` text transform (spec 05) is the one transform
whose effect must be captured **before** rendering: when it rewrites
`s → ſ` for shaping/rasterization, it must record the positions in a
per-token sidecar that the renderer copies into
`glyph_annotations.long_s_positions`. The renderer never re-derives
long-s positions from glyph IDs; the transform owns that knowledge.

The renderer derives ligatures **from** the shaping output: a
HarfBuzz cluster whose `glyph_id` resolves (via the font's GSUB
table) to a single ligature glyph that consumed multiple input
codepoints emits one `LigatureMark`.

Swash detection is recipe-declared: a font config flagged
`swash: true` (or feature `swsh` enabled per-sample) marks every word
rendered with that font as `swash: true`. v1 does not attempt
per-glyph swash discrimination.

## Recipe-level config

Each recipe declares which annotation features it uses. Features the
recipe does not declare are emitted as empty lists (or `false` for
`swash`), never omitted, so downstream consumers see a stable schema.

```yaml
annotations:
  enabled: true                 # default: true. set false to skip the side channel entirely.
  features:
    long_s: true                # populate long_s_positions
    ligatures:                  # populate ligatures[]; list = which kinds to record
      - CT
      - ST
      - FI
      - FL
      - FFI
      - FFL
      - LONG_S_T               # ſt (treated as ligature, distinct from plain long_s)
      - GAELIC_*               # see Cló Gaelach recipe below
    swash: false               # set true if any font in the pool is swash-enabled
```

Unknown ligature kinds are an error at recipe validate time. The
canonical kind enum lives in `pd-book-tools` so all repos agree on
spelling; synth imports it.

### Per-recipe declarations (initial set)

| Recipe | `long_s` | `ligatures` | `swash` |
|--------|----------|-------------|---------|
| `gaelic` (Cló Gaelach, target #1) | true | `LONG_S_T`, plus Gaelic-specific kinds documented in pd-book-tools as `GAELIC_*` once enumerated by font/feature inspection | false |
| `roman` (generic Roman, ct/st coverage) | true | `CT`, `ST`, `FI`, `FL`, `FFI`, `FFL` | false |
| `fraktur` (future) | true | `CH`, `CK`, `LONG_S_T`, `LONG_S_S` (ſs), plus Fraktur-specific | false |
| `early-modern-english` (future) | true | `CT`, `ST`, `FI`, `FL`, `FFI`, `FFL`, `LONG_S_T` | true (italic catchwords) |

The `gaelic` and `roman` recipes are the only two committed in M12;
the others are sketches for later milestones.

## Computing `char_span`

`char_span` is `[start, end)` in **char indices into the GT text of
that word** — the post-transform, semantic GT, *not* the shaping input
that the renderer fed to HarfBuzz.

For ligatures, the steps:

1. HarfBuzz emits a cluster `c` with one glyph that consumed input
   codepoints `[i, j)` of the **shaping input string**.
2. The transform layer maintains a `shaping_input → gt_text` index
   map (identity except where a transform like `long_s_medial`
   rewrites a single GT char to a single shaping char — still 1:1 by
   index for v1).
3. `char_span = [map[i], map[j])` in GT-text indices.

For multi-char ligatures spanning codepoints (e.g. `CT` consuming
`c`+`t`, `FFI` consuming `f`+`f`+`i`), `end - start` equals the input
codepoint count, and the GT text contains those literal letters.

`char_span` is `None` *only* when the renderer cannot trace the
ligature back to a unique input span — a defensive escape hatch for
future fonts with GSUB rules synth can't model. v1 must never emit
`char_span: None` for the recipes above; CI fails if it does.

For `long_s_positions`: each entry is the GT-text index of an `s`
that was rendered as `ſ`. The list is sorted ascending. Duplicates
are an error.

## Output schema

Annotations attach to the per-word output JSON as a sibling of the
existing word fields. They are **additive** — no existing field
changes shape — so older trainer code that reads `text` keeps working.

### Recognition mode (`labels.json`)

Spec 08's recognition `labels.json` is currently a flat
`{filename: text}` map. M12 adds a sibling file
`glyph_annotations.json`:

```json
{
  "0000000.png": {
    "ligatures": [
      { "kind": "CT", "char_span": [3, 5] }
    ],
    "long_s_positions": [1],
    "swash": false
  },
  "0000001.png": {
    "ligatures": [],
    "long_s_positions": [],
    "swash": false
  }
}
```

`labels.json` is unchanged so trainer code that does not opt in keeps
working. Trainer/labeler consumers that want the side channel read
both files keyed by filename. (We do not extend `labels.json`'s value
shape because `pd-ocr-trainer`'s `RecognitionDataset` parses it
directly; introducing nested values would force a coordinated trainer
release.)

`manifest.jsonl` gains an additive `glyph_annotations` field on each
record, mirroring the same per-word object — useful for downstream
tooling that consumes the manifest stream rather than `labels.json`.

### Detection mode (`labels.json`)

Spec 08's detection `labels.json` already has a per-word object
inside `lines[].words[]`. M12 extends each word object with a
`glyph_annotations` field of the same shape:

```json
{
  "bbox": [120, 205, 220, 235],
  "text": "Cuiḋ",
  "glyph_annotations": {
    "ligatures": [],
    "long_s_positions": [],
    "swash": false
  }
}
```

Doctr ignores unknown fields, so this is free.

## Edge cases

- **Word ends with hyphen, ligature crosses a linebreak.**
  Out of scope for v1. Word_crops layout never sees a soft-hyphen
  break (one word per crop). Lines/paragraphs/pages layouts may
  produce a hyphenated break inside a `ct`/`st` ligature; in that
  case the ligature is shaped within a single line and the GT word on
  the next line carries no ligature. M12 does not emit cross-line
  ligature annotations — flagged here so M09's detection writer
  doesn't attempt one. A future milestone can add a `linebreak_split`
  field if a real recipe needs it.
- **Combining-mark words (Cló Gaelach dotted consonants stored as
  base + U+0307).** Not a ligature event; not emitted in
  `ligatures`. A future `combining_marks` annotation field is a
  natural extension, but is out of scope for M12.
- **Skipped samples (`MissingGlyphError`).** No annotation entry
  written; the sample is absent from `labels.json` already.
- **`annotations.enabled: false`.** Writer omits
  `glyph_annotations.json` entirely and writes
  `glyph_annotations: null` (not `{}`) in detection word objects and
  manifest records, so consumers can distinguish "feature off" from
  "feature on but empty."

## Determinism

Annotations are a pure function of (recipe, seed, sample index). The
`recipe.snapshot.yaml` digest covers `annotations.*`; changing
declared features invalidates `--resume` exactly as changing fonts
does today (spec 08 §"Idempotency and resumption").

## Cross-references

- Data model (consumer-facing): see the corresponding
  `GlyphAnnotations` / `LigatureMark` spec in `pd-book-tools`.
- Spec 06 §"Ground truth captured per sample": baseline of what the
  renderer emits; M12 extends `glyph_runs` with the annotation derive
  step.
- Spec 08 §"Recognition mode" / §"Detection mode": output layouts
  this spec extends additively.
- Spec 05 §"long_s_medial" / §"ct_st_ligature_marker": text
  transforms whose effects feed the annotations.
