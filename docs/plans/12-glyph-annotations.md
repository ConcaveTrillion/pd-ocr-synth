# M12 — Glyph-level annotation emission

**Goal:** synth emits a per-word `GlyphAnnotations` side channel
alongside semantic GT text, capturing ligatures, long-s positions, and
swash usage as gold-standard training signal for ligature/long-s
classifiers downstream.

Spec: [`../specs/12-glyph-annotations-emission.md`](../specs/12-glyph-annotations-emission.md).

Depends on: M07 (recognition writer), M09 (detection writer). The
shared data model `GlyphAnnotations` / `LigatureMark` lives in
`pd-book-tools` — coordinate that change first; synth imports the
enums.

## Deliverables

### Data-model dependency

- [ ] Confirm `GlyphAnnotations` / `LigatureMark` / ligature-kind
      enum landed in `pd-book-tools` and pin its version in
      `pyproject.toml`.

### Recipe schema

- [ ] Add `annotations:` block (M02 schema). Fields: `enabled`,
      `features.long_s`, `features.ligatures` (list of kind strings,
      validated against the pd-book-tools enum), `features.swash`.
- [ ] Default `annotations.enabled: true` so existing recipes opt in
      automatically; document the back-compat in the M02 changelog.
- [ ] Update `recipe.schema.json` via `make schema`.

### Pipeline plumbing

- [ ] `long_s_medial` transform (spec 05) carries a per-token
      sidecar of `s → ſ` GT-index positions through to the renderer.
      No change to GT text output.
- [ ] HarfBuzz renderer derives `ligatures[]` from cluster→glyph
      mapping plus the font's GSUB substitution table.
- [ ] Renderer attaches `glyph_annotations` to each per-word ground
      truth dict (spec 06 §"Ground truth captured per sample").
- [ ] GT-text-vs-presentation invariant guard: writer rejects any
      sample whose `text` contains U+FB00–U+FB06, or U+017F without a
      matching `long_s_positions` entry. Treat as `RenderError`,
      record in `stats.json` as a new skip reason.

### Output writers

- [ ] M07 recognition writer adds `glyph_annotations.json` sibling
      file, keyed by image filename. `labels.json` shape unchanged.
- [ ] M07 manifest gains additive `glyph_annotations` field per
      record.
- [ ] M09 detection writer extends each word object inside
      `lines[].words[]` with `glyph_annotations`. `polygons` and
      `paragraphs` unchanged.

### Recipes

- [ ] Update `recipes/gaelic.yaml` to declare `annotations.features`:
      `long_s: true`, ligatures = Gaelic kinds (final list pinned
      from font GSUB inspection), `swash: false`.
- [ ] Add `recipes/roman.yaml` (generic Roman) declaring `long_s`
      + `CT, ST, FI, FL, FFI, FFL` ligatures, `swash: false`. Acts
      as the second-recipe sanity check the spec/roadmap principle
      "extract abstractions only when a second recipe makes them
      obvious" requires.

### Tests

- [ ] Determinism: same `(recipe, seed, index)` produces byte-equal
      `glyph_annotations.json`.
- [ ] Round-trip: a known fixture word rendered with a `ct`-enabled
      Roman font emits one `LigatureMark{kind=CT, char_span=[i,
      i+2]}` whose span lines up with the literal `ct` in GT text.
- [ ] Long-s round-trip: a fixture where `long_s_medial` fires emits
      `long_s_positions` matching every `s` rendered as `ſ`.
- [ ] Invariant guard: synthetic test injecting U+FB01 into a GT
      string fails the writer.
- [ ] Schema-valid recipe with `annotations.enabled: false` emits no
      side-channel file and renders identically to pre-M12 output for
      that recipe.
- [ ] Cross-repo contract: a structural test (no real
      pd-ocr-trainer import — same approach as M07's
      `test_render_labels_json_matches_trainer_recognition_contract`)
      asserts the `glyph_annotations.json` shape against the
      pd-book-tools `GlyphAnnotations` model.

### Spec cleanup

- [ ] Spec 00 §"Open questions" item 4 ("Glyph-level ground truth")
      updated to reference M12's outcome.
- [ ] Spec 06 §"Ground truth captured per sample" gains a
      `glyph_annotations` row.
- [ ] Spec 08 §"Recognition mode" / §"Detection mode" link to spec
      12 for the side channel.

## Out of scope (call out, don't do)

- Cross-linebreak ligatures (word ends with hyphen, ligature would
  cross). Spec 12 §"Edge cases" documents this; M12 does not emit
  any annotation for the broken half.
- Combining-mark side channel (e.g. Cló Gaelach base + U+0307). A
  future `combining_marks` field; not M12.
- Per-glyph swash discrimination (only font-level swash flagging in
  v1).
- Backfilling annotations onto already-rendered datasets — a
  re-render is required.

## Sizing

One to two sessions. Most of the cost is in (a) the HarfBuzz
cluster → ligature-kind mapping, which needs per-font GSUB
introspection, and (b) the recipe-schema change rippling through M02
validators + lint rules. Writer changes are small, additive, and
covered by existing test patterns from M07 / M09.
