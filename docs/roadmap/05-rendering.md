# M05 — Rendering

**Goal:** transformed text turns into a clean image plus per-glyph
ground truth. Word-crop layout works end-to-end; line/paragraph/page
layouts may slip to M09.

Spec: [`06-rendering.md`](../specs/06-rendering.md).

## Deliverables

### Shaping & rasterization

- [ ] `pd_ocr_synth.rendering.harfbuzz_engine`:
  - Use `uharfbuzz` for shaping.
  - Use `freetype-py` for glyph rasterization.
  - Per-font feature toggles (`liga`, `calt` enabled by default).
- [ ] `pd_ocr_synth.rendering.pillow_engine` (fallback):
  - Pillow's `ImageDraw.text` for plain Latin without shaping.
  - Auto-skipped when shaping is required by the script.

### Tokenization (the M03 placeholder gets real here)

- [ ] Word tokenizer for `word_crops` mode: whitespace + punctuation
      split, drop empties.
- [ ] Per-recipe `corpus_sampling`: `uniform`, `unique_weighted`,
      `frequency`. Default `unique_weighted` to avoid stop-word
      overfitting.

### Sample assembly (word_crops)

- [ ] Per-sample draws: font (weighted), font_size_pt, dpi, ink_color,
      background_color.
- [ ] Render to a tight bbox + padding sampled per-recipe.
- [ ] Capture per-sample ground truth:
  - `text` (codepoint string)
  - `bbox` (tight inked region, post-padding)
  - `font_path`, `font_size_pt`, `dpi`
  - `glyph_runs` (per-cluster bbox + cluster index)

### Font validation

- [ ] On recipe load, open each font and report its codepoint coverage.
- [ ] Surface coverage gaps to `pd-ocr-synth validate` (extends M02).
- [ ] At render time, samples requiring a missing glyph are skipped
      with the reason recorded for the manifest.

### Determinism

- [ ] All randomness flows from the recipe's `seed` through a single
      `Random` per render run, branched per sample by sample index.
      Same recipe + seed + sample index → identical output bytes.

### CLI surface

- [ ] `pd-ocr-synth preview <recipe>` — render N samples (default 50)
      to a configurable output dir. No degradation yet (M06).

### Tests

- [ ] Smoke: render 5 samples from `gaelic.yaml`; count files,
      check non-empty PNGs.
- [ ] Determinism: same seed → byte-identical PNGs.
- [ ] Glyph-coverage skip: a recipe whose font lacks `ḃ` produces a
      manifest with a `missing_glyph` reason for affected samples.
- [ ] Visual eyeball test: `make gaelic-preview` produces 50 samples
      a human can spot-check.

## Validation criteria

```bash
pd-ocr-synth fetch gaelic                                     # M03
pd-ocr-synth preview gaelic --count 50 --output /tmp/preview  # M05
ls /tmp/preview/images/   # 50 .png files
cat /tmp/preview/manifest.jsonl | head -1   # first record has font, size, glyph_runs
```

The preview directory should look like recognizable Gaelic words at
varied sizes and fonts, without paper texture or degradation yet.

## Out of scope

- Degradation pipeline (M06).
- Output adapter to `pd-ocr-trainer/v1` format (M07).
- Lines / paragraphs / pages layouts (M09).
- Detection-mode bbox geometry (M09).

## Risks / open items

- **Mark stacking on dotted consonants.** Some font + codepoint combos
  render with the dot offset. Validate visually during M05; document
  problem fonts.
- **MPS / GPU rendering.** Out of scope here — CPU rendering is the
  baseline.
- **HarfBuzz on the dev container.** Confirm `uharfbuzz` wheel is
  available for the container's Python; otherwise add system `libharfbuzz`
  via the container's apt step.
