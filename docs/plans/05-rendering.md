# M05 — Rendering

**Goal:** transformed text turns into a clean image plus per-glyph
ground truth. Word-crop layout works end-to-end; line/paragraph/page
layouts may slip to M09.

Spec: [`06-rendering.md`](../specs/06-rendering.md).

## Deliverables

### Shaping & rasterization

- [x] `pd_ocr_synth.render.word_crop` (HarfBuzz path):
  - Uses `uharfbuzz` for shaping.
  - Uses `freetype-py` for glyph rasterization.
  - Per-font feature toggles (`liga`, `calt` enabled by default).
- [ ] Pillow-only fallback engine:
  - Pillow's `ImageDraw.text` for plain Latin without shaping.
  - Deferred — Cló Gaelach needs shaping, so the spec's "auto-skip
    when shaping required" path covers M05. Revisit when a recipe
    actually targets a non-shaping script.

### Tokenization (the M03 placeholder gets real here)

- [x] Word tokenizer for `word_crops` mode: whitespace + edge-
      punctuation split, drop empties (`pd_ocr_synth.tokenization`).
- [ ] Per-recipe `corpus_sampling`: `uniform`, `unique_weighted`,
      `frequency`. Preview currently uses uniform-with-replacement;
      the weighting modes ship with the M07 dataset loop.

### Sample assembly (word_crops)

- [x] Per-sample draws: font (weighted), font_size_pt, dpi, ink_color,
      background_color, padding.
- [x] Render to a tight bbox + padding sampled per-recipe.
- [x] Capture per-sample ground truth:
  - `text` (codepoint string)
  - `bbox` (tight inked region, post-padding)
  - `font_path`, `font_size_pt`, `dpi`
  - `glyph_runs` (per-cluster bbox + cluster index)

### Font validation

- [x] On recipe load, open each font and surface `font_missing` /
      `font_unreadable` / `font_empty` (plus `optional_font_missing`
      warning) at `pd-ocr-synth validate`. `pd_ocr_synth.fonts.open_font`
      computes codepoint coverage; `_check_fonts` consumes only the
      open/empty signals.
- [x] At render time, samples requiring a missing glyph raise
      `MissingGlyphError`; the preview loop records `missing_glyph` as
      the manifest skip reason with the missing codepoints listed.
- [ ] Validate-time corpus-vs-font coverage report: walk the
      transformed corpus, intersect with each font's `FontInfo.codepoints`,
      emit a `font_corpus_coverage_*` warning when a font in the weighted
      pool can't render some codepoint set. Deferred — render-time
      `MissingGlyphError` already prevents silent corruption; this is a
      pure-UX upgrade so authors discover gaps before paying corpus
      fetch + render cost. Spec 06 §"Font selection" tracks this.
- [ ] Validate-time `liga`/`calt` GSUB feature presence inspection:
      open the font's GSUB table and report whether each requested
      `features` toggle has a corresponding feature record. Deferred —
      requires a font-table parser (`fonttools` or hand-rolled);
      `freetype-py` doesn't expose GSUB. Until it lands, requesting
      `features.liga: true` on a font without a `liga` lookup is a
      no-op rather than a crash, which is acceptable.

### Determinism

- [x] All randomness flows from the recipe's `seed` through a single
      `Random` per render run, branched per sample by sample index
      (`RenderContext.reseed_for_sample` + `branched_seed`). Same
      recipe + seed + sample index → identical output bytes.

### CLI surface

- [x] `pd-ocr-synth preview <recipe>` — renders N samples (default 50)
      to a configurable output dir. Honors `--count` / `--output` /
      `--seed`. Writes `images/`, `manifest.jsonl`, `stats.json`. No
      degradation yet (M06); no `pd-ocr-trainer/v1` adapter (M07).

### Tests

- [x] Smoke: render 5 samples from a tmp recipe pointed at the bundled
      Bunchló GC font; assert non-empty PNGs (`tests/test_render.py`).
- [x] Determinism: same seed + sample index → byte-identical PNG bytes.
- [x] Glyph-coverage skip: rendering a token containing U+1F600 (an
      emoji absent from the Gaelic font) raises `MissingGlyphError`
      with the missing codepoints set; preview records the same as a
      manifest skip reason.
- [ ] Visual eyeball test: `make gaelic-preview` produces 50 samples a
      human can spot-check. Deferred — depends on `make fetch-fonts`
      which is interactive (license confirmation).

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
