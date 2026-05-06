# M09 — Output: detection mode

**Goal:** layouts beyond word-crops, bbox-aware degradation, and a
detection-mode writer that emits `pd-ocr-trainer/v1` detection layout
plus parquet for HF.

Spec: [`06-rendering.md`](../specs/06-rendering.md) +
[`08-output-format.md`](../specs/08-output-format.md).

## Deliverables

### Layouts

- [x] `lines` mode — N words on a single baseline; line-level GT.
- [~] `paragraphs` mode — wrapped block; per-line + per-word bboxes.
      (`render_paragraph` primitive lands stacking + per-line/word
      ground truth; wrap-fitter, alignment, indent still TODO.)
- [ ] `pages` mode — multi-paragraph page synthesis with margins,
      configurable page size, optional headings + drop caps.

### Bbox-aware degradation

- [ ] Geometric stages update bboxes correctly: `skew`, `perspective`,
      `scale`.
- [x] Pixel-only stages pass bboxes through unchanged.
- [x] Tests verify bbox round-trip on a fixed seed.

### Detection writer

- [ ] Emit local `pd-ocr-trainer/v1` detection layout:
      `images/page_*.png`, `pages.json`, `manifest.jsonl`,
      `recipe.snapshot.yaml`, `stats.json`.
- [ ] `pages.json` schema confirmed against
      `pd-ocr-trainer/dataset_store.py` (cross-project integration test).

### HF detection publish

- [ ] Publish path uses `datasets.Dataset.from_generator(...)
      .push_to_hub(...)` for parquet sharding (~500 MB target shards).
- [ ] Schema matches spec 10's detection schema (image, size, lines,
      words, font, degradations).

### Tests

- [ ] Render 5 single-column pages; verify per-line and per-word
      bboxes intersect their reported text content.
- [ ] Trainer-side integration: `pd-ocr-trainer` reads the detection
      profile and reports expected page count + line count.
- [ ] HF parquet round-trip: load the published parquet, verify a
      sample image and its boxes decode.

## Validation criteria

```bash
# Edit recipe
sed -i 's/mode: word_crops/mode: pages/' recipes/gaelic.yaml

pd-ocr-synth render gaelic -c 200
# → 200 page PNGs + pages.json with line/word bboxes

pd-ocr-synth publish gaelic
# → parquet shards on HF; viewer renders pages with overlay
```

## Out of scope

- Multi-column page synthesis (stretch; could move to M10).
- Curved-baseline / strongly-warped historical layouts.
- Marginalia, footnotes, running headers — too many degrees of
  freedom for v1.

## Risks / open items

- **Layout realism.** Real 19th-c. printing has irregularities
  (uneven margins, page-by-page drift) that simple synthesis won't
  capture. Acceptable for synthetic *pretraining* data; final domain
  transfer relies on real labeled data via `pd-ocr-labeler`.
- **Bbox accuracy under ink_bleed.** Dilation can push glyphs beyond
  the original bbox. Decide: clip, expand, or accept slight
  over/under-coverage. Match what `pd-ocr-trainer` expects.
- **Parquet image embedding.** Encoding PNGs into parquet bytes is
  efficient but makes preview/debugging harder than imagefolder. The
  HF Dataset Viewer renders both fine; pick parquet for shard count
  reasons alone.
