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
      ground truth; `run_recipe` dispatch wires
      `output.mode = detection` + `layout.mode = paragraphs` end-to-end
      through `DetectionWriter`. Pure wrap-fitter `fit_lines` lands as
      a measure-only HarfBuzz-shaped greedy first-fit packer with hard-
      break preservation and long-word fallback. Wrap-fitter is now
      wired into the paragraphs dispatch path: when
      `layout.max_width_px` is set, long single-line corpus tokens
      wrap across multiple lines using the same font + pixel size the
      renderer paints with (pre-sampled `ParagraphStyle` threaded
      through both `fit_lines` and `render_paragraph`). Alignment and
      indent still TODO.)
- [~] `pages` mode — multi-paragraph page synthesis with margins,
      configurable page size, optional headings + drop caps.
      (Foundation: `ParagraphBox` dataclass + `paragraph_boxes` field
      on `RenderedSample` lands ahead of the renderer. Default `()`,
      so `word_crops` / `lines` keep emitting legal samples; populated
      with one entry by `render_paragraph` so single-paragraph and
      multi-paragraph samples can be consumed uniformly downstream.
      Round-trips through the parallel-worker boundary alongside
      `line_boxes` / `word_boxes`. The `paragraph_spacing` recipe
      field is now wired through the `Layout` model and the
      generated JSON schema; validation accepts it on `pages` mode
      and warns `layout_key_unused` on every other mode (it has no
      meaning between lines or for a single-paragraph sample). The
      `render_page` primitive lands as a multi-paragraph compositor
      that delegates per-paragraph rendering to `render_paragraph`
      via a zero-padded pre-sampled `ParagraphStyle` — single-font
      invariant lifted from paragraph to page level, with the
      paragraph_spacing multiplier sampled once per page. Outputs
      flatten into per-paragraph + per-line + per-word + per-cluster
      boxes in reading order. The `run_recipe` pages-mode dispatch
      now wires through `DetectionWriter`: tokenizer splits on a
      triple-blank-line page boundary, the dispatch re-splits each
      page token on the regular paragraph boundary, fits each inner
      paragraph through `fit_lines` against the pre-sampled
      `PageStyle`, and `render_page` composes the multi-paragraph
      canvas. Determinism + serial/parallel parity hold (the
      worker payload already round-trips `paragraph_boxes`).
      Alignment / indent / headings / drop caps / explicit
      `page_size_px` are still TODO.)

### Bbox-aware degradation

- [ ] Geometric stages update bboxes correctly: `skew`, `perspective`,
      `scale`.
- [x] Pixel-only stages pass bboxes through unchanged.
- [x] Tests verify bbox round-trip on a fixed seed.

### Detection writer

- [x] Emit local `pd-ocr-trainer/v1` detection layout:
      `images/page_*.png`, `labels.json`, `manifest.jsonl`,
      `recipe.snapshot.yaml`, `stats.json`.
      (`DetectionWriter` lands with full force/resume semantics +
      bbox→polygon expansion; `run_recipe` end-to-end dispatch wired
      through `paragraphs` layout. Filename matches the trainer's
      `labels.json` reader, not the spec's earlier `pages.json` draft —
      see spec 08 §Detection mode layout.)
- [ ] `labels.json` schema confirmed against
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
