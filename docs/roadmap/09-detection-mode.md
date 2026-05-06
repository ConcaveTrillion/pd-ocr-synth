# M09 — Output: detection mode

**Goal:** layouts beyond word-crops, bbox-aware degradation, and a
detection-mode writer that emits `pd-ocr-trainer/v1` detection layout
plus parquet for HF.

Spec: [`06-rendering.md`](../specs/06-rendering.md) +
[`08-output-format.md`](../specs/08-output-format.md).

## Deliverables

### Layouts

- [x] `lines` mode — N words on a single baseline; line-level GT.
      (`render_line` primitive lands per-word bboxes (`253684d`); the
      `run_recipe` dispatch wires `output.mode = recognition` +
      `layout.mode = lines` end-to-end through `RecognitionWriter`
      (`97495d5`). Recipe-level `output.mode = detection` +
      `layout.mode = lines` is rejected at validation time
      (`9d6c2e0`) — spec 08 § Modes pairs `lines` only with
      recognition.)
- [x] `paragraphs` mode — wrapped block; per-line + per-word bboxes.
      (`render_paragraph` primitive lands stacking + per-line/word
      ground truth (`200aa92`); `run_recipe` dispatch wires
      `output.mode = detection` + `layout.mode = paragraphs` end-to-end
      through `DetectionWriter` (`d0131d4`). Pure wrap-fitter
      `fit_lines` lands as a measure-only HarfBuzz-shaped greedy
      first-fit packer with hard-break preservation and long-word
      fallback (`83a2e47`), wired into the paragraphs dispatch path
      (`5ac5170`): when `layout.max_width_px` is set, long single-line
      corpus tokens wrap across multiple lines using the same font +
      pixel size the renderer paints with (pre-sampled
      `ParagraphStyle` threaded through both `fit_lines` and
      `render_paragraph`). Alignment / first-line indent are not
      implemented — see "Residual M09 work → Paragraph alignment" and
      "→ First-line indent" below.)
- [x] `pages` mode — multi-paragraph page synthesis with margins,
      configurable page size, optional headings + drop caps.
      (Foundation: `ParagraphBox` dataclass + `paragraph_boxes` field
      on `RenderedSample` lands ahead of the renderer (`3e51f57`).
      Default `()`, so `word_crops` / `lines` keep emitting legal
      samples; populated with one entry by `render_paragraph` so
      single-paragraph and multi-paragraph samples can be consumed
      uniformly downstream. Round-trips through the parallel-worker
      boundary alongside `line_boxes` / `word_boxes`. The
      `paragraph_spacing` recipe field is wired through the `Layout`
      model and the generated JSON schema (`039dbf1`); validation
      accepts it on `pages` mode and warns `layout_key_unused` on
      every other mode. The `render_page` primitive lands as a
      multi-paragraph compositor (`861ae8b`) that delegates
      per-paragraph rendering to `render_paragraph` via a zero-padded
      pre-sampled `ParagraphStyle` — single-font invariant lifted from
      paragraph to page level, with the `paragraph_spacing` multiplier
      sampled once per page. Outputs flatten into per-paragraph +
      per-line + per-word + per-cluster boxes in reading order. The
      `run_recipe` pages-mode dispatch wires through
      `DetectionWriter` (`e94ca2a`): tokenizer splits on a
      triple-blank-line page boundary, the dispatch re-splits each
      page token on the regular paragraph boundary, fits each inner
      paragraph through `fit_lines` against the pre-sampled
      `PageStyle`, and `render_page` composes the multi-paragraph
      canvas. Determinism + serial/parallel parity hold (the worker
      payload already round-trips `paragraph_boxes`). Alignment /
      indent / headings / drop caps / explicit `page_size_px` are not
      implemented — see "Residual M09 work" below.)

### Bbox-aware degradation

- [~] Geometric stages update bboxes correctly: `skew`, `perspective`,
      `scale`. (`skew` is the only geometric stage registered today
      (`builtins.py:542`); it rotates `sample.bbox`,
      `sample.glyph_runs`, `word_boxes`, `line_boxes`, and
      `paragraph_boxes` corner-by-corner around the expanded canvas.
      Round-trip locked in
      `tests/test_degradation.py::test_skew_updates_bbox_and_keeps_text_inside`
      / `::test_skew_glyph_runs_track_image_resize`; the multi-
      collection contract is locked in
      `::test_skew_propagates_to_every_box_collection` (the M09
      residual). `perspective` and `scale` are not yet registered;
      they should land with full bbox propagation built in from the
      start.)
- [x] Pixel-only stages pass bboxes through unchanged. (`6165e58` locks
      the invariant for every registered pixel stage —
      `tests/test_degradation.py::test_pixel_stages_preserve_bbox_glyph_runs_and_word_boxes`
      runs each pixel kind through the pipeline on a real
      `RenderedSample` and asserts `bbox`, `glyph_runs`, `word_boxes`,
      `line_boxes` and `paragraph_boxes` are unchanged.)
- [x] Tests verify bbox round-trip on a fixed seed. (Pixel side:
      `6165e58`. Geometric side: `_skew` round-trip is locked at a
      fixed angle in `tests/test_degradation.py`; broader detection-
      mode bbox propagation is the residual item below.)

### Detection writer

- [x] Emit local `pd-ocr-trainer/v1` detection layout:
      `images/page_*.png`, `labels.json`, `manifest.jsonl`,
      `recipe.snapshot.yaml`, `stats.json`.
      (`DetectionWriter` lands with full force/resume semantics +
      bbox→polygon expansion (`623da9a`); `run_recipe` end-to-end
      dispatch wired through `paragraphs` (`d0131d4`) and `pages`
      (`e94ca2a`) layouts. Filename matches the trainer's
      `labels.json` reader, not the spec's earlier `pages.json` draft —
      see spec 08 § Detection mode layout.)
- [ ] `labels.json` schema confirmed against
      `pd-ocr-trainer/dataset_store.py` (cross-project integration
      test). The writer emits the doctr-required `polygons` flat list
      plus our richer `lines` GT per spec 08, but no test imports
      `pd-ocr-trainer` to confirm the field names land where
      `RecognitionDataset` / `DetectionDataset` actually look.

### HF detection publish

- [ ] Publish path uses `datasets.Dataset.from_generator(...)
      .push_to_hub(...)` for parquet sharding (~500 MB target shards).
- [ ] Schema matches spec 10's detection schema (image, size, lines,
      words, font, degradations).

### Tests

- [~] Render 5 single-column pages; verify per-line and per-word
      bboxes intersect their reported text content.
      (`tests/test_cli_render_paragraphs.py` covers the paragraphs end
      of this — a deterministic recipe renders a multi-paragraph
      sample and asserts `labels.json` carries `lines` + `polygons`
      whose bboxes contain ink. The pages-mode equivalent is locked in
      `tests/test_cli_render_pages.py`. A standalone "5-page" smoke
      driver in the README-shaped form below has not been added; the
      existing tests cover the same invariants on smaller fixtures.)
- [ ] Trainer-side integration: `pd-ocr-trainer` reads the detection
      profile and reports expected page count + line count.
- [ ] HF parquet round-trip: load the published parquet, verify a
      sample image and its boxes decode.

## Residual M09 work

The renderer + writer spine is in place: all four layout modes run
end-to-end, validation pairs them correctly to `output.mode`, and the
detection writer emits `labels.json` that matches the trainer's
filename contract. The following gaps remain — pick any of them as a
future small chunk.

### Paragraph alignment

- [ ] Spec 06 § `paragraphs` advertises
      `alignment: justify | left | right | center` and spec 06
      § Ground truth captured per sample documents `lines` /
      `words` GT regardless of alignment. Today `render_paragraph`
      lays each shaped line at the left edge of the per-paragraph
      canvas (no `alignment` field on `Layout` / `ParagraphStyle`),
      which is `left` by default. Add a layout key
      `paragraph_alignment: "left" | "center" | "right" | "justify"`,
      default `"left"` (preserves current output bytes). Implement
      `"left"` + `"center"` first as a single chunk; `"right"` is
      mostly symmetric with `"center"`; `"justify"` is a separate
      chunk because it needs inter-word stretching (and the spec is
      silent on whether the last line of a paragraph stretches).

### First-line indent for paragraphs

- [ ] Spec 06 § `paragraphs` advertises `paragraph_indent_em`. Add a
      layout key `paragraph_indent_px: int | None = None` (resolved
      from `_em` via the sampled font size if we keep the spec's
      em-based name, otherwise straight px). Affects only
      `render_page` (paragraphs mode is single-paragraph by
      construction). Renderer prepends `indent_px` of horizontal
      whitespace to the first line of each paragraph and offsets the
      line's bbox accordingly.

### Explicit `page_size_px`

- [ ] Spec 06 § `pages` advertises
      `page_size_px: [1200, 1800]` as a fixed canvas. Today
      `render_page` auto-sizes the canvas from the laid-out content
      width/height (max paragraph width + summed paragraph heights +
      margins). Decide between (a) hard-clip to `page_size_px`, (b)
      pad-to-`page_size_px` with the sampled background colour, or
      (c) scale-to-fit. Option (b) is the cheapest and matches the
      "training data uniformity" intent — pages mode only needs a
      consistent canvas for the detection model to learn aspect
      ratios. Layout key shape: `page_size_px: tuple[int, int] |
      None = None`.

### Headings and drop caps

- [ ] Spec 06 § `pages` advertises `heading_probability` +
      `drop_cap_probability`. Headings need per-paragraph font-size
      / weight variation (likely a `paragraph_role:
      "body" | "heading"` enum on `ParagraphStyle` plus a recipe
      knob). Drop caps need glyph-level sub-rendering (first
      character oversized, multi-line wrap-around). Both are
      independent additions on top of the existing single-style
      renderer. Lower priority for v1 — the trainer can learn body
      text first; headings/drop-caps are domain transfer territory.

### Geometric-stage detection-bbox propagation

- [x] `_skew` now rotates `word_boxes`, `line_boxes`, and
      `paragraph_boxes` alongside `sample.bbox` and `glyph_runs` via
      the same per-box corner-rotation helper, so detection mode emits
      polygons that line up with the rendered text even when a `skew`
      stage is enabled. Lock test:
      `tests/test_degradation.py::test_skew_propagates_to_every_box_collection`
      (asserts survival, mutation, parent/child containment within a
      6-px slack to absorb axis-aligned-bounding-of-rotated-quad
      rounding). Empty-collection invariance is locked in
      `::test_skew_preserves_empty_optional_box_collections` so
      word-crops samples don't grow phantom annotations under skew.
      `perspective` and `scale` (currently unregistered) should land
      with bbox propagation built in from the start.

### `pd-ocr-trainer` cross-project integration test

- [ ] Cross-import the trainer's `RecognitionDataset` /
      `DetectionDataset` and run a synthetic `DetectionWriter`
      output through it. Belongs in `tests/integration/` with the
      same opt-in convention M08 established for live HF tests
      (`PD_OCR_SYNTH_TRAINER_INTEGRATION=1` env-var gate). Without
      this, `labels.json` schema drift between the two repos is only
      caught at CLI runtime.

### HF detection publish path

- [ ] Spec 10 calls for parquet-sharded detection publish via
      `datasets.Dataset.from_generator(...).push_to_hub(...)`. Today
      the `publish` CLI path only handles recognition imagefolder
      (M08). Detection adds: a per-shard generator that reads
      `images/page_*.png` + `labels.json`, encodes images into bytes
      for parquet, threads provenance columns through, and a
      `--shard-size 500MB` knob. Likely a new
      `pd_ocr_synth.publish.detection` module mirroring
      `publish.recognition`'s shape. Schema must match spec 10's
      detection schema exactly (image, size, lines, words, font,
      degradations).

## Validation criteria

```bash
# Edit recipe
sed -i 's/mode: word_crops/mode: pages/' recipes/gaelic.yaml

pd-ocr-synth render gaelic -c 200
# → 200 page PNGs + labels.json with line/word bboxes

pd-ocr-synth publish gaelic
# → parquet shards on HF; viewer renders pages with overlay
```

The first command works today; the second one is the residual HF
detection publish path above.

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
