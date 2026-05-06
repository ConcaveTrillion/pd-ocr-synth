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
- [~] Trainer-side integration: `pd-ocr-trainer` reads the detection
      profile and reports expected page count + line count.
      (Opt-in `tests/integration/test_trainer_dataset_contract.py`
      drives `doctr.datasets.DetectionDataset` on synth output and
      asserts `len(ds) == pages`; same pattern for
      `RecognitionDataset`. Always-on shape contract tests in the
      same file lock the `labels.json` schema doctr asserts at
      reader-construction time. The richer "page count + line
      count" assertion via the trainer's own `train_from_config`
      driver is the next step beyond the doctr-reader-only contract
      this lands.)
- [ ] HF parquet round-trip: load the published parquet, verify a
      sample image and its boxes decode.

## Residual M09 work

The renderer + writer spine is in place: all four layout modes run
end-to-end, validation pairs them correctly to `output.mode`, and the
detection writer emits `labels.json` that matches the trainer's
filename contract. The following gaps remain — pick any of them as a
future small chunk.

### Paragraph alignment

- [x] Spec 06 § `paragraphs` advertises
      `alignment: justify | left | right | center` and spec 06
      § Ground truth captured per sample documents `lines` /
      `words` GT regardless of alignment. All four values land as a
      layout key
      `paragraph_alignment: Literal["left", "center", "right", "justify"] | None = None`
      (`None` and `"left"` both preserve the historical un-aligned
      output bit-for-bit). The validator permits the field on
      `paragraphs` and `pages` modes and warns `layout_key_unused`
      on `word_crops` / `lines`, parametrized across all four
      values (`tests/test_validation.py::test_paragraph_alignment_*`).
      `render_paragraph` applies a per-line offset to image paste,
      glyph runs, word boxes, and line bbox: under `"center"` the
      offset is `(paragraph_width - line_natural_width) // 2`; under
      `"right"` it is the full `paragraph_width - line_natural_width`,
      so short lines flush their right edge to the longest line's
      right edge. The longest line gets offset 0 under both
      `"center"` and `"right"` so canvas size is unchanged
      (`tests/test_render_paragraph.py::*alignment_center*`,
      `*alignment_right_*`). `render_page` inherits alignment via
      the recipe (no extra wiring); per-paragraph alignment happens
      against each paragraph's own max line width, not the page's
      (`tests/test_render_page.py::*alignment_center_*`,
      `*alignment_right_*`). `"justify"` distributes the per-line
      slack (`paragraph_width - line_natural_width`) across the
      inter-word gaps in each line by re-shaping eligible lines with
      a `justify_target_width` and shifting all glyphs in word `i`
      (for `i >= 1`) right by an accumulating per-gap offset. Per
      standard book-typesetting practice, the **last line** of a
      paragraph and **single-word** lines fall back to left
      alignment (justify-stretching them would either look awkward
      or require glyph-tracking, neither of which serves OCR
      training). 8 paragraph-level tests in
      `tests/test_render_paragraph.py::*alignment_justify_*` lock
      the right-edge flush, the last-line / single-word / single-
      line-paragraph fallback, the per-word accumulating offset,
      glyph-run tracking, and word-box-disjointness invariants;
      2 page-level tests in
      `tests/test_render_page.py::*alignment_justify_*` lock the
      per-paragraph independence and canvas containment.

### First-line indent for paragraphs

- [x] Spec 06 § `paragraphs` advertises `paragraph_indent_em`. Layout
      key `paragraph_indent_px: int | None = None` lands as a fixed
      px integer (deterministic, no font-size dependence — `_em` may
      come back later as a derived convenience). Validator permits it
      only on `pages` mode and warns `layout_key_unused` elsewhere
      (`tests/test_validation.py::test_paragraph_indent_px_warns_on_non_pages_modes`).
      `render_page` reads `recipe.layout.paragraph_indent_px or 0`
      and threads it as `first_line_indent_px` into every per-
      paragraph `render_paragraph` call; `render_paragraph` shifts
      line 0 right by that many pixels (image, glyph runs, word boxes,
      line bbox), grows the canvas width to accommodate, and leaves
      every other line untouched. `None` and `0` produce byte-
      identical PNGs (regression test:
      `tests/test_render_page.py::test_render_page_indent_none_is_bit_identical_to_zero`),
      so existing recipes are unaffected. Right-shift, canvas-widen,
      and bbox propagation are locked in `test_render_page.py::*indent*`
      and `test_render_paragraph.py::*first_line_indent*`.
- [x] **Wrap-budget interaction fix.** Initial chunk shrank line 0 at
      paint time but did *not* propagate the indent to the wrap-fitter,
      so a recipe with `max_width_px=800` + `paragraph_indent_px=40`
      packed line 0 against the full 800-px budget and the renderer
      then shifted the painted strip right by 40 — the inked first line
      sat at `[40, 840]`, overflowing the user's wrap budget by exactly
      the indent. `fit_lines` now takes `first_line_indent_px` and
      shrinks line 0's budget to `max_width_px - indent` (with a clamp
      at 1 to keep pathological `indent >= budget` recipes
      well-defined); subsequent lines (whether soft-wrapped or
      hard-break-separated) keep the full budget so a wider line never
      pays for a feature only line 0 uses. `_split_paragraph_into_lines`
      and `_split_page_into_paragraph_lines` thread the recipe's indent
      through. Five wrap-fitter tests in `tests/test_render_wrap.py`
      lock the new contract (validation, zero-indent bit-identical,
      first-line shrink, non-first-line full budget, hard-break first-
      chunk-only); one integration test in
      `tests/test_render_page.py::test_split_page_into_paragraph_lines_passes_indent_to_wrap_fitter`
      proves the recipe-driven dispatch actually fewer-words line 0
      under a real (font, dpi, indent) tuple.

### Explicit `page_size_px`

- [x] Spec 06 § `pages` advertises
      `page_size_px: [1200, 1800]` as a fixed canvas. Layout key
      `page_size_px: tuple[int, int] | None = None` lands as a (width,
      height) tuple, validated positive at load time. Validator
      permits it only on `pages` mode and warns `layout_key_unused`
      elsewhere
      (`tests/test_validation.py::test_page_size_px_warns_on_non_pages_modes`,
      `::test_page_size_px_accepted_on_pages_mode`,
      `::test_page_size_px_rejects_non_positive_at_load`). When set,
      `render_page` composes content at its natural extent and pastes
      it top-left into a canvas of exactly the requested dimensions,
      filling the remainder with the sampled `background_color`. Bbox
      annotations remain unshifted in the natural-content rectangle
      (zero offset == top-left placement) so per-word/per-line
      detection annotations match the inked region pixel-for-pixel
      (`tests/test_render_page.py::test_render_page_page_size_px_*`).
      Natural content larger than the requested canvas in either
      dimension raises `RenderError` — silent truncation would corrupt
      annotations the trainer consumes. `None` and "exactly fits" both
      preserve byte-identical output for the auto-sized path
      (`::test_render_page_page_size_px_none_is_bit_identical_to_unset`,
      `::test_render_page_page_size_px_exact_fit_does_not_pad`).
      Spec 06 § pages now documents the pad-or-error behaviour.

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

- [x] Cross-import the trainer's `RecognitionDataset` /
      `DetectionDataset` and run a synthetic `DetectionWriter`
      output through it. Lands in
      `tests/integration/test_trainer_dataset_contract.py` with the
      same opt-in convention M08 established for live HF tests
      (`PD_OCR_SYNTH_TRAINER_E2E=1` env-var gate; doctr must also be
      importable, treated as a skip rather than failure when missing).
      Two layers: (1) **always-on** shape contract tests that
      re-implement the exact assertions
      `doctr.datasets.RecognitionDataset.__init__` /
      `DetectionDataset.__init__` make on `labels.json` (key →
      filename existence; recognition value is `str`; detection
      value has `polygons` of shape `(N, 4, 2)` numerically castable
      to `np.float32`). These run under default `make ci` and lock
      schema drift on every commit. (2) **Opt-in** live tests that
      import `doctr.datasets.RecognitionDataset` and `DetectionDataset`
      directly, instantiate them on synth-produced output, and
      assert sample count + class names; skipped under default
      `make ci` because doctr isn't a synth runtime dep. Plus
      gating-helper sanity tests that lock the truthy set in sync
      with `test_publish_live_hf.py::_TRUTHY`. Without this, only
      caught at trainer CLI runtime.

### HF detection publish path

- [~] Imagefolder-shaped detection staging lands in
      `pd_ocr_synth.publish.detection`. `build_detection_staging` reads
      the local detection layout (`images/page_*.png` + `labels.json`
      + `recipe.snapshot.yaml`) and emits an HF-shaped staging dir
      (`data/page_*.png` + `labels.json` + `recipe.snapshot.yaml` +
      `README.md` with `pd-ocr-shape: detection/v1` and
      `task_categories: [object-detection]`). `cmd_publish` dispatches
      on `recipe.output.mode` via `_staging_builder_for` (recognition
      → `build_recognition_staging`, detection →
      `build_detection_staging`, unknown → typed `ValueError` mapping
      to PUBLISH_USAGE_EXIT). The dry-run summary degrades to a single
      `Pages: N` line for detection (no `metadata.jsonl` to aggregate
      over). 19 staging tests in `tests/test_publish_detection.py` +
      3 dispatch tests in `tests/test_publish_cli_runner.py` + 1
      end-to-end `cmd_publish --dry-run` test in
      `tests/test_cli_publish.py`. Content-SHA, idempotency,
      preflight, dataset-card, transport orchestration, token
      resolution, and `publish_recognition` (which is shape-agnostic
      `upload_folder`) are all reused as-is from M08 — the
      imagefolder-shaped detection staging dir uploads through the
      same path. Spec 10 ultimately calls for parquet-sharded
      detection via `datasets.Dataset.from_generator(...)
      .push_to_hub(...)` for the 500 MB shard target; that's a
      separate transport-side chunk because (a) `datasets` isn't
      currently a runtime dependency and (b) `push_to_hub` is a
      different transport surface than `upload_folder`. The
      imagefolder staging built here is the prerequisite for either
      upload strategy and works through the existing transport for
      datasets up to the `upload_large_folder` ceiling.

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
