# M07 — Output: recognition mode (complete)

**Status:** ✅ landed on `main`. End-to-end `pd-ocr-synth render` writes
the `pd-ocr-trainer/v1` recognition profile that the trainer consumes
unchanged.

**Goal:** end-to-end render: `pd-ocr-synth render gaelic` produces the
configured `output.count` samples in the trainer's recognition layout.

Spec: [`08-output-format.md`](../specs/08-output-format.md).

## Closeout notes

- **Spec/contract reconciliation.** The spec's earlier draft called
  for `labels.csv`; the trainer's actual reader
  (`pd-ocr-trainer/src/pd_ocr_trainer/dataset_store.py` →
  `RecognitionDataset(img_folder=..., labels_path="labels.json")`)
  consumes a JSON map. Per the roadmap principle "spec wins, but if
  the milestone reveals the spec is wrong, fix the spec" — we updated
  spec 08 + spec 10 + the M08 roadmap to specify `labels.json`. The
  writer matches the trainer.
- **Progress reporting** is a tiny stderr bar plus a final `rendered
  N/N in T.Ts (R samples/s)` line — no tqdm dep added. The roadmap
  noted tqdm; the homegrown reporter is enough for v1 and saves a
  dependency. Re-evaluate if a milestone needs nested bars.
- **Cross-project integration test deferred.** The trainer's
  `dataset_store.py` does heavy work at *import* time (creates
  `ml-training/`, `ml-validation/`, `pd-ml-models/` under
  `~/.local/share/`), making it intrusive to import inside synth's
  test suite. Instead, we lock the contract structurally: the
  writer's `labels.json` shape (`{"<image>.png": "<text>"}`) matches
  what the trainer's reader expects, asserted by
  `tests/test_cli_render.py::test_render_labels_json_matches_trainer_recognition_contract`.
  When the trainer's UI/CLI grows a "load synth dataset" path, that's
  the right moment to add a real cross-repo integration check.

## Deliverables

### Writer

- [x] `pd_ocr_synth.output.RecognitionWriter` matching the documented
      layout:
  - `images/<NNNNNNN>.png` with zero-padded names sized for `count`
    (min 7 digits so smoke runs and full runs share the convention).
  - `labels.json` — JSON map `{image_name: text}` matching the
    trainer's `RecognitionDataset` reader.
  - `manifest.jsonl` with one record per attempted sample (rendered
    or skipped), full provenance per spec 08.
  - `recipe.snapshot.yaml` — the resolved recipe + tool version +
    effective seed + SHA-256 of every font and local corpus file.
  - `stats.json` — samples_planned, samples_written, samples_skipped,
    skip_reasons, fonts_used histogram, tokens_unique, wall time.

### Render orchestration

- [x] `pd_ocr_synth.render.run_recipe(recipe, output_dir, ...)` ties
      M03–M06 together: corpus → transforms → tokenize → render →
      degrade → write.
- [x] Worker pool (multiprocessing) keyed by sample index for
      determinism — same `seed + index` deterministic output bytes
      regardless of worker count.
- [x] Progress reporting + rate to stderr (homegrown, no tqdm).
- [x] `pd_ocr_synth.render.plan_recipe(...)` powers `--dry-run`.

### Resume / idempotency

- [x] `--force`: clear output before render.
- [x] `--resume`: continue, skipping `already_rendered(idx)`. Snapshot
      compare gates resume — drift in seed, tool version, or input
      file hashes refuses to proceed.
- [x] Default: refuses to write into a non-empty directory; error
      points at `--force` / `--resume` and exits 6 (DESTINATION_EXIT).
- [x] `--force` and `--resume` are mutually exclusive (USAGE_EXIT).

### CLI surface

- [x] `pd-ocr-synth render <recipe>` — full run.
- [x] `pd-ocr-synth render <recipe> -c 500 -o /tmp/X` — overrides for
      smoke tests.
- [x] `pd-ocr-synth render <recipe> --dry-run` — print plan (sample
      count, output dir, fonts, transforms, corpus chars) without
      writing.

### Trainer integration test

- [x] Structural contract test — `labels.json` is a JSON object whose
      keys are PNG filenames (no path components) and values are
      non-empty plain-text labels, every key pointing at an image on
      disk. (See closeout notes above for why a live
      `dataset_store.py` import was skipped.)

## Validation criteria

```bash
pd-ocr-synth render gaelic
# → N PNG files + labels.json + manifest.jsonl + recipe.snapshot.yaml + stats.json
# → wall time + rate printed
# → exit 0

pd-ocr-synth render gaelic   # re-run
# → "destination not empty; pass --force or --resume" (exit 6)

pd-ocr-synth render gaelic --resume
# → skips through existing samples; renders any remaining

pd-ocr-synth render gaelic --dry-run
# → prints planned config without writing
```

## Out of scope

- HF publish (M08).
- Detection mode (M09).

## Deferred

- Real cross-project integration test that imports
  `pd-ocr-trainer/dataset_store.py` end-to-end. See closeout notes —
  the structural test is sufficient until the trainer grows a
  cleaner import surface.
- Per-sample `corpus.{provider, key, offset}` provenance in the
  manifest. Currently we record `text` plus the recipe-level transform
  and degradation lists; per-sample source-document tracking lands when
  M08 (publish) needs it for `metadata.jsonl` and the dataset card.
- Per-degradation-stage *applied* telemetry (currently the manifest
  records configured stages, not which probability rolls fired). M09
  needs the per-stage roll outcome for bbox-aware geometric stages,
  so the surface lands there.
