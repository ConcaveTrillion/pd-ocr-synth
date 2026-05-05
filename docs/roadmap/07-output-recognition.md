# M07 — Output: recognition mode

**Goal:** end-to-end render: `pd-ocr-synth render gaelic` produces 50k
samples in `pd-ocr-trainer/v1` recognition layout that the trainer can
consume without changes.

Spec: [`08-output-format.md`](../specs/08-output-format.md).

## Deliverables

### Writer

- [ ] `pd_ocr_synth.output.RecognitionWriter` matching the documented
      layout:
  - `images/0000000.png` with zero-padded names sized for `count`.
  - `labels.csv` (no header, two columns).
  - `manifest.jsonl` with one record per sample (full provenance).
  - `recipe.snapshot.yaml` — the resolved recipe, paths absolute,
    fonts/corpora SHA-256 stamped.
  - `stats.json` — samples_planned, samples_written, samples_skipped,
    skip reasons, fonts_used histogram, unique tokens, wall time.

### Render orchestration

- [ ] `pd_ocr_synth.render.run_recipe(recipe, output_dir)` ties
      M03–M06 together: corpus → transforms → tokenize → render →
      degrade → write.
- [ ] Worker pool (multiprocessing) keyed by sample index for
      determinism.
- [ ] Progress bar (tqdm) and rate reporting in stderr.

### Resume / idempotency

- [ ] `--force`: clear output before render.
- [ ] `--resume`: continue from highest existing sample index, only if
      `recipe.snapshot.yaml` matches the current resolved recipe.
- [ ] Default: refuse to write into a non-empty directory; clean error
      message pointing at `--force` / `--resume`.

### CLI surface

- [ ] `pd-ocr-synth render <recipe>` — full run.
- [ ] `pd-ocr-synth render <recipe> -c 500 -o /tmp/X` — overrides for
      smoke tests.
- [ ] `pd-ocr-synth render <recipe> --dry-run` — print plan (sample
      count, output dir, fonts, transforms) without writing.

### Trainer integration test

- [ ] Render 50 samples from `gaelic.yaml` into a temp `ml-recognition/
      gaelic/recognition` dir.
- [ ] Use `pd-ocr-trainer`'s `dataset_store.py` to load it. The test
      passes if the trainer reports the expected sample count and
      doesn't raise on schema.

This is the first cross-project integration; treat the trainer's
loader as the API contract.

## Validation criteria

```bash
pd-ocr-synth render gaelic
# → 50000 .png files + labels.csv + manifest.jsonl + recipe.snapshot.yaml + stats.json
# → wall time printed; rate printed
# → exit 0

pd-ocr-synth render gaelic   # re-run
# → "destination not empty; pass --force or --resume" (exit 6)

pd-ocr-synth render gaelic --resume
# → skips through existing samples; renders any remaining
```

`pd-ocr-trainer` reads the resulting profile and reports:
`profile=gaelic, samples=50000, schema=pd-ocr-trainer/v1`.

## Out of scope

- HF publish (M08).
- Detection mode (M09).

## Risks / open items

- **Multi-process determinism.** Seed handling across workers must
  give the same per-sample output regardless of worker count. Use
  `seed + sample_index` for the per-sample RNG, not the worker ID.
- **Disk usage.** 50k word crops ≈ 250MB. Confirm the trainer's
  consumer doesn't choke on that before going to higher counts.
- **Trainer profile shape stability.** If the trainer's
  `dataset_store.py` shape evolves between M07 spec time and M07
  implementation time, the writer needs to follow. Lock the contract
  via the integration test.
