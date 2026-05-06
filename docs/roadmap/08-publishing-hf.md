# M08 — Publishing to Hugging Face

**Goal:** `pd-ocr-synth publish gaelic` ships rendered output to a HF
dataset repo, idempotent and provenance-stamped, consumable by the
trainer's HF source path (per the workspace
[`DATASETS.md`](../../../DATASETS.md)).

Spec: [`10-publishing.md`](../specs/10-publishing.md).

## Deliverables

### `pd_ocr_synth.publish.recognition`

- [x] Read local recognition output: `images/`, `labels.json`,
      `manifest.jsonl`, `recipe.snapshot.yaml`, `stats.json`.
      (`stats.json` is read at README-generation time only; the
      staging builder doesn't need it.)
- [x] Build HF imagefolder layout in a staging dir:
  - [x] `data/*.png` — copied (or symlinked then materialized).
  - [x] `metadata.jsonl` — one row per image with `file_name`, `text`,
        plus flat provenance columns (`font`, `font_size_pt`,
        `degradations`, `corpus`).
  - [x] `recipe.snapshot.yaml` — copied as-is.
  - [ ] `README.md` — generated dataset card with the documented YAML
        front matter (license, task, language, tags, `pd-ocr-shape`,
        `pd-ocr-source`, `pd-ocr-recipe-sha`,
        `pd-ocr-render-tool-version`).

### Auth resolution

- [ ] Order: `--token` flag → `HF_TOKEN` env → `~/.cache/huggingface/token`.
- [ ] Clear error message naming the resolution chain on failure.

### Idempotency

- [ ] Compute a content SHA over the staging directory (sorted file list
      + per-file SHA-256). Persist it as `pd-ocr-content-sha` in the
      dataset card front matter.
- [ ] Before uploading: read the latest commit's `card_data` from HF.
      If `pd-ocr-content-sha` matches → exit 0 with "no changes".

### Upload

- [ ] Use `huggingface_hub.HfApi.upload_large_folder` for recognition.
- [ ] Auto-`create_repo` unless `--no-create`; honor `--private`.
- [ ] Optional `--tag <version>` calls `HfApi.create_tag` after upload.
- [ ] `--message` overrides the auto-generated commit message.

### CLI surface

- [ ] `pd-ocr-synth publish <recipe>` (defaults from recipe
      `publish:` block).
- [ ] `--repo`, `--private`, `--public`, `--license`, `--tag`,
      `--message`, `--token`, `--render-first`, `--no-create`,
      `--dry-run`.
- [ ] `--dry-run` shows: target repo, file count, total size, dataset
      card preview, content SHA — no network calls.

### Tests

- [x] Staging-dir build: given a fixture `<destination>/` with 5
      samples, produce a valid imagefolder structure with
      `metadata.jsonl` matching expected content. Round-trip test
      against the real `RecognitionWriter` locks the M07/M08
      manifest contract.
- [ ] Idempotency: second publish without local changes is a no-op.
- [ ] `--dry-run`: no network, exits 0, prints the plan.
- [ ] Auth error path: missing token → exit 7 with the resolution
      chain printed.

End-to-end test against a private "scratch" repo on HF (gated by an
`HF_TOKEN` env var; skipped on CI without secrets).

## Validation criteria

```bash
# Local prerequisites
pd-ocr-synth render gaelic
export HF_TOKEN=hf_...

# Dry-run preview
pd-ocr-synth publish gaelic --dry-run
# → prints plan; no commit

# Real publish to a personal namespace
pd-ocr-synth publish gaelic --repo me/pd-ocr-synth-gaelic
# → auto-creates repo, uploads, reports commit SHA

# Re-run with no changes
pd-ocr-synth publish gaelic --repo me/pd-ocr-synth-gaelic
# → "no changes" exit 0; no new commit
```

The resulting HF repo opens in the Dataset Viewer with images and
labels rendered.

## Out of scope

- Detection-mode parquet export (M09).
- Pushing labeler-produced datasets (separate project; see workspace
  `DATASETS.md` migration plan).
- Trainer-side HF consumption (separate project; tracked in
  `DATASETS.md`, not this roadmap).

## Risks / open items

- **Large upload reliability.** `upload_large_folder` retries chunks
  but the network-flaky path needs validation. Test with a forced
  network interruption.
- **Card-data lint.** HF rejects cards with unknown front-matter keys
  in some configurations. Verify our `pd-ocr-*` keys land in
  `card_data` (free-form) rather than reserved spots.
- **Private repo defaults.** Default to public for synth (license
  permits and we want shareability) but honor recipe `private: true`.
