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
  - [x] `README.md` — generated dataset card with the documented YAML
        front matter (license, task, language, tags, `pd-ocr-shape`,
        `pd-ocr-source`, `pd-ocr-recipe-sha`,
        `pd-ocr-render-tool-version`). `pd-ocr-content-sha` is added
        at upload time, not by the staging builder.

### Auth resolution

- [x] Order: `--token` flag → `HF_TOKEN` env → `~/.cache/huggingface/token`.
      (`pd_ocr_synth.publish.auth.resolve_hf_token`; honors `HF_HOME`
      override.)
- [x] Clear error message naming the resolution chain on failure.
      (`AuthError` carries `format_resolution_chain` output.)

### Idempotency

- [x] Compute a content SHA over the staging directory (sorted file list
      + per-file SHA-256). Persist it as `pd-ocr-content-sha` in the
      dataset card front matter. (`pd_ocr_synth.publish.content_sha` +
      `apply_content_sha_to_readme`; staging builder embeds it before
      upload.)
- [x] Before uploading: read the latest commit's `card_data` from HF.
      If `pd-ocr-content-sha` matches → exit 0 with "no changes".
      (`pd_ocr_synth.publish.idempotency.check_idempotency`; orchestrator
      short-circuits on `IdempotencyState.UP_TO_DATE`.)

### Upload

- [x] Use `huggingface_hub.HfApi.upload_large_folder` for recognition.
      (`HfHubTransport.upload_folder` in `hf_hub_transport.py`; SDK
      pulled in via the `[publish]` optional-extra group.)
- [x] Auto-`create_repo` unless `--no-create`; honor `--private`.
      (`publish_recognition` orchestrator + CLI `--no-create` /
      `--private` flags.)
- [x] Optional `--tag <version>` calls `HfApi.create_tag` after upload.
      (Orchestrator runs the tag step only after a successful upload;
      `HfHubTransport.create_tag` uses `exist_ok=True`.)
- [x] `--message` overrides the auto-generated commit message.
      (`pd_ocr_synth.publish.commit_message.resolve_commit_message`;
      default falls back to `pd-ocr-synth render @<recipe-sha>` derived
      from the staging README's `pd-ocr-recipe-sha`. The
      `upload_large_folder` SDK call cannot stamp `commit_message`
      on the remote commit — flag accepted with a stderr warning;
      see "Residual M08 work → Commit-message limitation" below for
      the chosen strategy and citation.)

### CLI surface

- [x] `pd-ocr-synth publish <recipe>` (defaults from recipe
      `publish:` block). (`cli.py` + `publish.cli_runner.cmd_publish`;
      missing `--repo` falls back to `recipe.publish.hf_dataset.repo`.)
- [x] `--repo`, `--private`, `--public`, `--token`, `--tag`,
      `--message`, `--no-create`, `--dry-run` accepted and threaded
      through to the runner.
- [x] `--dry-run` shows: target repo, file count, total size, dataset
      card preview, content SHA — no network calls.
      (`run_publish_dry_run` + `format_dry_run_plan`.)

### Tests

- [x] Staging-dir build: given a fixture `<destination>/` with 5
      samples, produce a valid imagefolder structure with
      `metadata.jsonl` matching expected content. Round-trip test
      against the real `RecognitionWriter` locks the M07/M08
      manifest contract. (`tests/test_publish_recognition.py`.)
- [x] Idempotency: second publish without local changes is a no-op.
      (`tests/test_publish_orchestrator.py::test_round_trip_second_publish_is_no_op`,
      `tests/test_cli_publish_upload.py::test_real_upload_round_trip_first_then_no_op`.)
- [x] `--dry-run`: no network, exits 0, prints the plan.
      (`tests/test_cli_publish.py` covers the parser and end-to-end
      dry-run dispatch; `tests/test_publish_cli_runner.py` covers the
      structured `DryRunPlan` formatter.)
- [x] Auth error path: missing token → exit 7 with the resolution
      chain printed. (`tests/test_cli_publish.py::test_publish_real_upload_without_token_exits_seven`
      asserts `--token` and `HF_TOKEN` appear in stderr at exit 7.)

End-to-end test against a private "scratch" repo on HF (gated by an
`HF_TOKEN` env var; skipped on CI without secrets) — **not yet
implemented**; tracked under "Residual M08 work" below.

## Residual M08 work

The leaf primitives and end-to-end CLI dispatch are in place. The
following gaps remain — pick any of them as a future small chunk.

### CLI flags accepted but not wired through

- [x] `--license <LICENSE>` is wired end-to-end: argparse → `_cmd_publish`
      → `cmd_publish(license_override=...)` → `build_recognition_staging`
      → `load_card_inputs` → `DatasetCardInputs.license_override` →
      front-matter `license:` key. The flag wins over
      `recipe.publish.hf_dataset.license` per spec 10 § Recipe
      `publish:` block; a missing flag falls back to the recipe value
      (or omits the key entirely). Both dry-run and real-upload paths
      are covered by tests in `test_cli_publish.py` and
      `test_cli_publish_upload.py`; unit-level coverage in
      `test_publish_dataset_card.py`.
- [x] `--render-first` is wired end-to-end: argparse → `_cmd_publish`
      → `cmd_publish(render_first=...)` → `_default_render_first`
      (which lazy-imports `run_recipe`) → publish staging build. Spec
      10 § When to publish ("Pass `--render-first` to chain them") is
      satisfied; render failures map to RENDER_EXIT (5), distinct
      from publish-family failures (exit 7). The render callable is
      injectable for hermetic tests in
      `tests/test_cli_publish_render_first.py`. The default callable
      passes `force=True` so a re-run with the flag clears the
      destination — otherwise the existing render's "non-empty
      destination" guard would refuse and exit 6, surprising someone
      who just asked us to chain render + publish. Recipe-level
      `count` / `seed` / `workers` are NOT plumbed through the chain;
      `--render-first` runs the recipe as written. Users who want a
      smoke-sized subset render separately (`render -c N`) and then
      run plain `publish`.

### Commit-message limitation

- [x] **Strategy A picked: document + warn.** Investigation
      (huggingface_hub 1.13.0, `hf_api.py:5859-5973`) confirmed
      `upload_large_folder` does not even accept a `commit_message`
      argument; its docstring states "you cannot set a custom
      `commit_message` and `commit_description` since multiple commits
      are created." Strategies (B) post-hoc empty commit and (C/D)
      switch to `upload_folder` were rejected: (B) is fragile because
      `create_commit` requires at least one operation, and (C/D) would
      contradict spec 10 § Tooling used, which explicitly mandates
      `upload_large_folder` for the resumability behavior on large
      recognition datasets. Outcome: `HfHubTransport.upload_folder`
      continues to accept and ignore `commit_message` (Protocol
      contract), the CLI runner emits a single-line stderr warning
      when `--message` is explicitly supplied
      (`_MESSAGE_LIMITATION_WARNING` in `cli_runner.py`), and
      `docs/specs/10-publishing.md` § Tooling used grew a "Known
      limitation" subsection plus a row in § Errors and recovery.
      Tests in `tests/test_cli_publish_upload.py` lock both the
      warning-on-`--message` and silence-on-default behaviors. The
      `FakeTransport` test seam still records `commit_message`
      verbatim, which is correct — the Protocol is honest, and a
      future detection-mode `push_to_hub` path (which DOES honor the
      message) will use the same field.

### Content-SHA scope

- Spec 10 § Idempotency says the SHA should cover "image bytes +
  metadata + recipe snapshot". The current `compute_content_sha`
  hashes every regular file under the staging dir *except* the
  README's `pd-ocr-content-sha` line (which it strips before
  hashing). That matches the spec in practice. Worth a confirming
  test that hand-edits a staged image byte and observes a digest
  change end-to-end.

### End-to-end live HF test

- [x] Implemented at `tests/integration/test_publish_live_hf.py`.
      Gated behind `PD_OCR_SYNTH_HF_E2E=1` **and** `HF_TOKEN`
      (write-scope) — both must be set for the live test to run; with
      either missing, `pytest.mark.skipif` skips the test cleanly so
      `make ci` collects it but never reaches the network. The test
      builds a 2-sample local recognition output programmatically (no
      `run_recipe` — keeps the wire footprint tiny), stages it through
      `build_recognition_staging`, and drives the **production**
      transport via `make_default_transport(token)`. Asserts:
      first publish lands `CREATED`-or-`UPLOADED` with a non-empty
      commit SHA, the staged README's front matter carries every
      `pd-ocr-*` key (driven through the public
      `check_required_front_matter` helper so a regression in the
      builder fails the same way pre-flight would), and a second
      publish without local changes returns `PublishState.NO_CHANGE`
      with an empty `commit_sha` (the spec's "exit 0 with 'no changes'
      and do not commit" branch). Cleanup deletes the test repo via
      `HfApi.delete_repo(missing_ok=True)` in a `finally`-shaped
      fixture; cleanup errors are swallowed so a transient HF outage
      at teardown doesn't mask a real test failure. The default repo
      is `ConcaveTrillion/pd-ocr-synth-livetest-recognition`,
      overrideable via `PD_OCR_SYNTH_HF_E2E_REPO=OWNER/NAME`. Three
      always-on collection-sanity tests in the same file lock the
      gating helper (`_live_enabled`) and the default-repo invariant
      so a future refactor that accidentally inverts the env-var check
      can't silently disable live coverage. The opt-in convention
      (`tests/integration/`, `@pytest.mark.integration`,
      `PD_OCR_SYNTH_<SUITE>_…` env-var prefix) is documented in the
      module's docstring as the pattern future live tests should
      follow.

### Repo-id / branch validation

- Spec 10 is silent on the allowed character set for `OWNER/NAME` and
  `--tag`. The orchestrator currently forwards the values verbatim to
  the SDK and lets HF reject malformed ones via `TransportError`.
  Tighten this only after the spec gains explicit rules — out of
  scope until then.

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
