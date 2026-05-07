# Roadmap — pd-ocr-synth implementation

The specs in [`../specs/`](../specs/) describe the destination. This
roadmap is the path to get there. Each milestone is a vertical slice
that ends in something runnable; nothing is "framework first." The
order favors **dev tooling and feedback loops early** so every later
milestone benefits from them.

## Milestones

| # | Milestone | Goal | Outcome |
|---|-----------|------|---------|
| [00](00-bootstrap.md) | Bootstrap | Repo exists with workspace conventions | `git clone` works; matches peer-project layout |
| [01](01-dev-tooling.md) | Dev tooling parity | Make / lint / test / pre-commit / CI | `make setup && make test && make lint` all green on a stub |
| [02](02-recipe-schema.md) | Recipe schema + validator | Recipes load, validate, list, init | `pd-ocr-synth validate gaelic` passes |
| [03](03-corpus.md) | Corpus providers + cache | Pull text from local + web + Wikisource | `pd-ocr-synth fetch gaelic` warms a cache |
| [04](04-text-transforms.md) | Text transforms | Lenition, long-s, Tironian et, base ops | Round-trip transform tests pass |
| [05](05-rendering.md) | Rendering | HarfBuzz, fonts, word-crop layout | Single sample renders deterministically |
| [06](06-degradation.md) | Degradation pipeline | Geometric + optical + paper + jpeg stages | Sample with degradation looks plausible |
| [07](07-output-recognition.md) | Output: recognition mode | Writer for `pd-ocr-trainer/v1` recognition | Trainer can read 50k Gaelic crops |
| [08](08-publishing-hf.md) | HF publish | Push rendered output to HF dataset repo | A dataset on HF that the trainer can consume |
| [09](09-detection-mode.md) | Output: detection mode | Layouts: paragraphs, pages; bbox-aware degradations | Trainer detection profile fed |
| [10](10-stretch.md) | Stretch | Extra recipes, cloud render, polish | Opt-in follow-ups |
| [11](11-preview-ui.md) | Preview UI | NiceGUI for visual recipe tuning | `pd-ocr-synth-preview --recipe gaelic` works |
| [12](12-glyph-annotations.md) | Glyph-level annotations | Per-word ligature / long-s / swash side channel | Synth emits `glyph_annotations.json` alongside `labels.json` |

## NiceGUI surface — preview only, not edit

A full YAML editor reproduces VS Code at a worse quality bar — recipes
are short and well-documented, and the YAML extension already validates
against `recipe.schema.json` (M02 deliverable). Building a custom editor
surface adds maintenance for low leverage.

A **preview UI**, on the other hand, has high leverage: pick a recipe,
render N samples with the current degradation pipeline, display them in
a grid, and let you toggle stages or slide probabilities to see the
effect immediately. Collapses "edit YAML → re-render → open file
manager" into one page. It also fits the workspace pattern (both
`pd-ocr-labeler` and `pd-ocr-trainer` use NiceGUI).

Captured as **M11** with a separate spec at
[`../specs/11-preview-ui.md`](../specs/11-preview-ui.md). The UI is
read-only on recipes; its only write feature is an explicit "Diff &
save" panel that shows the recipe-vs-overrides diff before persisting.
M11 depends on M07 only — it can land alongside M08 / M09 without
ordering constraints.

## Working principles for the roadmap

1. **Vertical slices.** Every milestone leaves something the user can
   run. No "lay the foundation" milestones with no demo.
2. **Dev tooling first.** M00 + M01 are heavily weighted toward
   developer experience because every later milestone benefits.
3. **One recipe drives the work.** `recipes/gaelic.yaml` is the
   integration test through every milestone.
4. **Spec is the contract.** When in doubt, the spec wins; if a
   milestone reveals the spec is wrong, update the spec before the
   code.
5. **No stubs in main.** A milestone is done when its surface is real,
   not when there's a placeholder. Push WIP to a branch.

## Sequencing notes

- M00–M01 are pure setup; no recipe execution.
- M02 unlocks `validate` and `list`, which are nice for users tweaking
  recipes even before render works.
- M03 + M04 + M05 + M06 are pipeline stages; each one should be
  testable in isolation against fixtures.
- M07 ties them together end-to-end into recognition output.
- M08 (publish) deliberately follows M07 so the publish path has real
  output to ship.
- M09 (detection mode) is gated separately because it requires
  bbox-aware geometric degradation — different testing surface from
  recognition.

## Sizing

Each milestone is sized for a focused session, not weeks. If a
milestone is dragging past its scope, split it or move work to a
later one. The sizes below are *aspirational order of magnitude*,
not commitments.

| Milestone | Rough scope |
|-----------|-------------|
| 00 | half a session |
| 01 | one session |
| 02 | one session |
| 03 | one to two sessions |
| 04 | one session |
| 05 | two sessions |
| 06 | two sessions |
| 07 | one session |
| 08 | one session |
| 09 | two sessions |
| 10 | open-ended |
| 11 | two sessions (depends on M07; can run in parallel with M08/M09) |
| 12 | one to two sessions (depends on M07 + M09 + pd-book-tools data model) |

Total: ~12–15 focused sessions to a usable v0.1.0 (engine + publish);
add ~2 sessions for the preview UI.
