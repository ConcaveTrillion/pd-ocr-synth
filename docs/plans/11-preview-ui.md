# M11 — Preview UI (NiceGUI)

**Goal:** ship the preview UI specced in
[`11-preview-ui.md`](../specs/11-preview-ui.md) — a read-only-on-recipes
NiceGUI app for visual tuning of degradation, fonts, and sample
distribution. CLI remains the contract; the UI is a faster feedback
loop.

This milestone depends on M07 (recognition render) being in place. M08
(publish) and M09 (detection mode) are *not* prerequisites.

## Deliverables

### Optional dependency group

- [ ] Add `[project.optional-dependencies]` (or uv-style
      `[dependency-groups]`) entry `ui = ["nicegui>=2.0", "pillow"]`.
- [ ] Console script `pd-ocr-synth-preview` registered in
      `pyproject.toml`.
- [ ] `make setup-ui` target installs the `ui` group separately,
      mirroring how peers separate optional deps.

### `pd_ocr_synth.preview` package

- [ ] Layered structure mirroring `pd-ocr-labeler` / `pd-ocr-trainer`:
  - `state.py` — `PreviewSession`, `RecipeOverride`, immutable types
  - `operations.py` — `load_recipe`, `render_samples`,
    `apply_override`, `save_override_to_recipe`
  - `viewmodels/` — one per page (`picker_vm.py`, `recipe_vm.py`,
    `sample_vm.py`)
  - `views/` — NiceGUI component implementations
  - `app.py` — `main()` entry point, route registration, CLI flags

### Three pages

- [ ] `/` — recipe picker (uses the same recipe-search-path logic as
      the CLI's `list`).
- [ ] `/recipe/<name>` — config summary + sample grid + override
      controls (toggle, probability slider, regenerate, re-render).
- [ ] `/recipe/<name>/sample/<id>` — sample detail with full manifest
      record and clean-vs-final comparison.

### Override mechanics

- [ ] `RecipeOverride` model with `stage_enabled` and
      `stage_probability` keyed by stage name + index.
- [ ] Render path accepts an optional `overrides: RecipeOverride`
      argument that wraps the recipe's degradation pipeline.
- [ ] "Diff & save" panel:
  - YAML diff of recipe → recipe + overrides
  - Apply writes via `ruamel.yaml` round-trip (preserves comments)
  - Cancel discards

### Reuse, no duplication

- [ ] Sample rendering uses `pd_ocr_synth.render.run_recipe` (M07).
- [ ] Recipe loading uses `pd_ocr_synth.recipe` (M02).
- [ ] No copy of degradation, font sampling, or layout logic in
      `preview/`.

### Local-only by default

- [ ] Bind to `127.0.0.1` by default; `--host` flag required to
      expose externally.
- [ ] No auth in v1 — single-user assumption.

### CLI wiring

- [ ] `pd-ocr-synth-preview` flags:
  - `--recipe <name>` preselects a recipe.
  - `--port <n>` default 8000-something free.
  - `--host <addr>` default `127.0.0.1`.
  - `--workers <n>` plumbed through to the renderer.

### Tests

- [ ] Unit tests for viewmodels (input state → expected operations).
- [ ] Integration: `nicegui.testing` client that renders the picker,
      navigates to a recipe, clicks regenerate, asserts the grid
      populates.
- [ ] Smoke: `make preview-ui-smoke` runs the integration test
      headlessly in CI.

## Validation criteria

```bash
make setup-ui
pd-ocr-synth-preview --recipe gaelic
# → opens on http://127.0.0.1:8080 (or wherever)
```

Manually exercising the UI:

- Recipe picker lists `gaelic` (and any others on the search path).
- Clicking `gaelic` shows the config summary on the left and a grid
  of 24 rendered samples on the right.
- Toggling `paper_texture` off → "Re-render" → 24 samples without
  texture.
- Sliding `jpeg.quality` to 50 → re-render → samples are visibly
  more compressed.
- Clicking a sample → detail page shows the full manifest record,
  including the rendered ink RGB and applied stages with their
  drawn parameters.
- "Diff & save" shows a recipe diff with the changed values; "Apply"
  writes the recipe; reload picker shows updated last-modified time.

## Out of scope

- Free-form YAML editing — VS Code is the editor.
- Auth, publish, fetch flows.
- Detection-mode page preview (could be added once M09 lands; v1
  assumes word_crops/lines).
- Comparison-matrix views (with vs. without per stage).
- Hosted / multi-user mode.

## Risks / open items

- **NiceGUI version drift.** Confirm the version peer projects pin;
  match to avoid mismatched components in CI.
- **Render-on-main-thread blocking.** NiceGUI components freeze if
  the renderer runs synchronously on the UI thread. Use
  `run.io_bound` (NiceGUI helper) or a worker pool.
- **Override diff fidelity.** `ruamel.yaml` round-trip preserves
  comments most of the time; specific edge cases (anchors, nested
  maps) can lose formatting. Prefer minimal in-place edits over
  reserialization.
- **Headless CI.** The integration test must run without a display.
  Use NiceGUI's built-in test client; do not require Playwright in
  CI for v1.

## Sequencing notes

This milestone can begin as soon as M07 lands. M08 and M09 are
parallelizable with M11; nothing in the publish or detection paths
gates the preview UI.

Sizing target: two focused sessions. If it's running long, ship the
recipe picker + sample grid (drop the overrides + diff/save) and
follow up.
