# M11 — Preview UI scoping notes

Companion to [`11-preview-ui.md`](./11-preview-ui.md) and the spec at
[`../specs/11-preview-ui.md`](../specs/11-preview-ui.md). Written ahead of
implementation to scope a clean **first chunk** that a future supervised
iteration can land in one session, plus an ordered chunk menu so subsequent
iterations know exactly what to pick up.

The spec is good, but it is a **finished feature description**, not an
implementation plan. This doc translates that spec into landable chunks —
each chunk is small enough that `make ci` can stay green throughout.

## Headline findings from the M11 read

1. **NiceGUI is already declared.** `pyproject.toml` has
   `optional-dependencies.ui = ["nicegui>=2.0"]` and the spec calls for the
   same group plus `pillow` (already a hard dep, so no change). No new
   transitive dep cost beyond NiceGUI itself.
2. **`pd-ocr-labeler` already configures `nicegui.testing.user_plugin`.** Its
   `tests/conftest.py` is the working pattern to copy: `pytest_plugins =
   ["nicegui.testing.user_plugin"]` plus an XDG isolation fixture. Our suite
   would take the same shape under a guarded marker.
3. **`pd_ocr_synth.render.preview` already exists** (`src/pd_ocr_synth/
   render/preview.py`) and exposes `run_preview(...)` — exactly the kind of
   re-render entry point the UI needs. It writes to a directory and returns
   a `PreviewStats` dataclass. M11 should call this function (or a thin
   wrapper) rather than duplicate render driving.
4. **CLI has the building blocks.** `recipe_search.iter_recipes`,
   `validation.validate_recipe`, `lint.lint_recipe`, `recipe.load_recipe`
   are all importable, well-tested, and sufficient for the picker page +
   recipe view's left pane.
5. **Two specs disagree on `ruamel.yaml`.** Roadmap M02 says "pyyaml only,
   ruamel deferred". Roadmap M11 + spec M11 require `ruamel.yaml` for the
   round-trip "diff & save" path. Resolution: the diff/save chunk is the
   one that introduces `ruamel.yaml` (under the `ui` extras, not the core
   dep set). All chunks before it stay on `pyyaml`.
6. **Open question on package layout.** Spec says `pd_ocr_synth.preview`,
   but the existing `pd_ocr_synth.render.preview` module would shadow it.
   The new package can live at `pd_ocr_synth.ui` (cleaner: `ui` matches the
   extras group name) — this is a **decision flagged for the user** before
   the first implementation chunk lands.

## NiceGUI features actually needed

Cataloged from the spec so we know exactly what API surface to learn:

- **Page routing** — `@ui.page('/')`, `@ui.page('/recipe/{name}')`,
  `@ui.page('/recipe/{name}/sample/{id}')`. Standard NiceGUI.
- **Components** — `ui.label`, `ui.button`, `ui.slider`, `ui.switch`,
  `ui.image`, `ui.grid`, `ui.linear_progress`, `ui.notify`. All stock.
- **Async render dispatch** — `nicegui.run.io_bound(...)` to keep the
  renderer off the UI thread (called out as a risk in the roadmap).
- **Per-tab state** — `app.storage.tab` (NiceGUI 2.x) for `PreviewSession`.
  Browser tab close ⇒ session GC'd.
- **Static-file serving** — `app.add_static_files(...)` for the temp-dir
  preview images. Simpler than streaming through a route.
- **Test client** — `nicegui.testing.User` (the `user_plugin` plugin
  pd-ocr-labeler already uses). Headless, no browser needed for unit-level
  integration. Playwright is **out of scope for v1** (matches roadmap).

## Reusable existing infrastructure

Every M11 surface can be built on something that already exists:

| M11 surface                       | Reuses (path)                                                      |
|-----------------------------------|--------------------------------------------------------------------|
| `/` recipe picker                 | `pd_ocr_synth.recipe_search.iter_recipes`                          |
| Recipe load + validate            | `pd_ocr_synth.recipe.load_recipe`, `pd_ocr_synth.validation`       |
| Recipe lint banner                | `pd_ocr_synth.lint.lint_recipe`                                    |
| Sample grid render                | `pd_ocr_synth.render.preview.run_preview` (already deterministic)  |
| Manifest read for sample-detail   | The `manifest.jsonl` produced by `run_preview`                     |
| Resolve recipe by name            | `pd_ocr_synth.recipe_search.resolve_recipe`                        |
| Stats display ("last rendered")   | `<destination>/stats.json` (existing format)                       |
| CLI arg parsing pattern           | `pd_ocr_synth.cli.build_parser` (mirror its `argparse` shape)      |

**No duplicated logic** is needed for the first chunk — the UI is purely
a presentation layer on top of `run_preview`.

## Tests we'd ship per chunk

Following the project's pattern (no network, deterministic, fast):

- **viewmodel unit tests** — pure-Python tests of state transitions; no
  NiceGUI imported. Fastest tier.
- **integration tests** — `nicegui.testing.User` programmatic clicks;
  guarded by an `@pytest.mark.ui` marker so the default `make test` can
  skip them when the `ui` extras aren't installed.
- **CLI smoke** — `pd-ocr-synth-preview --help` exits 0 and prints flags.
  Doesn't need the NiceGUI server actually running.

A new pytest marker `ui` lands with the first chunk; `make ci` stays green
on a default install (without the `ui` extras) by `--deselect`-ing the
marker, while a new `make preview-ui-smoke` target runs only `-m ui`.

## Dep-add cost

- **NiceGUI 2.x**: ships with `fastapi`, `starlette`, `uvicorn`,
  `python-multipart`, `vbuild`, `markdown2`, `httpx` (already a synth dep),
  `aiofiles`, `pscript`. ~40 MB installed, no native extensions.
- **`ruamel.yaml`**: pure Python, ~1 MB. Lands only with the diff/save
  chunk (chunk M11.4 below), keeping the cheaper chunks dep-free.
- All cost is paywalled behind `pip install pd-ocr-synth[ui]`. Default
  installs and `make test` are unaffected.

## First chunk: M11.1 — picker page + CLI scaffold

Smallest unit that demonstrates "NiceGUI app starts, lists recipes" without
introducing render plumbing or override mechanics. Definition of done:

1. New package `pd_ocr_synth.ui/` with:
   - `__init__.py` (empty)
   - `state.py` — frozen `PreviewSession` with `recipe: Recipe | None` only;
     no `RecipeOverride` yet. Just enough to type-check the picker.
   - `operations.py` — `list_recipes() -> list[RecipeEntry]` (one-line
     wrapper around `iter_recipes`) and `load_recipe_by_name(name: str)
     -> Recipe`.
   - `views/picker.py` — NiceGUI page at `/` showing a table of recipe
     name + path + click → "(coming soon)" notify. No real recipe view yet.
   - `app.py` — `main()` entry point with argparse: `--port`, `--host`,
     `--recipe` (no-op flag in this chunk; reserved for chunk M11.2).
2. Console script `pd-ocr-synth-preview = "pd_ocr_synth.ui.app:main"`
   registered in `pyproject.toml`.
3. New `Makefile` targets:
   - `setup-ui` — `uv sync --extra ui`.
   - `preview-ui-smoke` — runs the `-m ui` test selection.
4. New pytest marker `ui` registered in `pyproject.toml`'s
   `tool.pytest.ini_options.markers`.
5. Tests:
   - Unit: `tests/ui/test_operations.py` exercises `list_recipes()` against
     a tmp recipes dir (mirrors existing `recipe_search` tests).
   - Smoke: `tests/ui/test_app_cli.py` runs `pd-ocr-synth-preview --help`
     via `subprocess.run` and asserts exit 0 + `--port` substring.
   - Integration: `tests/ui/test_picker_page.py` (marker `ui`) — uses
     `nicegui.testing.User` to navigate to `/` and assert a recipe name
     appears as a table row.

What's **not** in M11.1:

- No `/recipe/<name>` page.
- No render call, no images served.
- No overrides, no diff/save.
- No sample-detail page.

Why this chunk: it pays the dep + boilerplate cost once, gets routing
wired, gets the test plumbing in place, and is something the maintainer
can `make setup-ui && pd-ocr-synth-preview` and click around in to confirm
the foundation is right before we invest in render integration.

## Chunk menu (in order, for future iterations)

| Chunk    | Surface                                  | New deps        | Tests                                  |
|----------|------------------------------------------|-----------------|----------------------------------------|
| M11.1    | picker page + CLI scaffold + tests       | `nicegui`       | unit + smoke + 1 integration           |
| M11.2    | `/recipe/<name>` page, left-pane summary | —               | viewmodel unit + integration nav       |
| M11.3    | Sample grid + render call (no overrides) | —               | render-with-temp-dir integration       |
| M11.4    | Stage toggle + slider + re-render        | —               | override-applied integration           |
| M11.5    | Diff & save panel                        | `ruamel.yaml`   | round-trip preservation unit tests     |
| M11.6    | `/recipe/<name>/sample/<id>` detail page | —               | manifest-row display integration       |
| M11.7    | Polish: progress bar, errors, banners    | —               | error-path integration                 |

Each chunk is a single commit (or two, if the integration test needs to
land separately from the feature). After M11.7 the milestone closes.

## Decisions to confirm with the user before M11.1 lands

1. **Package name** — `pd_ocr_synth.ui` (recommended; mirrors `[ui]`
   extras and avoids shadowing `pd_ocr_synth.render.preview`) vs. the
   spec's `pd_ocr_synth.preview`.
2. **NiceGUI version pin** — `nicegui>=2.0` (current pyproject) vs. match
   `pd-ocr-labeler`'s pin (it floats `>=1.4`). Recommend pinning a
   minimum 2.x to avoid the 1.x→2.x storage API split.
3. **Default port** — spec doesn't pick. Recommend `8765` (memorable, not
   a common conflict). NiceGUI's default 8080 is too collision-prone in a
   workspace with multiple `pd-*` UIs.
4. **Test marker name** — `ui` vs. `nicegui` vs. `preview_ui`. Recommend
   plain `ui` (short, matches the extras name).

Once those four are answered, M11.1 is roughly an hour of focused work.
