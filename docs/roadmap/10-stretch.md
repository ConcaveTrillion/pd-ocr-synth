# M10 — Stretch

The engine works after M00–M09. M10 is open-ended polish, additional
recipes, and cloud rendering. The preview UI moved out of stretch and
is now [M11](11-preview-ui.md) with its own
[spec](../specs/11-preview-ui.md).

## Status — substantially complete

The engine-quality polish chunks shipped in M10 — recipe linter,
per-render audit log, and visual regression — are all on `main` and
exercised by default `make ci`. M10 is intentionally open-ended (a
"polish + extra recipes + cloud" bucket rather than a single
deliverable), so "complete" here means **the polish loop closed**:
the linter catches recipe footguns before render time, the audit
log gives a forensic trail across runs, and the visual regression
pin catches unintended rendering drift on every commit. New
recipes (Fraktur, Greek, etc.) and cloud rendering remain genuinely
open and live in **Future work — beyond M10** below; promoting any
one of them to its own milestone is a deliberate next-step decision
rather than M10 residual.

M10 polish landed across twelve commits: `834e449` (recipe linter),
`2f05c3a` (per-render audit JSONL), `862afd6` (audit read
subcommand), `96bbc99` (lint `--json`), `9e1e876`
(`lint_zero_weight_font` + `lint_all_optional_fonts`), `ff2f858`
(audit `--since` / `--until` / `--recipe-sha`), `cb5a287` (audit
`--summary`), `69e9060` (lint `--strict` CI gate), `33df4d2` (visual
regression sha-digest pins), `27f4bd1` (audit timezone normalization
fix), `07dccd1` (audit `schema_version` forward-compat skip),
`a087054` (FD-leak regression test + Pillow version pin doc).

Test count: 882 passed / 4 skipped under default `make ci`.

## M10 polish — done

The five capability chunks below shipped in M10 and are locked in
tests under default `make ci`. Each is sized as a "promotable
candidate" the QoL section originally listed; none of them grew big
enough to warrant their own milestone. Future chunks under each
heading are tracked in **Future work — beyond M10**.

### Recipe linter

- [x] `pd-ocr-synth lint <recipe>` layered on top of `validate`,
      with seven heuristic warning codes:
      `lint_degradation_always_certain`, `lint_single_font`,
      `lint_no_text_transforms`, `lint_low_sample_count`,
      `lint_seed_default`, `lint_zero_weight_font`,
      `lint_all_optional_fonts` (`834e449`, `9e1e876`).
- [x] `--json` emits a single object with `recipe`, `path`, `ok`,
      `validation`, `lint`, and `summary` keys; issue dicts carry
      `severity` / `code` / `message` / `location` for editor / CI
      consumption (`96bbc99`).
- [x] `--strict` flips exit code from 0 to 1 when any warning is
      present, suitable as a CI / pre-commit gate. Validation errors
      still take precedence (exit 3 — strict never _downgrades_ a
      stricter code), and the body is identical between strict and
      lenient runs (text or `--json`) (`69e9060`).

### Audit log

- [x] `run_recipe` appends one JSONL line per invocation to
      `<output_dir>/_audit.jsonl` carrying timestamp, recipe name +
      source-bytes SHA-256, output dir, seed, effective count,
      worker count, rendered/skipped counts, and wall-time. Disabled
      via CLI `--no-audit` or env var `PD_OCR_SYNTH_NO_AUDIT=1`
      (`2f05c3a`).
- [x] `pd-ocr-synth audit <output_dir>` reads JSONL back, with
      `--json` for machine-readable output and `--limit N` to tail to
      the most recent N entries (`862afd6`).
- [x] `--since` and `--until` accept ISO-8601 timestamps (date-only
      allowed; both bounds inclusive on the second-precision audit
      timestamp) and normalize non-UTC offsets before lex-comparing
      (`27f4bd1`); `--recipe-sha PREFIX` does case-insensitive
      prefix-match against `recipe_sha` (entries with a null SHA are
      excluded — a SHA filter implies "find runs of recipe X").
      Filters apply _before_ `--limit` so "last N matching" composes
      naturally (`ff2f858`).
- [x] `--summary` aggregates the matched window into entry / sample
      / runtime totals, distinct recipe-name and recipe-SHA counts,
      oldest / newest timestamp, and a top-3 recipe-SHA frequency
      list. Composes with all existing filters so "summarize the
      last N entries from today" is a one-liner; pair with `--json`
      for a single fixed-shape object suitable for dashboards
      (`cb5a287`).
- [x] `read_audit_entries` skip-with-warning on unknown
      `schema_version` so newer-than-reader log files don't break
      existing tools (`07dccd1`).
- [x] `--audit-file PATH` overrides the default
      `<output_dir>/_audit.jsonl` lookup so users can replay archived
      staging-dir logs or aggregate files without copying them into a
      render dir. Composes with all existing filters (`--since`,
      `--until`, `--recipe-sha`, `--limit`, `--summary`, `--json`);
      missing override path maps to the same exit-6 destination
      family as a missing default file.

### Visual regression tests

- [x] `tests/test_render_visual_regression.py` pins the PNG sha256
      of the four render entry points (`render_word_crop`,
      `render_line`, `render_paragraph`, `render_page`) under a tiny
      canonical recipe (single Gaelic font, fixed seed, no
      degradations). Drift in the rendering pipeline — refactor,
      font upgrade, RNG-state shift — flips the digest and fails CI
      (`33df4d2`). To intentionally update the pin, run with
      `PD_OCR_SYNTH_REGEN_VISUAL_DIGESTS=1` (prints the new sha256
      alongside the failure message), eyeball-verify the change is
      intentional, paste the digest into `_PINS`, and commit with
      the reason. The Pillow version pin is documented inline since
      sha-digest stability across Pillow majors is not guaranteed
      (`a087054`).
- [x] Degraded variant: a 5th canonical recipe pipes the `lines`
      render through one certain (`probability: 1.0`) `skew` stage
      (`angle_deg: 2.0`, `fill: background`) and pins the
      post-degradation PNG sha256. This locks the geometry-stage
      dispatch path in `apply_degradation`, PIL's `Image.rotate`
      BILINEAR resampler, and the `_resolve_fill('background', ...)`
      call site — none of which the four undegraded pins exercise.
      Same self-bootstrap regen path via
      `PD_OCR_SYNTH_REGEN_VISUAL_DIGESTS=1`.

### Real bug fixes shipped under M10

- [x] `_parse_audit_timestamp` normalizes non-UTC offsets before lex
      comparing against `--since` / `--until` bounds. Before the
      fix, an audit entry written with a `+05:00` timestamp would
      sort lexicographically against a `--since` written in `+00:00`
      and select the wrong window. Lock test:
      `tests/test_audit_filter.py::test_audit_filter_handles_non_utc_timestamps`
      (`27f4bd1`).
- [x] FD-leak regression test for the audit JSONL writer locks the
      open-and-close-on-write contract so the audit log doesn't
      accumulate dangling FDs across `run_recipe` invocations
      (`a087054`).

## Quality-of-life follow-ups — future chunks

Captured here as candidates; promote any of them into a real
milestone if they start mattering. None of these block M10 from
shipping or M11 from starting.

- **Recipe linter — additional codes.** Corpus-language detection
  (modern-spelling heuristic for a Gaelic / early-modern recipe),
  per-stage option sanity checks, cross-stage interaction warnings
  (e.g. `binarize` after `noise`).
- **Visual regression — golden end-to-end pin.** The degraded-sample
  pin now ships under "Visual regression tests" above (locks one
  certain `skew` stage on a `lines` render). Still open: a "golden"
  pinned digest for `run_recipe` end-to-end output (manifest + first
  sample) so the publish-ready artifact contract is locked.
- **Audit — global aggregate cache.** A
  `~/.cache/pd-ocr-synth/audit.jsonl` aggregate so cross-recipe runs
  share one timeline. Today the audit log is per-output-dir;
  cross-output-dir queries require pointing the `audit` subcommand
  at each location separately.
- **Per-recipe `Makefile.local`.** Expose recipe-specific dev targets
  (e.g., `make gaelic-publish`) without bloating the main Makefile.

## Future work — beyond M10

These items remain genuinely open and are larger than the polish
chunks that landed in M10. None block M11 (preview UI) from
starting; each is a candidate for its own milestone if and when it
gains priority.

### Additional recipes

Each is its own scoped piece of work; pick whichever has highest
training value at the time. Each recipe is one milestone-equivalent
of work, mostly in:

1. Sourcing free fonts with appropriate licenses.
2. Identifying public-domain text corpora.
3. Writing the recipe-specific transforms (e.g., u/v swap rules).

- [ ] **Fraktur** — German blackletter; long-s, ß, capital
      ligatures. Likely the highest-leverage next recipe (Fraktur
      OCR is a known weak spot in off-the-shelf models). Needs a
      font fetch contract (license-clean Fraktur fonts) and a new
      spec doc under `docs/specs/recipes/`.
- [ ] **Early-modern English** — long-s + medial ligatures (ct, st),
      u/v and i/j swap, italic catchwords. Uses CELT-equivalent text
      from EEBO-TCP if available.
- [ ] **Greek polytonic** — accents and breathings; mostly a
      font / shaping challenge.
- [ ] **Cyrillic Old Slavonic** — titlos, abbreviations, archaic
      forms.
- [ ] **Math notation** — opens up STEM corpora.

### Cloud rendering

Per the workspace `newarch.md`, GPU/cloud render targets are
documented (Modal, Celery, AWS Batch). For pd-ocr-synth this means:

- [ ] Refactor render orchestration to be worker-pluggable.
- [ ] Add a `--workers cloud:modal` flag.
- [ ] Cache at-rest in object storage instead of local disk.

This is only worth doing once render time becomes a bottleneck —
50k word crops on CPU is fast enough for now.

### Real-time stats during render

- [ ] Render progress today reports on completion (rendered /
      skipped counts in the audit entry). Streaming per-sample
      progress through a callback / TUI surface — words/second,
      ETA, current font + degradation pipeline being applied —
      would help when iterating on a slow recipe. Likely lands as
      a `--progress {none,bar,json}` flag on `render` rather than
      its own subcommand.
