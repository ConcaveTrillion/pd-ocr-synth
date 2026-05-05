# M10 — Stretch

The engine works after M00–M09. M10 is open-ended polish, additional
recipes, and cloud rendering. The preview UI moved out of stretch and
is now [M11](11-preview-ui.md) with its own
[spec](../specs/11-preview-ui.md).

## Additional recipes

Each is its own scoped piece of work; pick whichever has highest
training value at the time.

- **Fraktur** — German blackletter; long-s, ß, capital ligatures.
- **Early-modern English** — long-s + medial ligatures (ct, st), u/v
  and i/j swap, italic catchwords. Uses CELT-equivalent text from
  EEBO-TCP if available.
- **Greek polytonic** — accents and breathings; mostly a font/shaping
  challenge.
- **Cyrillic Old Slavonic** — titlos, abbreviations, archaic forms.
- **Math notation** — opens up STEM corpora.

Each recipe is one milestone-equivalent of work, mostly in:

1. Sourcing free fonts with appropriate licenses.
2. Identifying public-domain text corpora.
3. Writing the recipe-specific transforms (e.g., u/v swap rules).

## Cloud rendering

Per the workspace `newarch.md`, GPU/cloud render targets are
documented (Modal, Celery, AWS Batch). For pd-ocr-synth this means:

- Refactor the render orchestration to be worker-pluggable.
- Add a `--workers cloud:modal` flag.
- Cache at-rest in object storage instead of local disk.

This is only worth doing once render time becomes a bottleneck —
50k word crops on CPU is fast enough for now.

## Quality-of-life follow-ups

Captured here as candidates; promote any of them into a real milestone
if they start mattering:

- **Recipe linter.** Beyond schema validation, flag suspicious
  patterns: degradation always at probability 1.0, single font, no
  text transforms when the corpus is in modern spelling.
- **Visual regression tests.** A "golden" sample set for the Gaelic
  recipe; CI re-renders and compares to detect accidental changes
  in the rendering or degradation pipeline.
- **Per-recipe `Makefile.local`.** Expose recipe-specific dev targets
  (e.g., `make gaelic-publish`) without bloating the main Makefile.
- **Audit log.** A record of every render run + its content SHA for
  the project journal; useful when a model trained from a specific
  dataset version is later debugged.
