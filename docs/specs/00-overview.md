# 00 — Overview

## Purpose

Generate labeled synthetic images for OCR training of historical and specialty
typography. Output drops directly into a `pd-ocr-trainer` profile.

## Goals

1. **Recipe-driven.** A YAML file fully describes a synthetic dataset:
   text source, transforms, fonts, layout, degradation, output target.
2. **Web-aware corpora.** Recipes can pull text from URLs, Wikisource, the
   Internet Archive, or Hugging Face datasets, with on-disk caching.
3. **CLI-first.** `pd-ocr-synth render <recipe.yaml>` produces a dataset.
   Other subcommands handle scaffolding, validation, preview, and corpus
   fetching.
4. **Pluggable seams.** Corpus provider, text transform, and degradation
   stages are extension points so new scripts can be added without forking.
5. **Concrete first.** Gaelic is the first recipe; abstractions are
   extracted only when a second recipe (Fraktur, blackletter, etc.) makes
   the seams obvious.

## Non-goals

- **Not a model trainer.** That's `pd-ocr-trainer`.
- **Not a labeler / corrector.** That's `pd-ocr-labeler`.
- **Not a full layout simulator.** Page-level synthesis is a stretch goal;
  the v1 emphasis is recognition data (word/line crops) plus simple
  paragraph layouts for detection.
- **Not a font ingestion tool.** Fonts must be present on disk and licensed
  by the user. The recipe references them.

## Workspace integration

| Direction | Touchpoint |
|-----------|-----------|
| Inbound (text) | Local files, URLs, Wikisource, CELT, HF datasets, Internet Archive |
| Inbound (fonts) | User-provided font files referenced by absolute or recipe-relative paths |
| Outbound (local) | A `pd-ocr-trainer` profile directory (recognition or detection layout) |
| Outbound (HF) | Optional Hugging Face dataset repo via `pd-ocr-synth publish` (see [10 — Publishing](10-publishing.md) and the workspace-level [`DATASETS.md`](../../../DATASETS.md)) |

The outbound contract is defined by `pd-ocr-trainer`'s `dataset_store.py`. See
[08 — Output format](08-output-format.md) for the layout this project must
match.

## First recipe: Gaelic

`recipes/gaelic.yaml` targets Cló Gaelach (insular Irish typography) plus the
orthographic features of pre-reform Irish text:

- Punctum delens (`ḃ ċ ḋ ḟ ġ ṁ ṗ ṡ ṫ`) — replaces modern `bh ch dh fh gh
  mh ph sh th`
- Tironian et (`⁊`) for "agus" / "and"
- Long s (`ſ`) in word-medial position
- Insular letterforms via Cló Gaelach fonts (Bunchló, Seanchló, Duibhlinn)

## Future recipes (sketches, not committed)

- **Fraktur** — German blackletter, long-s rules, ß handling
- **Early-modern English** — long-s, ligatures (ct, st), interchange of u/v
  and i/j, italic catchwords
- **Greek** — polytonic accents, breathings
- **Cyrillic Old Slavonic** — titlos, abbreviations, archaic letterforms

Each becomes a YAML recipe plus optionally a small Python module of
script-specific transforms.

## Open questions

These are flagged here so the spec doesn't pretend they're settled.

1. **Font licensing — resolved.** This is a non-commercial project, so
   all four font families used by `recipes/gaelic.yaml` (Gaelchló,
   Gadelica, Gaedhilge, Cló Tuamach) are free to *use*. Gaelchló forbids
   redistribution and the other non-OFL families have unclear
   redistribution terms, so none are bundled in the repo. See
   [`fonts-gaelic.md`](fonts-gaelic.md) for download instructions. The
   OFL-licensed Gaedhilge is the candidate for a future CI/test fallback.
2. **Scope of the first recipe.** Pure Irish-language text vs. mixed-script
   pages (English grammar of Irish, dictionaries) is a layout question that
   may justify a second recipe rather than complicating the first.
3. **Detection vs. recognition first.** Recognition word-crops are the
   easier and higher-volume path; detection requires plausible page layouts.
   v1 focuses on recognition.
4. **Glyph-level ground truth.** For ligatures and insular forms, the
   character a reader perceives may not match the codepoint string. We
   record the codepoint string as ground truth and let the trainer's vocab
   handle the rest; this matches `pd-ocr-trainer`'s current behavior.
