# pd-ocr-synth

Synthetic OCR training-data generator. Produces labeled image+text pairs for
historical and specialty typography (first target: Cló Gaelach / early Irish
typography), in the format consumed by `pd-ocr-trainer` profiles.

**Status:** spec-only. No implementation yet. The `docs/specs/` set defines the
target system; nothing in `src/` exists.

## Why

Real labeled data for historical scripts is scarce. The standard playbook for
training an OCR model on uncommon typography is:

1. Generate a large synthetic corpus (this project)
2. Pretrain on it
3. Fine-tune on a small set of real labeled pages from `pd-ocr-labeler`

This project covers step 1. It is recipe-driven so adding a new script
(Fraktur, blackletter, early-modern English with long-s, Greek, etc.) is a
matter of writing a YAML file plus optional custom transforms.

## How it fits the workspace

```
[ pd-ocr-synth ]  →  recognition crops + detection pages
        │
        ▼
[ pd-ocr-trainer ]  reads new profile (e.g. `gaelic`)
        │
        ▼
[ pd-ocr-cli ]  uses fine-tuned model on real scans
```

The output adapter writes directly into the directory layout
`pd-ocr-trainer` already understands.

## Getting started

Read the specs in order:

| # | Spec | What it covers |
|---|------|----------------|
| 00 | [Overview](docs/specs/00-overview.md) | Goals, non-goals, integration |
| 01 | [CLI](docs/specs/01-cli.md) | Commands and flags |
| 02 | [Recipe format](docs/specs/02-recipe-format.md) | YAML schema reference |
| 03 | [Tutorial](docs/specs/03-tutorial-writing-a-recipe.md) | Build the Gaelic recipe end-to-end |
| 04 | [Corpus providers](docs/specs/04-corpus-providers.md) | Local, web, Wikisource, HF datasets |
| 05 | [Text transforms](docs/specs/05-text-transforms.md) | Lenition, long-s, Tironian et |
| 06 | [Rendering](docs/specs/06-rendering.md) | Fonts, shaping, layouts |
| 07 | [Degradation](docs/specs/07-degradation.md) | Augmentation pipeline |
| 08 | [Output format](docs/specs/08-output-format.md) | pd-ocr-trainer profile layout |
| 09 | [Extending](docs/specs/09-extending.md) | Adding providers / transforms / degradations |
| 10 | [Publishing](docs/specs/10-publishing.md) | Push rendered datasets to Hugging Face |
| 11 | [Preview UI](docs/specs/11-preview-ui.md) | NiceGUI for visual recipe tuning (read-only on recipes) |
| 12 | [Glyph annotations emission](docs/specs/12-glyph-annotations-emission.md) | Per-word ligature / long-s side channel |
| 13 | [Dev-local mode and deps](docs/specs/13-dev-local-mode-and-deps.md) | `upgrade-deps` must not silently revert dev-local venvs |

A full worked recipe lives at [`recipes/gaelic.yaml`](recipes/gaelic.yaml).

## Implementation roadmap

Specs describe the destination; [`docs/roadmap/`](docs/roadmap/) is the
path. Start at [00 — Bootstrap](docs/roadmap/00-bootstrap.md).
