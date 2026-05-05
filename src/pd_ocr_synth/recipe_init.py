"""Recipe scaffolding for ``pd-ocr-synth init``.

Writes a minimal-but-valid recipe directory with a commented YAML
template and a starter README. The template targets the ``word_crops``
recognition mode (the most common starting point); authors edit it
in place.
"""

from __future__ import annotations

from pathlib import Path

_RECIPE_TEMPLATE = """\
# Recipe: {name}
# See docs/specs/02-recipe-format.md for the full schema reference, or
# run `pd-ocr-synth schema` to dump the JSON Schema for editor support.

schema_version: 1
name: {name}
description: |
  TODO: one-paragraph description of the typography / orthography
  this recipe targets.
seed: 0

output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out/{name}
  count: 1000

corpus:
  - type: local
    path: ./{name}/seed-words.txt

text_transforms:
  - normalize_whitespace

fonts:
  - path: ./{name}/fonts/REPLACE-ME.otf
    weight: 1.0

rendering:
  shaping_engine: harfbuzz
  font_size_pt: {{ min: 12, max: 18 }}
  dpi: {{ min: 200, max: 400 }}
  ink_color:
    r: {{ min: 5, max: 50 }}
    g: {{ min: 5, max: 50 }}
    b: {{ min: 5, max: 50 }}
  background_color:
    r: {{ min: 220, max: 250 }}
    g: {{ min: 220, max: 250 }}
    b: {{ min: 220, max: 250 }}
  antialiasing: true

layout:
  mode: word_crops
  padding_px: {{ min: 4, max: 12 }}

degradation:
  - kind: blur
    probability: 0.5
    sigma: {{ min: 0.0, max: 1.0 }}
  - kind: jpeg
    probability: 0.4
    quality: {{ min: 70, max: 95 }}
"""

_README_TEMPLATE = """\
# {name}

Synthetic OCR training data recipe for `pd-ocr-synth`.

## Workflow

```sh
pd-ocr-synth validate {name}    # check schema + paths
pd-ocr-synth describe {name}    # print resolved config
pd-ocr-synth fetch    {name}    # cache corpora (M03+)
pd-ocr-synth render   {name}    # generate the dataset (M05+)
```

## TODO

- [ ] Fill in the description in `recipe.yaml`.
- [ ] Add corpus sources (local paths, URLs, HF datasets, Wikisource).
- [ ] Replace the placeholder font path with one or more real `.otf` /
      `.ttf` files.
- [ ] Tune rendering / degradation knobs to match the target domain.
"""

_SEED_WORDS = "the\nquick\nbrown\nfox\njumps\nover\nthe\nlazy\ndog\n"


def scaffold_recipe(*, name: str, target_dir: Path) -> list[Path]:
    """Create ``target_dir`` and write the starter recipe files.

    Returns the list of files written (absolute paths) for caller
    diagnostics. ``target_dir`` is created if missing.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    inner = target_dir / name
    inner.mkdir(exist_ok=True)
    (inner / "fonts").mkdir(exist_ok=True)

    written: list[Path] = []

    recipe_path = target_dir / "recipe.yaml"
    recipe_path.write_text(_RECIPE_TEMPLATE.format(name=name), encoding="utf-8")
    written.append(recipe_path)

    readme_path = target_dir / "README.md"
    readme_path.write_text(_README_TEMPLATE.format(name=name), encoding="utf-8")
    written.append(readme_path)

    seed_path = inner / "seed-words.txt"
    seed_path.write_text(_SEED_WORDS, encoding="utf-8")
    written.append(seed_path)

    return written
