# 09 — Extending

Three extension points: corpus providers, text transforms, and degradation
stages. Each is a small Python interface registered via entry points or
referenced inline by file path.

## Inline (per-recipe) extensions

For one-off code that lives next to the recipe, reference a Python module
by path. Lookups are relative to the recipe directory.

```yaml
text_transforms:
  - python:
      module: ./transforms.py
      callable: my_transform
      options:
        threshold: 0.5

corpus:
  - type: python
    module: ./providers.py
    callable: pull_my_corpus
    options:
      tag: irish

degradation:
  - kind: python
    module: ./degradations.py
    callable: my_stage
    probability: 0.5
    options: {}
```

Inline extensions are isolated: they don't pollute the global registry
and only apply within the recipe that names them.

## Package extensions (entry points)

For reusable extensions, register via `pyproject.toml`:

```toml
[project.entry-points."pd_ocr_synth.corpus_providers"]
my_corpus = "my_pkg.providers:MyCorpus"

[project.entry-points."pd_ocr_synth.text_transforms"]
my_transform = "my_pkg.transforms:my_transform"

[project.entry-points."pd_ocr_synth.degradation_stages"]
my_stage = "my_pkg.degradation:my_stage"
```

The CLI auto-discovers anything installed in the active Python env.

## Interface — corpus provider

```python
from typing import Iterable
from pd_ocr_synth.corpus import ProviderContext

class MyCorpus:
    type_name = "my_corpus"

    def fetch(self, ctx: ProviderContext, options: dict) -> Iterable[str]:
        """
        Yield UTF-8 strings (one logical chunk per yield — typically a
        document or a page). The framework concatenates and tokenizes
        downstream.
        """
        ...

    def cache_key(self, options: dict) -> str:
        """Stable key for the on-disk cache. Distinct configs → distinct keys."""
        ...
```

`ProviderContext` exposes:

| Attr | Type | Purpose |
|------|------|---------|
| `recipe_dir` | `Path` | For resolving relative paths |
| `cache_dir` | `Path` | Where to write cache files |
| `http` | `httpx.Client` | Pre-configured client (UA, retries, rate limit) |
| `offline` | `bool` | If true, raise instead of fetching |
| `logger` | `Logger` | Structured logger |

## Interface — text transform

```python
from random import Random

def my_transform(text: str, options: dict, rng: Random) -> str:
    """Pure function: text in, text out. Use rng for any randomness."""
    ...
```

Transforms must be deterministic given the rng. They run on the full
concatenated corpus (post-fetch, pre-tokenization).

## Interface — degradation stage

```python
from PIL import Image
from random import Random

def my_stage(image: Image.Image, options: dict, rng: Random) -> Image.Image:
    """Image in, image out. Use rng for any randomness."""
    ...
```

Stages may also accept and return ground-truth metadata when geometry
changes. The richer signature:

```python
from pd_ocr_synth.types import RenderedSample

def my_stage(sample: RenderedSample, options: dict, rng: Random) -> RenderedSample:
    """For stages that affect geometry (skew, perspective, scale)."""
    ...
```

`RenderedSample` carries the image plus all bounding boxes; geometric
stages must update boxes consistently. Pixel-only stages (blur, noise,
JPEG) use the simpler `Image -> Image` form.

## Validation hooks

A custom extension can opt into recipe-validation:

```python
def validate(options: dict, ctx: ValidationContext) -> list[ValidationIssue]:
    """Return zero or more issues. Empty list = OK."""
```

This runs during `pd-ocr-synth validate` so the CLI surfaces problems
before render time.

## Versioning the registry

Each extension declares the schema version it expects:

```python
class MyCorpus:
    type_name = "my_corpus"
    schema_version = 1
```

If a recipe targets `schema_version: 1` but the loaded extension declares
`2`, validation fails with a clear migration prompt.

## Testing your extension

The repo ships a `pd_ocr_synth.testing` module with helpers:

- `make_recipe(...)` — minimal recipe builder
- `dummy_rng()` — deterministic RNG
- `tiny_corpus(text)` — in-memory provider for tests
- `tmp_destination()` — pytest fixture for isolated output dirs

A custom extension's test suite should cover:

1. The happy path (input produces expected output).
2. Determinism under a fixed seed.
3. Validation surfacing user errors with helpful messages.
4. Behavior when `offline=True` (for providers).

## Boundaries (what extensions can't do)

- Mutate the recipe object after load (it's frozen post-validation).
- Read or write files outside `cache_dir`, `recipe_dir`, or
  `output.destination`.
- Make network calls when `ctx.offline` is true.

These constraints exist so `validate` and `--dry-run` give honest
signals.
