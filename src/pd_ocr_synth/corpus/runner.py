"""Drive corpus providers from a loaded ``Recipe``.

Walks ``recipe.corpus`` in order, dispatches each entry to the right
provider via ``default_registry()``, applies any provider-level
filter, and yields one ``ProviderRunResult`` per entry. Used by:

- ``pd-ocr-synth fetch`` to warm the cache up front.
- ``pd-ocr-synth describe`` (in M03+) to compute corpus statistics.
- The render pipeline (M05+) to gather text before tokenization.

``collect_corpus_text`` is the higher-level convenience: it runs the
providers, joins per-entry text, and pipes the result through
``recipe.text_transforms`` (M04) using the recipe's seed.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pd_ocr_synth.corpus.context import ProviderContext
from pd_ocr_synth.corpus.filters import apply_filter
from pd_ocr_synth.corpus.registry import default_registry

if TYPE_CHECKING:
    from pd_ocr_synth.recipe import Recipe


@dataclass(frozen=True, slots=True)
class ProviderRunResult:
    """One corpus entry's outcome."""

    index: int
    type_name: str
    cache_key: str
    text: str
    was_cached: bool
    elapsed_s: float


def run_providers(
    recipe: Recipe,
    *,
    ctx: ProviderContext,
    apply_filters: bool = True,
) -> Iterator[ProviderRunResult]:
    """Iterate ``recipe.corpus`` and run each entry's provider.

    ``apply_filters=True`` runs the per-entry filter after fetch; pass
    ``False`` for tests or callers that want raw provider output.
    """

    registry = default_registry()
    for index, entry in enumerate(recipe.corpus):
        options = _options_for(entry)
        provider = registry.get(entry.type)  # type: ignore[arg-type]
        cache_key = provider.cache_key(options)
        was_cached = ctx.cache.has(provider.type_name, cache_key)
        started = time.monotonic()
        chunks = list(provider.fetch(ctx, options))
        elapsed = time.monotonic() - started
        text = "\n".join(chunks)
        if apply_filters:
            text = apply_filter(text, options.get("filter"))
        yield ProviderRunResult(
            index=index,
            type_name=provider.type_name,
            cache_key=cache_key,
            text=text,
            was_cached=was_cached,
            elapsed_s=elapsed,
        )


def _options_for(entry: object) -> dict:
    """Convert a typed corpus entry to a plain dict for the provider.

    Pydantic v2's ``model_dump(mode='python')`` keeps Path objects as
    Path (which providers expect) and unwraps any nested submodels.
    """

    return entry.model_dump(mode="python")  # type: ignore[attr-defined]


def collect_corpus_text(
    recipe: Recipe,
    *,
    ctx: ProviderContext,
) -> str:
    """Run providers, join their text, then apply ``recipe.text_transforms``.

    Per-provider filters run inside ``run_providers``. Entries are
    joined with a blank-line separator so paragraph-aware transforms
    see distinct provider boundaries. The recipe's ``seed`` drives
    the text-transform RNG.

    This is the entry point M05 (render) will call to materialize the
    full pre-tokenization corpus.
    """

    # Imported here so the corpus package stays usable in environments
    # that haven't installed the text_transforms layer (e.g. tests of
    # individual providers).
    from pd_ocr_synth.text_transforms import PipelineStep, apply_pipeline

    chunks = [r.text for r in run_providers(recipe, ctx=ctx)]
    text = "\n\n".join(c for c in chunks if c)

    steps = [PipelineStep(name=t.name, options=dict(t.options)) for t in recipe.text_transforms]
    if not steps:
        return text
    return apply_pipeline(text, steps, seed=recipe.seed)
