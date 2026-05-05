"""Degradation pipeline — turns clean rendered samples into realistic ones.

Public surface (M06 first cut):

- ``Stage`` / ``GeometryStage`` protocols — the contract every stage
  honors. Pixel-only stages take ``(image, options, rng) -> image``;
  geometry-aware stages take ``(sample, options, rng) -> sample`` so
  they can update bbox metadata too.
- ``apply_degradation`` — walk the ordered stage list, draw a
  per-sample probability per stage, dispatch to the registered
  handler. Returns a fresh ``RenderedSample``.
- ``REGISTRY`` — kind → handler mapping. Closed-over for M06; the
  M09 plugin extension point lives in spec 09.

Determinism: every stage draws from the ``RenderContext.rng`` already
branched per sample by ``RenderContext.reseed_for_sample(index)``.
The pipeline does not create new RNGs, so output bytes remain a
function of (recipe, seed, sample_index).
"""

from __future__ import annotations

from pd_ocr_synth.degradation.pipeline import (
    REGISTRY,
    DegradationError,
    GeometryStage,
    Stage,
    apply_degradation,
)

__all__ = [
    "REGISTRY",
    "DegradationError",
    "GeometryStage",
    "Stage",
    "apply_degradation",
]
