"""Render layer — turns text + recipe knobs into images + ground truth.

Public surface (M05 first cut):

- ``RenderContext`` — opened font cache + per-run RNG.
- ``RenderedSample`` — image + ground-truth metadata produced per call.
- ``render_word_crop`` — render one ``word_crops``-mode sample.
- ``sample_value`` — draw a value from a recipe scalar / range /
  weighted-choice field.

Lines / paragraphs / pages renderers and the dataset loop arrive in
later commits this milestone.
"""

from __future__ import annotations

from pd_ocr_synth.render.context import RenderContext, branched_seed
from pd_ocr_synth.render.sample import GlyphRun, RenderedSample
from pd_ocr_synth.render.sampling import sample_color, sample_value, weighted_choice
from pd_ocr_synth.render.word_crop import (
    MissingGlyphError,
    RenderError,
    render_word_crop,
)

__all__ = [
    "GlyphRun",
    "MissingGlyphError",
    "RenderContext",
    "RenderError",
    "RenderedSample",
    "branched_seed",
    "render_word_crop",
    "sample_color",
    "sample_value",
    "weighted_choice",
]
