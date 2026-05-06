"""Render layer — turns text + recipe knobs into images + ground truth.

Public surface:

- ``RenderContext`` — opened font cache + per-run RNG.
- ``RenderedSample`` — image + ground-truth metadata produced per call.
- ``render_word_crop`` — render one ``word_crops``-mode sample.
- ``sample_value`` — draw a value from a recipe scalar / range /
  weighted-choice field.
- ``run_recipe`` (M07) — full dataset loop into the
  ``pd-ocr-trainer/v1`` recognition layout.
- ``plan_recipe`` — dry-run summary of what ``run_recipe`` would do.
"""

from __future__ import annotations

from pd_ocr_synth.render.context import RenderContext, branched_seed
from pd_ocr_synth.render.line import render_line
from pd_ocr_synth.render.paragraph import render_paragraph
from pd_ocr_synth.render.run import RunPlan, RunResult, plan_recipe, run_recipe
from pd_ocr_synth.render.sample import GlyphRun, LineBox, RenderedSample, WordBox
from pd_ocr_synth.render.sampling import sample_color, sample_value, weighted_choice
from pd_ocr_synth.render.word_crop import (
    MissingGlyphError,
    RenderError,
    render_word_crop,
)
from pd_ocr_synth.render.wrap import fit_lines

__all__ = [
    "GlyphRun",
    "LineBox",
    "MissingGlyphError",
    "RenderContext",
    "RenderError",
    "RenderedSample",
    "RunPlan",
    "RunResult",
    "WordBox",
    "branched_seed",
    "fit_lines",
    "plan_recipe",
    "render_line",
    "render_paragraph",
    "render_word_crop",
    "run_recipe",
    "sample_color",
    "sample_value",
    "weighted_choice",
]
