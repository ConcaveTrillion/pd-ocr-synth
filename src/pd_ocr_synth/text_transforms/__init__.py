"""Text transforms — deterministic, ordered, applied pre-tokenization.

A transform is a callable ``(text: str, options: dict, rng: Random)
-> str``. The recipe lists transforms in order; each sees the output
of the previous one. Output is then tokenized by the layout pass
(M05+).

Public surface (M04):

- ``apply_pipeline(text, steps, *, seed)`` — run a list of transforms
  end-to-end with deterministic per-step RNGs.
- ``Registry`` + ``default_registry()`` — name → callable lookup.
- The built-in transforms documented in
  ``docs/specs/05-text-transforms.md``: ``normalize_whitespace``,
  ``lowercase``, ``uppercase``, ``strip_punctuation``, ``nfc`` /
  ``nfd`` / ``nfkc`` / ``nfkd``, ``regex_replace``, ``keep_only``,
  ``min_token_length``, ``max_token_length``,
  ``apply_lenition_dots`` / ``seimhiu_to_dot`` / ``dot_to_seimhiu``,
  ``tironian_et``, ``long_s_medial``.

Antique-conventions transforms (``u_v_swap``, ``i_j_swap``,
``ct_st_ligature_marker``) and the ``python:`` inline loader are
intentionally not in this commit — neither blocks the bundled gaelic
recipe.
"""

from __future__ import annotations

from pd_ocr_synth.text_transforms.builtins import register_builtins
from pd_ocr_synth.text_transforms.pipeline import PipelineStep, apply_pipeline
from pd_ocr_synth.text_transforms.registry import (
    Registry,
    Transform,
    UnknownTransformError,
    default_registry,
)

__all__ = [
    "PipelineStep",
    "Registry",
    "Transform",
    "UnknownTransformError",
    "apply_pipeline",
    "default_registry",
    "register_builtins",
]
