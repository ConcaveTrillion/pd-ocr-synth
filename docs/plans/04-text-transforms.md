# M04 — Text transforms (mostly complete)

**Status:** ✅ every transform the bundled gaelic recipe needs is
landed. Antique-conventions transforms and the ``python:`` inline
loader are deferred — see below.

**Goal:** every transform documented in spec 05 works, is
deterministic under a fixed seed, and is independently testable.

Spec: [`05-text-transforms.md`](../specs/05-text-transforms.md).

## Deliverables

### Transform interface

- [x] ``Transform`` protocol: ``(text: str, options: dict, rng: Random) -> str``.
- [x] ``Registry`` with eager built-ins + lazy entry-point loading
      (``pd_ocr_synth.text_transforms`` group).
- [x] ``apply_pipeline(text, steps, *, seed)`` derives a stable
      per-step RNG from the recipe seed so order changes don't
      destabilize earlier steps.

### Generic built-ins

- [x] ``normalize_whitespace`` (preserves paragraph breaks).
- [x] ``lowercase`` / ``uppercase``.
- [x] ``strip_punctuation`` (Unicode P-categories).
- [x] ``nfc`` / ``nfd`` / ``nfkc`` / ``nfkd``.
- [x] ``regex_replace`` (pattern, replacement, flags).
- [x] ``keep_only`` (chars allowlist).
- [x] ``min_token_length`` / ``max_token_length`` (whitespace-token
      filters; paragraph breaks preserved).

### Gaelic / pre-reform Irish built-ins

- [x] ``apply_lenition_dots`` with ``mode: aggressive |
      conservative`` and a probability knob. Conservative mode
      requires the next character to be a vowel.
- [x] ``tironian_et`` with ``replace_words``, ``probability``,
      ``case_sensitive``. Word-boundary aware via ``\b``.
- [x] ``long_s_medial`` with ``probability``. Skips word-final and
      ``s`` immediately before another ``s`` or ``h``. Word-initial
      lowercase ``s`` is allowed (early-modern practice). Uppercase
      ``S`` is preserved.
- [x] ``seimhiu_to_dot`` (alias of ``apply_lenition_dots``) +
      ``dot_to_seimhiu`` (reverse mapping for round-trip tests).

### Antique-conventions built-ins

- [ ] ``u_v_swap`` / ``i_j_swap`` (deferred — gaelic does not use
      them; trivial to add when a recipe demands).
- [ ] ``ct_st_ligature_marker`` (deferred — depends on the M05
      renderer recognizing markers).

### ``python:`` inline loader

- [ ] Path-relative module loader (deferred). The entry-point hook
      already covers reusable transforms; inline loading is a
      one-off-recipe convenience.

### Tests

- [x] One round-trip test per implemented transform.
- [x] Determinism: same seed → identical output across runs (covered
      for lenition probability, tironian probability, full pipeline).
- [x] ``apply_lenition_dots`` aggressive vs conservative.
- [x] ``long_s_medial`` skips word-final + ``ss``/``sh`` clusters,
      allows word-initial.
- [x] ``tironian_et`` honors case-sensitivity + word boundaries.
- [x] Pipeline integration: ``collect_corpus_text`` runs providers
      then transforms end-to-end.
- [ ] Inline ``python:`` loader test (deferred with the feature).

## Validation criteria

The throwaway pipeline from the original roadmap now works:

```python
from pd_ocr_synth.text_transforms import apply_pipeline
out = apply_pipeline(
    "agus do bhi an fear bocht...",
    [
        {"name": "tironian_et", "options": {"probability": 1.0}},
        {"name": "apply_lenition_dots", "options": {"mode": "aggressive"}},
        {"name": "long_s_medial", "options": {"probability": 1.0}},
    ],
    seed=0,
)
# → "⁊ do ḃi an fear boċt..." (with ſ wherever an eligible 's' exists)
```

The Gaelic recipe loads with all five of its transforms registered
and ``collect_corpus_text(recipe, ctx)`` returns the
post-transform text ready for tokenization.

## Closeout notes

- The ``describe`` corpus-stats extension named in spec 05 / M03
  still hangs on tokenization choices — keep that with M05 (render).
- ``conservative`` lenition mode here is a "vowel-follows" rule, not
  a full English-digraph aware rule. Recipes that mix English and
  Irish should rely on ``keep_only`` plus per-recipe regex rather
  than this transform's English-detection.
