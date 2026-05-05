# M04 — Text transforms

**Goal:** every transform documented in spec 05 works, is deterministic
under a fixed seed, and is independently testable. The `python:` inline
extension form also works.

Spec: [`05-text-transforms.md`](../specs/05-text-transforms.md).

## Deliverables

### Transform interface

- [ ] `Transform` protocol: `(text: str, options: dict, rng: Random) -> str`.
- [ ] Registry with built-ins and `python:` inline loader.
- [ ] Recipe loader resolves transform names → callables at load time.

### Generic built-ins

- [ ] `normalize_whitespace`
- [ ] `lowercase`, `uppercase`
- [ ] `strip_punctuation`
- [ ] `nfc`, `nfd`, `nfkc`, `nfkd`
- [ ] `regex_replace` (pattern, replacement, flags)
- [ ] `keep_only` (chars allowlist)
- [ ] `min_token_length`, `max_token_length` (corpus filters)

### Gaelic / pre-reform Irish built-ins

- [ ] `apply_lenition_dots` with `mode: aggressive | conservative`
      and a probability knob.
- [ ] `tironian_et` with `replace_words` list and probability.
- [ ] `long_s_medial` with probability.
- [ ] `seimhiu_to_dot` / `dot_to_seimhiu` (bidirectional alias of the
      lenition transform).

### Antique-conventions built-ins (deferred-OK if scope is tight)

- [ ] `u_v_swap` / `i_j_swap`
- [ ] `ct_st_ligature_marker` (no visible character change; sets
      internal markers consumed by the renderer in M05).

### `python:` inline loader

- [ ] Load module from path relative to recipe directory.
- [ ] Validate the callable signature matches the protocol.
- [ ] Sandboxing: refuse to import modules outside `recipe_dir` (mild
      hygiene; not a security boundary).

### Tests

- [ ] One round-trip test per transform with hand-picked input/output
      pairs.
- [ ] Determinism: same seed → identical output across runs.
- [ ] `apply_lenition_dots` aggressive vs conservative on a known
      mixed-language sample.
- [ ] `long_s_medial` never modifies word-final or `ss`/`sh` clusters.
- [ ] `tironian_et` honors case-sensitivity and word boundaries.
- [ ] Inline `python:` loader: load a fixture, run the transform, drop
      the loaded module from `sys.modules`.

## Validation criteria

A throwaway pipeline:

```python
from pd_ocr_synth.text_transforms import pipeline
out = pipeline(
    "agus do ḃí an fear bocht...",
    [{"name": "tironian_et", "options": {"probability": 1.0}},
     {"name": "long_s_medial", "options": {"probability": 1.0}}],
    seed=0,
)
assert "⁊" in out
assert "ſ" in out  # if any medial s exists
```

The Gaelic recipe loads with all transforms registered; running
`pd-ocr-synth describe gaelic` after M04 prints the post-transform
token count.

## Out of scope

- Tokenization (still in M05 with layout).
- Glyph-level handling of dotted consonants — that's a font/render
  concern (M05), not a transform concern.

## Risks / open items

- **Lenition rule completeness.** Pre-reform Irish has edge cases the
  digraph rules don't capture (e.g., `bhf` for eclipsed `f`). Cover
  the 95% case in v1; carve-outs go in `keep_only` or a
  recipe-specific transform.
- **Probability semantics.** Per-token vs per-occurrence? Match the
  spec — per-match for codepoint-level transforms, per-token for
  word-level transforms. Document in tests.
