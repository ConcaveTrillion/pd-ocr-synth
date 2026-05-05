# M06 — Degradation pipeline

**Goal:** apply realistic dirt to clean rendered samples. Word-crop
samples are sufficient for M06; bbox-aware geometric stages can be
M09 if the recognition path doesn't need them yet.

Spec: [`07-degradation.md`](../specs/07-degradation.md).

## Deliverables

### Stage interface

- [x] `Stage` protocol: `(image, options, rng) -> image` for pixel-only
      stages.
- [x] Extended `Stage` for geometry-aware stages: `(sample, options,
      rng) -> sample` where `sample` carries the image plus bbox
      metadata. (Word-crops have only one bbox; lines/pages have many.)
- [x] Pipeline runner: walk the ordered list, draw probability per
      sample, apply or skip.

### Built-in stages (M06 minimum for recognition)

- [x] `skew` — affine rotation; updates bbox.
- [x] `blur` — gaussian / motion / defocus.
- [x] `noise` — gaussian / salt_pepper / poisson / speckle.
- [x] `brightness`, `contrast`, `gamma`.
- [x] `ink_bleed` (dilate), `ink_thin` (erode).
- [x] `paper_texture` (multiply / overlay / screen / hard_light).
- [x] `foxing`.
- [x] `noise` salt-pepper.
- [x] `jpeg`, `webp`.
- [x] `grayscale`.

### Built-ins that can slip to M09

- [ ] `perspective` (4-point warp + bbox update).
- [ ] `scale`.
- [ ] `bleed_through`.
- [ ] `scratches`.
- [ ] `fold_line`.
- [ ] `binarize`.

### Composition presets

- [x] `degradation_presets` block in the recipe expands inline at load
      time (per spec 07).
- [x] Ship two named presets: `light_book_scan`, `heavy_19c_print`.
      Reference them from `recipes/gaelic.yaml`.

### Tests

- [x] One per stage: input shape preserved (or transformed correctly
      for geometric), determinism under seed.
- [x] Pipeline composition: stages run in order, probabilities honored
      across many seeds.
- [x] Bbox round-trip on `skew`: rendered text remains inside the
      reported bbox after rotation.

## Validation criteria

```bash
pd-ocr-synth preview gaelic --count 200 --output /tmp/preview
```

Looking at the resulting samples:

- Mix of clean, lightly-degraded, and heavily-degraded crops.
- Paper texture visible on ~50% of samples.
- JPEG artifacts evident at zoom on ~40%.
- No samples with text obliterated past readability.
- A grid view shows visual variety, not 200 near-identical images.

## Out of scope

- Detection-mode bbox tracking through every geometric stage (M09 if
  not done here).
- GPU acceleration of degradation kernels (out of v1 scope).

## Risks / open items

- **Texture asset sourcing.** The recipe's `paper_texture.directory`
  needs real textures. Ship a small set of CC0 paper textures with
  the repo (acceptable to bundle — non-font, no IP issue) so the
  preview works out of the box. Source: archive.org or
  cc0textures.com equivalents.
- **`opencv-python` size.** Adds ~80MB to the venv. Acceptable but
  confirm peer projects already pull it; otherwise consider Pillow-only
  for v1.
- **Order matters.** Document the canonical order in the spec (already
  done): geometric → optical → paper → noise → JPEG. Tests should
  catch reorderings that produce wrong-looking output.
