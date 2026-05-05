# 07 — Degradation

The degradation pipeline takes a clean rendered image and produces a
realistically dirty one. It is the single most important component for
domain transfer to real scans.

```yaml
degradation:
  - kind: skew
    probability: 0.6
    angle_deg: { min: -2, max: 2 }
  - kind: blur
    probability: 0.5
    sigma: { min: 0.0, max: 1.2 }
  - kind: paper_texture
    probability: 0.5
    directory: ./textures/aged-paper/
    blend: multiply
    opacity: { min: 0.2, max: 0.6 }
  - kind: jpeg
    probability: 0.4
    quality: { min: 60, max: 95 }
```

## Common keys

| Key | Default | Meaning |
|-----|---------|---------|
| `kind` | required | Stage type (see catalog below) |
| `probability` | `1.0` | Per-sample probability of applying the stage |
| `name` | derived | Optional label recorded in the manifest |

Order matters. The list is applied top-to-bottom. JPEG and resize-down
should generally be last.

## Geometric

### `skew`
Affine rotation around the image center.
```yaml
- kind: skew
  angle_deg: { min: -3, max: 3 }
  fill: background          # background | white | black | transparent
```

### `perspective`
Random 4-point warp.
```yaml
- kind: perspective
  max_corner_offset_px: { min: 0, max: 10 }
```

### `scale`
Uniform or non-uniform resize. Useful for simulating bad scan resolution.
```yaml
- kind: scale
  factor_x: { min: 0.7, max: 1.0 }
  factor_y: { min: 0.7, max: 1.0 }
  resample: bilinear        # nearest | bilinear | bicubic | lanczos
```

## Optical

### `blur`
```yaml
- kind: blur
  filter: gaussian          # gaussian | motion | defocus
  sigma: { min: 0.0, max: 1.5 }
  motion_angle_deg: { min: -10, max: 10 }   # only for motion
  motion_length_px: { min: 0, max: 6 }
```

### `noise`
```yaml
- kind: noise
  noise_kind: gaussian      # gaussian | salt_pepper | poisson | speckle
  stddev: { min: 0, max: 8 }              # gaussian / speckle
  amount: { min: 0.0, max: 0.02 }         # salt_pepper
```

### `brightness` / `contrast`
```yaml
- kind: brightness
  factor: { min: 0.85, max: 1.15 }
- kind: contrast
  factor: { min: 0.8, max: 1.2 }
```

### `gamma`
```yaml
- kind: gamma
  gamma: { min: 0.7, max: 1.3 }
```

## Print / paper

### `ink_bleed`
Morphological dilation simulating ink that has spread.
```yaml
- kind: ink_bleed
  iterations: { min: 1, max: 2 }
  kernel_size_px: { min: 1, max: 2 }
```

### `ink_thin`
Erosion — under-inked print.
```yaml
- kind: ink_thin
  iterations: { min: 1, max: 1 }
```

### `paper_texture`
Blend a paper texture image.
```yaml
- kind: paper_texture
  directory: ./textures/aged-paper/
  blend: multiply           # multiply | overlay | screen | hard_light
  opacity: { min: 0.2, max: 0.6 }
  scale: { min: 0.5, max: 1.5 }
  rotate_deg: { min: -180, max: 180 }
```

The directory is sampled per-sample. Recommended texture sizes: at least
2× the largest output dimension.

### `foxing`
Adds reddish-brown spots characteristic of aged paper.
```yaml
- kind: foxing
  count: { min: 0, max: 5 }
  radius_px: { min: 2, max: 8 }
  color: [120, 60, 30]
  opacity: { min: 0.2, max: 0.5 }
```

### `bleed_through`
Faint mirror of text from the back of the page.
```yaml
- kind: bleed_through
  source: same_corpus       # same_corpus | text_file | rendered_other
  opacity: { min: 0.05, max: 0.2 }
  flip_horizontal: true
```

### `scratches`
Linear streaks.
```yaml
- kind: scratches
  count: { min: 0, max: 3 }
  thickness_px: { min: 1, max: 2 }
  length_px: { min: 30, max: 200 }
  color: [200, 200, 200]
```

### `fold_line`
Adds a horizontal or vertical fold shadow.
```yaml
- kind: fold_line
  axis: random              # horizontal | vertical | random
  position: { min: 0.3, max: 0.7 }    # fraction of axis
  intensity: { min: 0.05, max: 0.2 }
```

## Compression

### `jpeg`
```yaml
- kind: jpeg
  quality: { min: 50, max: 95 }
  chroma_subsampling: random        # random | 4:4:4 | 4:2:0
```

### `webp`
```yaml
- kind: webp
  quality: { min: 50, max: 95 }
```

## Color space

### `grayscale`
```yaml
- kind: grayscale
  method: luminosity        # luminosity | average | red | green | blue
```

### `binarize`
```yaml
- kind: binarize
  method: otsu              # otsu | sauvola | niblack | adaptive
  threshold: 128            # only for fixed
```

For modeling photocopier output. Heavy: only enable when the target domain
is binarized (xerox / fax).

## Composition presets

Recipes can declare named presets to keep the file short:

```yaml
degradation_presets:
  light_book_scan:
    - { kind: skew, probability: 0.4, angle_deg: { min: -1, max: 1 } }
    - { kind: blur, probability: 0.4, sigma: { min: 0.0, max: 0.8 } }
    - { kind: jpeg, probability: 0.3, quality: { min: 80, max: 95 } }
  heavy_19c_print:
    - { kind: skew, probability: 0.7, angle_deg: { min: -3, max: 3 } }
    - { kind: ink_bleed, probability: 0.5, iterations: { min: 1, max: 2 } }
    - { kind: paper_texture, probability: 0.7, directory: ./textures/aged-paper/, blend: multiply, opacity: { min: 0.3, max: 0.6 } }
    - { kind: foxing, probability: 0.3, count: { min: 0, max: 4 } }
    - { kind: noise, probability: 0.5, noise_kind: gaussian, stddev: { min: 2, max: 8 } }
    - { kind: jpeg, probability: 0.4, quality: { min: 60, max: 90 } }

degradation:
  - preset: heavy_19c_print
```

A `preset:` entry expands inline. Presets are local to one recipe.

## Custom degradation stages

Same extension contract as transforms / providers — see
[09 — Extending](09-extending.md).
