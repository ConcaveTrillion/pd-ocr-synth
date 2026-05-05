# 03 — Tutorial: writing a recipe

This walks through `recipes/gaelic.yaml` section by section so you can build
your own recipe by analogy.

## 0. Decide what you're targeting

Before opening an editor, answer:

1. **What output does the trainer need?** Recognition word-crops, or
   detection page layouts? (Start with recognition — it's higher volume and
   easier to get right.)
2. **Where will the text come from?** Public domain books? Wikisource? A
   scraped HTML page? A local corpus you already have?
3. **What orthographic transforms apply?** Modern spelling vs. historical?
   Any character-level substitutions (long-s, dotted lenition)?
4. **What fonts do you have?** And are they licensed for your use?
5. **What does real-world degradation look like?** Old book scans differ
   from photocopies differ from modern phone captures.

## 1. Scaffold

```bash
pd-ocr-synth init gaelic
```

This creates:

```
recipes/gaelic/
├── recipe.yaml       # the recipe (heavily commented template)
├── README.md         # space for recipe-specific notes
├── corpora/          # empty; place local text files here
├── fonts/            # empty; place font files here
└── textures/         # empty; place degradation textures here
```

You may also keep recipes as a single file (`recipes/gaelic.yaml`) — the
folder layout is just a convention for recipes with bundled assets.

## 2. Identify the recipe and its output

```yaml
schema_version: 1
name: gaelic
description: Cló Gaelach + pre-reform Irish orthography
seed: 42

output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ${PD_ML_MODELS}/ml-recognition/gaelic/recognition
  count: 50000
```

`name` should match the trainer profile you want to feed. `count` is the
total number of samples; for word-crops, 25k–100k is a reasonable starting
range.

## 3. Pull in text

Three corpora chained together:

```yaml
corpus:
  - type: web
    url: https://celt.ucc.ie/published/G100001A.html
    parser: html-text
    cache: true

  - type: wikisource
    language: mul
    titles:
      - "Séadna"
      - "Mo Sgéal Féin"
    cache: true

  - type: local
    path: ./corpora/seed-words.txt
```

The first two pull from the web (cached on disk after first fetch). The
third is a hand-curated word list you bundle with the recipe — useful for
guaranteeing coverage of rare codepoints.

See [04 — Corpus providers](04-corpus-providers.md) for the full set,
including `hf_dataset` and `internet_archive`.

## 4. Apply text transforms

For Gaelic, the corpus may already have dotted consonants — but if it
doesn't, this transform applies them:

```yaml
text_transforms:
  - normalize_whitespace
  - apply_lenition_dots:
      mode: aggressive          # 'conservative' only converts unambiguous cases
  - tironian_et:
      probability: 0.7
  - long_s_medial:
      probability: 0.85
```

Order matters. `normalize_whitespace` first, codepoint substitutions after.
See [05 — Text transforms](05-text-transforms.md).

## 5. Add fonts

```yaml
fonts:
  - path: ./fonts/Bunchlo-GC.otf
    weight: 0.4
  - path: ./fonts/Seanchlo-GC.otf
    weight: 0.4
  - path: ./fonts/Duibhlinn.ttf
    weight: 0.2
```

Run `pd-ocr-synth validate gaelic` — the validator opens each font and
reports which codepoints from your transformed corpus are missing. If a
font lacks `ḃ ċ ḋ`, it can't render Gaelic — drop it or accept the
fallback (rendering will skip samples that need missing glyphs).

## 6. Configure rendering

```yaml
rendering:
  shaping_engine: harfbuzz
  font_size_pt: { min: 12, max: 22 }
  dpi: { min: 200, max: 400 }
  ink_color:
    r: { min: 5, max: 50 }
    g: { min: 5, max: 50 }
    b: { min: 5, max: 50 }
  background_color:
    r: { min: 215, max: 250 }
    g: { min: 210, max: 245 }
    b: { min: 195, max: 235 }
```

Cream/yellow backgrounds and dark-but-not-black ink approximate aged paper.
The wider the range, the more the trained model generalizes.

## 7. Pick a layout

```yaml
layout:
  mode: word_crops
  padding_px: { min: 4, max: 14 }
```

For recognition, `word_crops` is the right starting point. Each rendered
sample is a single word; the ground truth is the word's text.

## 8. Add degradation

```yaml
degradation:
  - kind: skew
    probability: 0.6
    angle_deg: { min: -2, max: 2 }
  - kind: blur
    probability: 0.5
    sigma: { min: 0.0, max: 1.2 }
  - kind: ink_bleed
    probability: 0.3
    iterations: { min: 1, max: 2 }
  - kind: paper_texture
    probability: 0.5
    directory: ./textures/aged-paper/
    blend: multiply
    opacity: { min: 0.2, max: 0.6 }
  - kind: noise
    probability: 0.5
    kind_inner: gaussian
    stddev: { min: 0, max: 8 }
  - kind: jpeg
    probability: 0.4
    quality: { min: 60, max: 95 }
```

Order matters: skew before texture (so the texture isn't skewed too), JPEG
last (it should reflect the final compression).

## 9. Validate and preview

```bash
pd-ocr-synth validate gaelic
pd-ocr-synth fetch gaelic           # one-time corpus download
pd-ocr-synth preview gaelic -c 200  # eyeball the output
```

Open the preview directory. Look for:

- Glyphs that didn't render (boxes / `.notdef`)
- Degradations that destroy legibility
- Backgrounds that look unnaturally clean
- Crops that are too tight or too loose

Iterate on the recipe until preview output looks like the books you want
to OCR.

## 10. Render the full dataset

```bash
pd-ocr-synth render gaelic
```

Output lands in the configured `destination`, in
`pd-ocr-trainer/v1` format. A `manifest.jsonl` records provenance for each
sample (source corpus, font, rendering params, applied degradations) — see
[08 — Output format](08-output-format.md).

## Common pitfalls

- **Corpus too small** → repetitive crops, poor generalization. Aim for at
  least 5–10× the sample count in unique tokens.
- **Single font** → model overfits to that font's glyph forms. Use 2+.
- **No degradation** → model fails on real scans. Always include some.
- **Too much degradation** → model never learns the clean signal.
  Probabilities < 1.0 ensure some samples stay clean.
- **Forgetting to seed** → re-runs aren't reproducible. Set `seed`.
