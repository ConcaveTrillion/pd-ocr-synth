# 06 — Rendering

Rendering turns transformed text + a chosen font into a clean (un-degraded)
image plus per-glyph ground-truth metadata. Degradation runs after, on the
clean output.

## Shaping engine

Two engines are supported:

| Engine | Use when |
|--------|----------|
| `harfbuzz` (default) | Scripts with shaping rules: ligatures, contextual forms, mark positioning |
| `pillow` | Plain Latin without shaping; faster but no `liga`/`calt` support |

Cló Gaelach uses HarfBuzz. So does anything with marks (Greek polytonic,
Arabic, Devanagari, Gaelic dotted forms when stored as base + combining
mark rather than precomposed).

The HarfBuzz path uses `uharfbuzz` for shaping and `freetype-py` for
rasterization. Anti-aliasing is on by default; `subpixel: false` is a
common knob to harden against sub-pixel artifacts.

## Font selection

```yaml
fonts:
  - path: ./fonts/Bunchlo-GC.otf
    weight: 0.4
  - path: ./fonts/Seanchlo-GC.otf
    weight: 0.4
  - path: ./fonts/Duibhlinn.ttf
    weight: 0.2
    features:
      liga: true
      calt: true
```

For each sample, a font is drawn by weight. `features` enables / disables
specific OpenType features for the renderer. (Defaults: `liga: true`,
`calt: true`, others off.)

The validator inspects each font and reports:

- The set of codepoints covered
- Codepoints in the corpus that the font does **not** cover
- Whether `liga`/`calt` features are present

A sample whose token requires a missing glyph is skipped; the manifest
records the skip reason.

## Size, color, DPI

```yaml
rendering:
  font_size_pt: { min: 10, max: 22 }
  dpi: { min: 200, max: 400 }
  ink_color:
    r: { min: 5, max: 50 }
    g: { min: 5, max: 50 }
    b: { min: 5, max: 50 }
  background_color:
    r: { min: 215, max: 250 }
    g: { min: 210, max: 245 }
    b: { min: 195, max: 235 }
  antialiasing: true
  subpixel: false
```

Per-sample, each value is drawn from its distribution. Independent draws
across R/G/B let you sample warm cream backgrounds and cool ink without
correlation matrices.

## Layout modes

`layout.mode` selects the geometry of each rendered sample.

### `word_crops`

The default for recognition training. Each sample is a single word.

```yaml
layout:
  mode: word_crops
  padding_px: { min: 4, max: 14 }
  baseline_jitter_px: { min: -2, max: 2 }   # vertical wiggle
```

Output: tight image around the word; ground truth = the word string.

### `lines`

Each sample is one line of text (multiple words on a baseline).

```yaml
layout:
  mode: lines
  max_width_px: 800
  padding_px: { min: 4, max: 12 }
  word_spacing: { min: 1.0, max: 1.4 }
  baseline_jitter_px: { min: -2, max: 2 }
```

Output: image of N words on one baseline; ground truth = the line.

### `paragraphs`

Wrapped multi-line block.

```yaml
layout:
  mode: paragraphs
  max_width_px: 800
  max_lines: 8
  line_spacing: { min: 1.1, max: 1.5 }
  alignment: justify        # left | right | center | justify
  paragraph_indent_em: 0
```

Output: image of a paragraph; ground truth = paragraph text plus per-line
bounding boxes (used by detection mode in spec 08).

### `pages`

Multi-paragraph page synthesis. v1 supports a single column with
configurable margins and chapter-style headings; multi-column is a stretch
goal.

```yaml
layout:
  mode: pages
  page_size_px: [1200, 1800]
  margins_px: { min: 60, max: 140 }
  paragraphs_per_page: { min: 3, max: 8 }
  heading_probability: 0.2
  drop_cap_probability: 0.1
```

Output: page image + per-paragraph + per-line + per-word bounding boxes.

## Ground truth captured per sample

Regardless of layout mode, each rendered sample emits:

| Field | Meaning |
|-------|---------|
| `text` | Codepoint string of the rendered text |
| `font_path` | Resolved font path |
| `font_size_pt` | Drawn size |
| `dpi` | Drawn DPI |
| `bbox` | Tight bounding box of the inked region (px) |
| `glyph_runs` | Per-glyph (or per-cluster) bbox + cluster index, when shaping_engine = harfbuzz |
| `lines` | (lines/paragraphs/pages) per-line bbox + text |
| `words` | (lines/paragraphs/pages) per-word bbox + text |

This metadata flows into the manifest (spec 08). Detection-mode output
uses `lines`/`words` to write `pages.json`-compatible records.

## Rendering pitfalls

- **Mark stacking.** Gaelic dotted consonants stored as base + combining
  mark may render with the dot misaligned if the font's mark positioning
  is bad. Validate visually; prefer precomposed codepoints when both forms
  exist.
- **Tofu / .notdef.** A square box means the font doesn't have that glyph.
  The validator should catch this; the renderer skips and logs.
- **Hinting.** TrueType vs CFF/PostScript fonts hint differently at small
  sizes — let DPI variation cover this rather than sweeping hint settings.
- **Italic fonts at low DPI.** Glyphs touch and confuse the model. Use
  italics sparingly and at higher DPI.
