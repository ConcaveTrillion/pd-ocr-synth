# 08 — Output format

This project produces datasets in a layout that `pd-ocr-trainer`'s
`dataset_store.py` already knows how to consume. The contract is
versioned: `output.format: pd-ocr-trainer/v1`.

## Modes

| Mode | Geometry | Trainer task |
|------|----------|--------------|
| `recognition` | Tight word/line crops | Train recognition model |
| `detection` | Full pages with bbox annotations | Train detection model |

Detection-mode rendering requires `layout.mode` ∈ {`paragraphs`, `pages`}.
Recognition-mode rendering requires `layout.mode` ∈ {`word_crops`, `lines`}.

## Recognition mode layout

```
<destination>/
├── images/
│   ├── 0000000.png
│   ├── 0000001.png
│   └── ...
├── labels.json         # JSON map: {image_name: text, ...}
├── manifest.jsonl      # one record per attempted sample (provenance)
├── recipe.snapshot.yaml  # the resolved, validated recipe
└── stats.json
```

`labels.json` is a single JSON object whose keys are the image
filenames (no path components) and whose values are the plain-text
labels:

```json
{
  "0000000.png": "Séadna",
  "0000001.png": "⁊ ḋuḃairt sé"
}
```

This format is what `pd-ocr-trainer/src/pd_ocr_trainer/dataset_store.py`
consumes via `RecognitionDataset(img_folder=..., labels_path=...)`.
(An earlier draft of this spec specified `labels.csv`; the trainer's
existing reader is the canonical contract, so we matched it during
M07.)

Filenames are zero-padded to enough digits for the configured `count`,
with a minimum of seven digits so a smoke run and a full run produce
the same naming convention.

## Detection mode layout

```
<destination>/
├── images/
│   ├── page_0000000.png
│   └── ...
├── labels.json         # per-image annotation map
├── manifest.jsonl
├── recipe.snapshot.yaml
└── stats.json
```

`labels.json` is what `doctr.datasets.DetectionDataset` (used by
`pd-ocr-trainer/train_detect.py`) actually reads. Its top-level keys
are page filenames; each value is an annotation object that carries
both the doctr-required fields and our richer ground truth:

```json
{
  "page_0000000.png": {
    "img_dimensions": [1200, 1800],
    "img_hash": "<sha256 hex>",
    "polygons": [
      [[120, 200], [1080, 200], [1080, 240], [120, 240]],
      ...
    ],
    "lines": [
      {
        "bbox": [120, 200, 1080, 240],
        "polygon": [[120, 200], [1080, 200], [1080, 240], [120, 240]],
        "text": "Cuiḋ ḋ'á aimsir...",
        "words": [
          { "bbox": [120, 205, 220, 235], "text": "Cuiḋ" }
        ]
      }
    ]
  }
}
```

`polygons` is the flat list (one 4-corner polygon per detected line)
that doctr's detection head consumes; `lines` is the rich GT we use
ourselves and emit so the labeler / parquet publish path can recover
the full annotation without re-rendering. Doctr ignores fields it
doesn't recognize.

(An earlier draft of this spec specified `pages.json`; the trainer's
existing reader is the canonical contract, so we matched its
`labels.json` filename during M09 — same precedent as recognition mode
matching `labels.json` over the original `labels.csv` draft.)

## `manifest.jsonl`

One JSON object per line, per generated sample. Captures provenance so
training failures can be traced back to source.

```json
{
  "id": "0000000",
  "image": "images/0000000.png",
  "text": "Séadna",
  "corpus": {
    "provider": "wikisource",
    "key": "wikisource:ga:Séadna",
    "offset": 1024
  },
  "transforms_applied": ["normalize_whitespace", "long_s_medial"],
  "font": {
    "path": "/abs/path/Bunchlo-GC.otf",
    "size_pt": 14.0,
    "features": {"liga": true}
  },
  "render": {
    "dpi": 300,
    "ink_rgb": [22, 18, 25],
    "bg_rgb": [240, 232, 215]
  },
  "degradations_applied": [
    { "kind": "skew", "params": {"angle_deg": -1.2} },
    { "kind": "paper_texture", "params": {"texture": "aged-1.png", "opacity": 0.42} },
    { "kind": "jpeg", "params": {"quality": 78} }
  ],
  "warnings": []
}
```

Why this is worth the bytes:

- Diagnose model failure modes (does it always fail on `paper_texture` >
  0.5? on a particular font?).
- Reproduce a single sample exactly given the seed.
- Compute per-font / per-degradation coverage stats.

## `recipe.snapshot.yaml`

The fully-resolved recipe written next to the output. This includes:

- Path expansions (absolute paths)
- Resolved presets
- The exact tool version and seed used
- A SHA-256 of every font and corpus file at render time

A future re-render with the same snapshot should produce identical output.

## `stats.json`

Run-level statistics:

```json
{
  "samples_planned": 50000,
  "samples_written": 49997,
  "samples_skipped": 3,
  "skip_reasons": { "missing_glyph": 3 },
  "fonts_used": {
    "Bunchlo-GC.otf": 19998,
    "Seanchlo-GC.otf": 20015,
    "Duibhlinn.ttf": 9984
  },
  "tokens_unique": 8217,
  "wall_time_seconds": 412.3
}
```

## Naming convention

`<destination>` is the literal trainer profile directory. With

```yaml
destination: ${PD_ML_MODELS}/ml-recognition/gaelic/recognition
```

you end up at the path `pd-ocr-trainer` reads when its profile is
`gaelic`. No further import step is needed; the trainer picks up the
dataset on next load.

## Idempotency and resumption

By default, `render` refuses to write into a non-empty directory unless
`--force` (clears first) or `--resume` (continues from the highest-
numbered existing sample, reusing the same seed-derived stream).

`--resume` requires that `recipe.snapshot.yaml` matches the current
recipe (modulo `count`). Mismatch aborts.
