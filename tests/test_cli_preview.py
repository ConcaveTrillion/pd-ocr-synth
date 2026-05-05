"""Tests for ``pd-ocr-synth preview`` (M05 deliverable).

Exercises the CLI happy-path against a hermetic tmp recipe that
points at the bundled Bunchló GC font + a tiny local seed-words
corpus. Verifies the directory layout the roadmap requires:

```
<output>/
    images/         # PNG files
    manifest.jsonl  # one record per sample
    stats.json      # summary counters
```
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pd_ocr_synth.cli import main

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; preview test skipped.")
    return _BUNDLED_FONT


_RECIPE_TEMPLATE = """\
schema_version: 1
name: preview-smoke
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 1
corpus:
  - type: local
    path: ./seed-words.txt
fonts:
  - path: {font_path}
    weight: 1.0
rendering:
  font_size_pt: {{ min: 14, max: 18 }}
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: word_crops
  padding_px: 4
"""

_SEED_WORDS = "\n".join(["ḃeaḋ", "ċeann", "ḋuine", "ḟear", "ġloine", "ṁaṫair"]) + "\n"


def _setup(tmp_path: Path) -> Path:
    font = _require_font()
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE_TEMPLATE.format(font_path=font), encoding="utf-8")
    (tmp_path / "seed-words.txt").write_text(_SEED_WORDS, encoding="utf-8")
    return rp


def test_preview_writes_images_manifest_and_stats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    output = tmp_path / "preview-out"

    rc = main(
        [
            "preview",
            str(rp),
            "--count",
            "3",
            "--output",
            str(output),
            "--seed",
            "7",
        ]
    )
    assert rc == 0, f"preview exited with {rc}: {capsys.readouterr().err}"

    images = sorted((output / "images").glob("*.png"))
    assert len(images) == 3, f"expected 3 PNGs, got {len(images)}"
    for image_path in images:
        assert image_path.stat().st_size > 100  # non-trivial PNG bytes

    manifest_path = output / "manifest.jsonl"
    assert manifest_path.exists()
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    assert len(records) == 3
    for record in records:
        assert record["status"] == "rendered"
        assert "image" in record and record["image"].startswith("images/")
        assert record["text"]
        assert "font_path" in record and "font_size_pt" in record and "dpi" in record
        assert "bbox" in record and len(record["bbox"]) == 4
        assert "glyph_runs" in record and isinstance(record["glyph_runs"], list)

    stats_path = output / "stats.json"
    assert stats_path.exists()
    stats = json.loads(stats_path.read_text())
    assert stats["recipe"] == "preview-smoke"
    assert stats["seed"] == 7
    assert stats["count"] == 3
    assert stats["rendered"] == 3
    assert stats["skipped"] == 0


def test_preview_rejects_zero_count(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    rc = main(
        [
            "preview",
            str(rp),
            "--count",
            "0",
            "--output",
            str(tmp_path / "out"),
        ]
    )
    assert rc == 2  # USAGE_EXIT
    assert "must be positive" in capsys.readouterr().err


def test_preview_default_count_is_50_helper() -> None:
    """The preview module's default count is the spec's ``50``."""

    from pd_ocr_synth.render.preview import DEFAULT_PREVIEW_COUNT

    assert DEFAULT_PREVIEW_COUNT == 50
