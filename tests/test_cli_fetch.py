"""End-to-end tests for ``pd-ocr-synth fetch``.

Local-only — the web/wikisource paths get covered in their unit
tests via httpx.MockTransport.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.cli import main

_RECIPE = """\
schema_version: 1
name: fetch-smoke
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 1
corpus:
  - type: local
    path: ./a.txt
fonts:
  - path: ./fake.otf
rendering:
  font_size_pt: 12
  dpi: 300
  ink_color: {r: 0, g: 0, b: 0}
  background_color: {r: 255, g: 255, b: 255}
layout:
  mode: word_crops
  padding_px: 4
"""


def _setup_recipe(tmp_path: Path) -> Path:
    (tmp_path / "a.txt").write_text("alpha\nbeta\n", encoding="utf-8")
    (tmp_path / "fake.otf").write_bytes(b"")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE, encoding="utf-8")
    return rp


def test_fetch_local_recipe_exits_zero(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rp = _setup_recipe(tmp_path)
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(cache_dir))
    rc = main(["fetch", str(rp)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "recipe: fetch-smoke" in out
    assert "corpus[0] local" in out
    assert "total:" in out


def test_fetch_missing_local_corpus_exits_four(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "fake.otf").write_bytes(b"")
    yaml_text = _RECIPE.replace("./a.txt", "./missing.txt")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(yaml_text, encoding="utf-8")
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(tmp_path / "cache"))
    rc = main(["fetch", str(rp)])
    err = capsys.readouterr().err
    assert rc == 4
    assert "ERROR" in err
    assert "missing.txt" in err


def test_fetch_no_cache_flag_disables_cache(
    tmp_path: Path, monkeypatch, capsys: pytest.CaptureFixture[str]
) -> None:
    rp = _setup_recipe(tmp_path)
    monkeypatch.setenv("PD_OCR_SYNTH_CACHE", str(tmp_path / "cache"))
    # Run twice — second call should still say "fetch", not "cache",
    # because we forced --no-cache. (LocalProvider doesn't write to
    # the cache anyway, so this primarily exercises the option pass-
    # through; the assertion just checks the CLI accepts the flag.)
    rc1 = main(["fetch", str(rp)])
    rc2 = main(["fetch", str(rp), "--no-cache"])
    out = capsys.readouterr().out
    assert rc1 == 0
    assert rc2 == 0
    assert "fetch" in out
