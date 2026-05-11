"""Integration tests for the render audit log (M10 stretch).

Drives ``pd-ocr-synth render`` end-to-end and asserts the audit JSONL
sidecar lands at ``<output>/_audit.jsonl`` with the right shape, that
``--no-audit`` suppresses emission, and that the
``PD_OCR_SYNTH_NO_AUDIT`` env var overrides even an enabled call.

Recipe shape mirrors ``test_cli_render.py`` so we exercise the same
end-to-end path the user runs, just with the audit assertion folded
in.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pd_ocr_synth.audit import AUDIT_DISABLE_ENV, AUDIT_FILENAME, AUDIT_SCHEMA_VERSION
from pd_ocr_synth.cli import main

_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; render audit test skipped.")
    return _BUNDLED_FONT


_RECIPE = """\
schema_version: 1
name: audit-smoke
seed: 21
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./trainer-out
  count: 4
corpus:
  - type: local
    path: ./seed-words.txt
fonts:
  - path: {font}
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
    rp.write_text(_RECIPE.format(font=font), encoding="utf-8")
    (tmp_path / "seed-words.txt").write_text(_SEED_WORDS, encoding="utf-8")
    return rp


def _render(rp: Path, out: Path, *extra_args: str) -> int:
    return main(
        [
            "render",
            str(rp),
            "--count",
            "3",
            "--output",
            str(out),
            "--seed",
            "21",
            "--workers",
            "1",
            *extra_args,
        ]
    )


def test_render_writes_audit_jsonl_with_expected_fields(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = _render(rp, out)
    assert rc == 0, capsys.readouterr().err

    audit_path = out / AUDIT_FILENAME
    assert audit_path.is_file()
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["schema_version"] == AUDIT_SCHEMA_VERSION
    assert payload["recipe_name"] == "audit-smoke"
    # Recipe SHA is the hash of the YAML on disk — non-empty hex.
    assert isinstance(payload["recipe_sha"], str) and len(payload["recipe_sha"]) == 64
    assert payload["seed"] == 21
    assert payload["count"] == 3
    assert payload["workers"] == 1
    # Outcome counts must add up to the requested count (rendered or
    # skipped — degradation isn't enabled in this recipe so a clean
    # render is the expected case).
    assert payload["rendered"] + payload["skipped"] == 3
    assert payload["runtime_seconds"] >= 0.0
    # ``output_dir`` is captured as the absolute resolved path.
    assert Path(payload["output_dir"]) == out.resolve()
    # Timestamp is ISO-8601 UTC ``Z`` form.
    assert payload["timestamp"].endswith("Z")


def test_render_appends_second_audit_entry_on_resume(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Two renders into the same dir → two audit entries."""

    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = _render(rp, out)
    assert rc == 0, capsys.readouterr().err

    # Second run resumes the same destination — must not clobber the
    # existing audit line.
    rc = _render(rp, out, "--resume")
    assert rc == 0, capsys.readouterr().err

    audit_path = out / AUDIT_FILENAME
    lines = audit_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payloads = [json.loads(line) for line in lines]
    # Both entries reference the same recipe + seed.
    assert {p["seed"] for p in payloads} == {21}
    assert {p["recipe_sha"] for p in payloads} == {payloads[0]["recipe_sha"]}


def test_render_no_audit_flag_suppresses_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = _render(rp, out, "--no-audit")
    assert rc == 0, capsys.readouterr().err

    assert not (out / AUDIT_FILENAME).exists()


def test_render_no_audit_env_var_suppresses_log(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``PD_OCR_SYNTH_NO_AUDIT=1`` overrides the default-on behavior."""

    monkeypatch.setenv(AUDIT_DISABLE_ENV, "1")
    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = _render(rp, out)
    assert rc == 0, capsys.readouterr().err

    assert not (out / AUDIT_FILENAME).exists()


def test_render_audit_does_not_count_toward_labels_or_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The audit sidecar lives outside the trainer-consumed file set.

    Trainer loaders read ``labels.json`` + ``images/`` only; the
    audit JSONL must not pollute those surfaces, and must not be
    referenced from ``stats.json`` (which is the trainer-consumed
    summary). This test pins the contract so a future refactor that
    accidentally folds the audit row into stats.json fails loudly.
    """

    from pd_ocr_synth.output.recognition import LABELS_FILENAME, STATS_FILENAME

    rp = _setup(tmp_path)
    out = tmp_path / "trainer-out"

    rc = _render(rp, out)
    assert rc == 0, capsys.readouterr().err

    labels = json.loads((out / LABELS_FILENAME).read_text(encoding="utf-8"))
    assert AUDIT_FILENAME not in labels
    # No trainer-consumed sidecar mentions the audit filename.
    stats = json.loads((out / STATS_FILENAME).read_text(encoding="utf-8"))
    assert AUDIT_FILENAME not in json.dumps(stats)
