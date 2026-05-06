"""Unit tests for the dataset-card README generator (M08).

Covers ``pd_ocr_synth.publish.dataset_card``: front-matter assembly,
body sections, override paths, and the wiring through
``build_recognition_staging``. Pure file-IO; no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from PIL import Image

from pd_ocr_synth.publish import (
    README_FILENAME,
    DatasetCardInputs,
    build_recognition_staging,
    load_card_inputs,
    render_dataset_card,
    write_dataset_card,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _snapshot(
    *,
    name: str = "gaelic",
    description: str | None = None,
    tool_version: str = "0.1.2",
    seed: int = 7,
    fonts: list[dict[str, Any]] | None = None,
    corpus: list[dict[str, Any]] | None = None,
    publish: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal-but-realistic snapshot dict for the renderer.

    Mirrors the on-disk schema produced by ``output.snapshot``: top
    level has ``tool_version``, ``seed``, ``recipe`` (the resolved
    recipe payload), ``input_hashes``.
    """

    recipe: dict[str, Any] = {
        "schema_version": 1,
        "name": name,
        "seed": seed,
        "fonts": fonts or [{"path": f"/abs/fonts/{name}.otf"}],
        "corpus": corpus or [],
    }
    if description is not None:
        recipe["description"] = description
    if publish is not None:
        recipe["publish"] = {"hf_dataset": publish}
    return {
        "tool_version": tool_version,
        "seed": seed,
        "recipe": recipe,
        "input_hashes": {},
    }


def _stats(
    *,
    samples_written: int = 100,
    samples_planned: int = 100,
    samples_skipped: int = 0,
    fonts_used: dict[str, int] | None = None,
    tokens_unique: int = 0,
    wall_time_seconds: float = 0.0,
) -> dict[str, Any]:
    return {
        "samples_planned": samples_planned,
        "samples_written": samples_written,
        "samples_skipped": samples_skipped,
        "skip_reasons": {},
        "fonts_used": fonts_used or {"gaelic.otf": samples_written},
        "tokens_unique": tokens_unique,
        "wall_time_seconds": wall_time_seconds,
    }


def _split_card(card_text: str) -> tuple[dict[str, Any], str]:
    """Parse the YAML front matter and return ``(front_matter, body)``."""

    assert card_text.startswith("---\n"), "front matter must lead the file"
    rest = card_text[4:]
    end = rest.index("\n---\n")
    fm_text = rest[:end]
    body = rest[end + len("\n---\n") :].lstrip("\n")
    fm = yaml.safe_load(fm_text)
    assert isinstance(fm, dict), "front matter must parse to a mapping"
    return fm, body


def _inputs(
    snapshot: dict[str, Any] | None = None,
    *,
    snapshot_bytes: bytes | None = None,
    stats: dict[str, Any] | None = None,
    description_override: str | None = None,
) -> DatasetCardInputs:
    snap = snapshot if snapshot is not None else _snapshot()
    if snapshot_bytes is None:
        snapshot_bytes = yaml.safe_dump(snap, sort_keys=False).encode("utf-8")
    return DatasetCardInputs(
        snapshot=snap,
        snapshot_bytes=snapshot_bytes,
        stats=stats,
        description_override=description_override,
    )


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------


def test_front_matter_has_pd_ocr_conventional_keys() -> None:
    card = render_dataset_card(_inputs())
    fm, _ = _split_card(card)
    assert fm["pd-ocr-shape"] == "recognition/v1"
    assert fm["pd-ocr-source"] == "pd-ocr-synth"
    assert fm["pd-ocr-render-tool-version"] == "0.1.2"
    # SHA-256 hex is 64 lowercase hex chars; we don't pin the value
    # because it's a hash of yaml.safe_dump output (Python-version
    # stable but locking it would just make the test brittle).
    sha = fm["pd-ocr-recipe-sha"]
    assert isinstance(sha, str)
    assert len(sha) == 64
    assert all(c in "0123456789abcdef" for c in sha)


def test_front_matter_carries_publish_block_values() -> None:
    snapshot = _snapshot(
        publish={
            "repo": "ntw8532/pd-ocr-synth-gaelic",
            "license": "cc-by-4.0",
            "tags": ["ocr", "gaelic", "irish", "pd-ocr", "synthetic"],
            "language": ["ga"],
        },
    )
    card = render_dataset_card(_inputs(snapshot, stats=_stats(samples_written=50_000)))
    fm, _ = _split_card(card)
    assert fm["license"] == "cc-by-4.0"
    assert fm["task_categories"] == ["text-recognition"]
    assert fm["language"] == ["ga"]
    assert fm["tags"] == ["ocr", "gaelic", "irish", "pd-ocr", "synthetic"]


def test_front_matter_omits_keys_when_recipe_has_no_publish_block() -> None:
    """A recipe without a publish block should still produce a valid
    card — the pd-ocr-* keys come from the snapshot, not the publish
    block."""

    snapshot = _snapshot(publish=None)
    card = render_dataset_card(_inputs(snapshot))
    fm, _ = _split_card(card)
    assert "license" not in fm
    assert "language" not in fm
    assert "tags" not in fm
    # task_categories is fixed so it's always set.
    assert fm["task_categories"] == ["text-recognition"]
    assert fm["pd-ocr-shape"] == "recognition/v1"


def test_size_categories_buckets_by_sample_count() -> None:
    """Spec example uses ``10K<n<100K`` for 50000 samples; verify the
    bucket boundaries land on the canonical HF set."""

    cases = [
        (500, "n<1K"),
        (5_000, "1K<n<10K"),
        (50_000, "10K<n<100K"),
        (500_000, "100K<n<1M"),
        (5_000_000, "1M<n<10M"),
        (50_000_000, "n>10M"),
    ]
    for samples, expected in cases:
        card = render_dataset_card(_inputs(stats=_stats(samples_written=samples)))
        fm, _ = _split_card(card)
        assert fm["size_categories"] == [expected], f"{samples} → {expected}"


def test_size_categories_omitted_without_stats() -> None:
    card = render_dataset_card(_inputs(stats=None))
    fm, _ = _split_card(card)
    assert "size_categories" not in fm


def test_recipe_sha_changes_with_snapshot_bytes() -> None:
    """Two snapshots that differ in any byte should produce different
    recipe-SHA values — that's the whole point of the pin."""

    a = render_dataset_card(_inputs(snapshot_bytes=b"recipe: a\n"))
    b = render_dataset_card(_inputs(snapshot_bytes=b"recipe: b\n"))
    fm_a, _ = _split_card(a)
    fm_b, _ = _split_card(b)
    assert fm_a["pd-ocr-recipe-sha"] != fm_b["pd-ocr-recipe-sha"]


def test_recipe_sha_omitted_when_snapshot_bytes_empty() -> None:
    card = render_dataset_card(_inputs(snapshot_bytes=b""))
    fm, _ = _split_card(card)
    assert "pd-ocr-recipe-sha" not in fm


# ---------------------------------------------------------------------------
# Body content
# ---------------------------------------------------------------------------


def test_body_uses_recipe_name_in_title_and_reproduce_block() -> None:
    snapshot = _snapshot(name="gaelic", description="Synthetic Gaelic OCR dataset.")
    card = render_dataset_card(_inputs(snapshot))
    _, body = _split_card(card)
    assert body.startswith("# pd-ocr-synth — gaelic")
    assert "Synthetic Gaelic OCR dataset." in body
    # Reproduce block uses the recipe name in every command.
    assert "pd-ocr-synth fetch gaelic" in body
    assert "pd-ocr-synth render gaelic" in body
    assert "pd-ocr-synth publish gaelic --repo <this-repo>" in body


def test_body_renders_stats_section() -> None:
    snapshot = _snapshot(
        fonts=[
            {"path": "/abs/bungc.otf"},
            {"path": "/abs/seangc.otf"},
        ]
    )
    stats = _stats(
        samples_written=50_000,
        fonts_used={"bungc.otf": 30_000, "seangc.otf": 20_000},
        tokens_unique=8217,
        wall_time_seconds=412.4,
    )
    card = render_dataset_card(_inputs(snapshot, stats=stats))
    _, body = _split_card(card)
    assert "## Stats" in body
    assert "- Samples: 50000" in body
    # Either ordering is fine; both names must show up.
    assert "bungc" in body and "seangc" in body
    assert "- Tokens (unique): 8217" in body
    # Wall-time gets rounded to whole seconds for display.
    assert "- Render time: 412s" in body


def test_body_falls_back_to_recipe_fonts_when_stats_missing() -> None:
    """Without stats.json we still know which fonts the recipe
    references, so the Stats section can list them — just without the
    sample / token / time counters that need stats to be meaningful."""

    snapshot = _snapshot(
        fonts=[
            {"path": "/abs/bungc.otf"},
            {"path": "/abs/seangc.otf"},
        ]
    )
    card = render_dataset_card(_inputs(snapshot, stats=None))
    _, body = _split_card(card)
    assert "## Stats" in body
    assert "- Fonts used: bungc, seangc" in body
    # No sample count without stats.
    assert "- Samples:" not in body
    assert "- Tokens" not in body
    assert "- Render time" not in body


def test_body_renders_provenance_section_with_corpus_and_recipe_sha() -> None:
    snapshot = _snapshot(
        name="gaelic",
        corpus=[
            {"type": "wikisource", "key": "ga:Séadna"},
            {"type": "celt", "id": "G100001A"},
            {"type": "local", "path": "/abs/seed.txt"},
        ],
    )
    card = render_dataset_card(_inputs(snapshot))
    fm, body = _split_card(card)
    assert "## Provenance" in body
    assert f"- Recipe SHA: {fm['pd-ocr-recipe-sha']}" in body
    assert "- Tool version: pd-ocr-synth 0.1.2" in body
    assert "wikisource:ga:Séadna" in body
    assert "celt:G100001A" in body
    assert "local:seed.txt" in body
    assert "- Fonts: see recipe for licenses; not bundled" in body


def test_description_override_replaces_recipe_description() -> None:
    snapshot = _snapshot(description="Default description.")
    card = render_dataset_card(
        _inputs(snapshot, description_override="Custom hand-written intro.\n")
    )
    _, body = _split_card(card)
    assert "Custom hand-written intro." in body
    assert "Default description." not in body


def test_body_handles_recipe_without_name() -> None:
    """A snapshot whose recipe lacks a name should still render a
    reasonable title and skip the reproduce-block (we don't have a
    valid CLI argument to put in it)."""

    snapshot = _snapshot(name="")
    card = render_dataset_card(_inputs(snapshot))
    _, body = _split_card(card)
    assert body.startswith("# pd-ocr-synth dataset")
    assert "pd-ocr-synth fetch" not in body


# ---------------------------------------------------------------------------
# load_card_inputs
# ---------------------------------------------------------------------------


def test_load_card_inputs_reads_snapshot_and_stats(tmp_path: Path) -> None:
    snapshot_dict = _snapshot()
    (tmp_path / "recipe.snapshot.yaml").write_text(
        yaml.safe_dump(snapshot_dict, sort_keys=False), encoding="utf-8"
    )
    stats_dict = _stats(samples_written=42)
    (tmp_path / "stats.json").write_text(json.dumps(stats_dict), encoding="utf-8")

    inputs = load_card_inputs(tmp_path)

    assert inputs.snapshot["tool_version"] == "0.1.2"
    assert inputs.snapshot_bytes  # non-empty
    assert inputs.stats == stats_dict


def test_load_card_inputs_tolerates_missing_stats(tmp_path: Path) -> None:
    snapshot_dict = _snapshot()
    (tmp_path / "recipe.snapshot.yaml").write_text(
        yaml.safe_dump(snapshot_dict, sort_keys=False), encoding="utf-8"
    )
    inputs = load_card_inputs(tmp_path)
    assert inputs.stats is None


def test_load_card_inputs_tolerates_corrupt_stats(tmp_path: Path) -> None:
    """A truncated / malformed stats.json shouldn't block card
    generation — we just lose the Stats section."""

    snapshot_dict = _snapshot()
    (tmp_path / "recipe.snapshot.yaml").write_text(
        yaml.safe_dump(snapshot_dict, sort_keys=False), encoding="utf-8"
    )
    (tmp_path / "stats.json").write_text("not json", encoding="utf-8")
    inputs = load_card_inputs(tmp_path)
    assert inputs.stats is None


def test_load_card_inputs_returns_empty_when_no_snapshot(tmp_path: Path) -> None:
    inputs = load_card_inputs(tmp_path)
    assert inputs.snapshot == {}
    assert inputs.snapshot_bytes == b""


def test_description_file_resolved_relative_to_local_parent(tmp_path: Path) -> None:
    """``description_file: ./gaelic/README.md.template`` (per the spec)
    should resolve relative to the local output's parent — the recipe
    dir in typical usage."""

    recipe_dir = tmp_path / "recipes"
    recipe_dir.mkdir()
    (recipe_dir / "gaelic").mkdir()
    template_path = recipe_dir / "gaelic" / "README.md.template"
    template_path.write_text("Hand-authored intro for the Gaelic dataset.\n", encoding="utf-8")

    local = recipe_dir / "out"  # parent is recipe_dir, where the relative path resolves
    local.mkdir()
    snapshot_dict = _snapshot(
        publish={"repo": "x/y", "description_file": "./gaelic/README.md.template"},
    )
    (local / "recipe.snapshot.yaml").write_text(
        yaml.safe_dump(snapshot_dict, sort_keys=False), encoding="utf-8"
    )

    inputs = load_card_inputs(local)
    assert inputs.description_override == "Hand-authored intro for the Gaelic dataset."


def test_description_file_missing_falls_back_to_recipe_description(tmp_path: Path) -> None:
    snapshot_dict = _snapshot(
        description="Recipe-level description.",
        publish={"repo": "x/y", "description_file": "./does/not/exist.md"},
    )
    (tmp_path / "recipe.snapshot.yaml").write_text(
        yaml.safe_dump(snapshot_dict, sort_keys=False), encoding="utf-8"
    )
    inputs = load_card_inputs(tmp_path)
    assert inputs.description_override is None
    card = render_dataset_card(inputs)
    _, body = _split_card(card)
    assert "Recipe-level description." in body


# ---------------------------------------------------------------------------
# write_dataset_card
# ---------------------------------------------------------------------------


def test_write_dataset_card_writes_to_readme_in_staging(tmp_path: Path) -> None:
    inputs = _inputs()
    target = write_dataset_card(tmp_path, inputs)
    assert target == tmp_path / README_FILENAME
    assert target.is_file()
    text = target.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "pd-ocr-shape: recognition/v1" in text


# ---------------------------------------------------------------------------
# Integration: README is written by build_recognition_staging
# ---------------------------------------------------------------------------


def _write_local_for_integration(tmp_path: Path) -> Path:
    """Build a tiny local recognition output used by the integration
    tests below. Recognition layout matches what RecognitionWriter
    produces — except we build it by hand so the tests don't depend on
    M07's full render stack."""

    local = tmp_path / "local"
    images = local / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(220, 220, 220)).save(images / "0000000.png", format="PNG")
    (local / "labels.json").write_text(
        json.dumps({"0000000.png": "alpha"}) + "\n", encoding="utf-8"
    )
    (local / "manifest.jsonl").write_text(
        json.dumps(
            {
                "index": 0,
                "id": "0000000",
                "image": "images/0000000.png",
                "text": "alpha",
                "status": "rendered",
                "font": {"name": "fake.otf", "path": "/abs/fake.otf", "size_pt": 14.0},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    snapshot = _snapshot(
        name="integration",
        publish={
            "repo": "me/integration",
            "license": "cc-by-4.0",
            "tags": ["ocr", "test"],
            "language": ["ga"],
        },
    )
    (local / "recipe.snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
    )
    (local / "stats.json").write_text(
        json.dumps(_stats(samples_written=1, fonts_used={"fake.otf": 1})),
        encoding="utf-8",
    )
    return local


def test_build_recognition_staging_writes_readme(tmp_path: Path) -> None:
    local = _write_local_for_integration(tmp_path)
    staging = tmp_path / "staging"

    result = build_recognition_staging(local, staging)

    assert result.readme_written is True
    readme = staging / README_FILENAME
    assert readme.is_file()
    text = readme.read_text(encoding="utf-8")
    fm, body = _split_card(text)
    assert fm["license"] == "cc-by-4.0"
    assert fm["pd-ocr-shape"] == "recognition/v1"
    assert "# pd-ocr-synth — integration" in body


def test_build_recognition_staging_skips_readme_without_snapshot(tmp_path: Path) -> None:
    """A local output without a snapshot can't ground the README's
    front matter (no recipe block, no tool version, no recipe-SHA), so
    we deliberately skip generation rather than emit a misleading stub."""

    local = tmp_path / "local"
    images = local / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(images / "0000000.png", format="PNG")
    (local / "labels.json").write_text(
        json.dumps({"0000000.png": "alpha"}) + "\n", encoding="utf-8"
    )
    staging = tmp_path / "staging"

    result = build_recognition_staging(local, staging)

    assert result.snapshot_copied is False
    assert result.readme_written is False
    assert not (staging / README_FILENAME).exists()


def test_round_trip_with_real_recognition_writer(tmp_path: Path) -> None:
    """The writer's snapshot, written by ``output.snapshot``, should
    feed cleanly into the dataset-card generator. Locks the M07/M08
    contract for the README path the same way the existing round-trip
    test locks it for ``metadata.jsonl``."""

    from types import SimpleNamespace

    from pd_ocr_synth.output import RecognitionWriter
    from pd_ocr_synth.recipe import load_recipe

    font = tmp_path / "fake.otf"
    font.write_bytes(b"\x00fake")
    seed = tmp_path / "seed.txt"
    seed.write_text("alpha\n", encoding="utf-8")
    rp = tmp_path / "recipe.yaml"
    rp.write_text(
        f"""schema_version: 1
name: round-trip
description: Round-trip integration recipe.
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: ./out
  count: 1
corpus:
  - type: local
    path: {seed}
fonts:
  - path: {font}
rendering:
  font_size_pt: 14
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 240, b: 240 }}
layout:
  mode: word_crops
  padding_px: 4
publish:
  hf_dataset:
    repo: me/round-trip
    license: cc-by-4.0
    tags: [ocr, round-trip]
    language: [ga]
""",
        encoding="utf-8",
    )
    recipe = load_recipe(rp)
    local = tmp_path / "render"

    sample = SimpleNamespace(
        text="alpha",
        image=Image.new("RGB", (16, 8), color=(240, 240, 240)),
        font_path=font,
        font_size_pt=14.0,
        dpi=300,
        ink_color=(10, 10, 10),
        background_color=(240, 240, 240),
        size=(16, 8),
        bbox=(0, 0, 16, 8),
        glyph_runs=(),
    )
    with RecognitionWriter.open(recipe, local, seed=recipe.seed) as writer:
        writer.write_rendered(0, sample, text="alpha", applied_degradations=[])

    staging = tmp_path / "staging"
    result = build_recognition_staging(local, staging)

    assert result.readme_written is True
    text = (staging / README_FILENAME).read_text(encoding="utf-8")
    fm, body = _split_card(text)
    assert fm["license"] == "cc-by-4.0"
    assert fm["language"] == ["ga"]
    assert fm["tags"] == ["ocr", "round-trip"]
    assert fm["pd-ocr-shape"] == "recognition/v1"
    # The tool_version key gets written by output.snapshot — its
    # presence proves the round-trip wired through.
    assert "pd-ocr-render-tool-version" in fm
    assert "# pd-ocr-synth — round-trip" in body
    assert "Round-trip integration recipe." in body


def test_render_dataset_card_is_pure(tmp_path: Path) -> None:
    """The renderer must not touch the filesystem — used by the
    future ``--dry-run`` path which previews the card without writing
    anything."""

    inputs = _inputs()
    before = sorted(p.name for p in tmp_path.iterdir())
    out = render_dataset_card(inputs)
    after = sorted(p.name for p in tmp_path.iterdir())
    assert before == after
    assert isinstance(out, str)
    assert out.startswith("---\n")


# pytest fixtures referenced indirectly so ruff doesn't flag the
# import as unused if a test is removed during iteration.
_ = pytest
