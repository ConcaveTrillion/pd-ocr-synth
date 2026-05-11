"""Unit tests for the staging-dir pre-flight check (M08).

Covers ``pd_ocr_synth.publish.preflight``: the read-only validator
that asserts a built staging dir's README front matter carries every
required ``pd-ocr-*`` key. Pure file-IO; no network, no HF SDK.

The pre-flight is the natural last gate before the upload step
contacts HF, and it's what a future ``--dry-run`` would echo "card
looks good" / "card is missing X" against. Tests treat the required
keys as a *contract*: any drift in the dataset-card writer that drops
one of those keys must surface here as a failure.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from pd_ocr_synth.publish import (
    REQUIRED_FRONT_MATTER_KEYS,
    PreflightError,
    PreflightReport,
    assert_staging_publish_ready,
    build_recognition_staging,
    check_required_front_matter,
)
from pd_ocr_synth.publish.dataset_card import README_FILENAME

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _build_staging(tmp_path: Path) -> Path:
    """Build a real staging dir from a synthetic local recognition output.

    Uses the production ``build_recognition_staging`` so the test
    exercises the actual README the writer emits, not a hand-rolled
    fixture. That matters for this module: we want to detect drift if
    the writer ever drops one of the required keys.
    """

    local = tmp_path / "local"
    images = local / "images"
    images.mkdir(parents=True)
    Image.new("RGB", (8, 8), color=(180, 180, 180)).save(images / "0000000.png", format="PNG")
    (local / "labels.json").write_text('{"0000000.png": "Séadna"}\n', encoding="utf-8")
    (local / "manifest.jsonl").write_text(
        '{"image": "images/0000000.png", "text": "Séadna", "status": "ok",'
        ' "font": {"name": "bungc/bungc.otf", "size_pt": 14.0},'
        ' "degradations_applied": ["jpeg"], "corpus": {"provider": "wikisource", "key": "Séadna"}}\n',
        encoding="utf-8",
    )
    snapshot_yaml = (
        "tool_version: 0.1.2\n"
        "seed: 7\n"
        "recipe:\n"
        "  schema_version: 1\n"
        "  name: gaelic\n"
        "  seed: 7\n"
        "  fonts:\n"
        "    - path: /abs/fonts/gaelic.otf\n"
        "  corpus: []\n"
        "  publish:\n"
        "    hf_dataset:\n"
        "      repo: ntw8532/pd-ocr-synth-gaelic\n"
        "      license: cc-by-4.0\n"
        "      language: [ga]\n"
        "      tags: [ocr, gaelic]\n"
        "input_hashes: {}\n"
    )
    (local / "recipe.snapshot.yaml").write_text(snapshot_yaml, encoding="utf-8")
    (local / "stats.json").write_text(
        '{"samples_planned": 1, "samples_written": 1, "samples_skipped": 0,'
        ' "skip_reasons": {}, "fonts_used": {"gaelic.otf": 1},'
        ' "tokens_unique": 1, "wall_time_seconds": 0.01}\n',
        encoding="utf-8",
    )

    staging = tmp_path / "staging"
    build_recognition_staging(local, staging)
    return staging


def _rewrite_readme(staging: Path, text: str) -> None:
    """Helper: overwrite the README on disk verbatim."""

    (staging / README_FILENAME).write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Required-keys contract
# ---------------------------------------------------------------------------


def test_required_keys_set_matches_spec() -> None:
    """Lock the spec contract: keys named in ``docs/specs/10-publishing.md``.

    This is intentionally a string-match test against the documented
    set rather than reading from another module. The point is to catch
    a silent edit to ``REQUIRED_FRONT_MATTER_KEYS`` that drops a key
    the spec mandates.
    """

    assert REQUIRED_FRONT_MATTER_KEYS == (
        "pd-ocr-shape",
        "pd-ocr-source",
        "pd-ocr-recipe-sha",
        "pd-ocr-render-tool-version",
        "pd-ocr-content-sha",
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_check_passes_for_freshly_built_staging(tmp_path: Path) -> None:
    """A staging dir built end-to-end satisfies every required key."""

    staging = _build_staging(tmp_path)
    report = check_required_front_matter(staging)

    assert report.ok
    assert report.missing_keys == ()
    assert report.empty_keys == ()
    # All required keys land in the front matter the writer produces.
    for key in REQUIRED_FRONT_MATTER_KEYS:
        assert key in report.front_matter, f"writer must emit {key}"
        value = report.front_matter[key]
        assert isinstance(value, str)
        assert value.strip(), f"{key} should be non-empty"


def test_assert_publish_ready_returns_report_on_success(tmp_path: Path) -> None:
    """The strict variant returns the same report when everything is OK."""

    staging = _build_staging(tmp_path)
    report = assert_staging_publish_ready(staging)

    assert isinstance(report, PreflightReport)
    assert report.ok


# ---------------------------------------------------------------------------
# Missing-key detection
# ---------------------------------------------------------------------------


def test_missing_required_key_surfaces_in_report(tmp_path: Path) -> None:
    """If the writer produced a card lacking a required key, surface it."""

    staging = _build_staging(tmp_path)
    # Hand-rewrite the README to drop one required key. The
    # parametrized choice is the field most-likely-to-go-missing
    # (snapshot has no ``tool_version`` → writer omits the line).
    _rewrite_readme(
        staging,
        "---\n"
        "license: cc-by-4.0\n"
        "pd-ocr-shape: recognition/v1\n"
        "pd-ocr-source: pd-ocr-synth\n"
        "pd-ocr-recipe-sha: " + ("a" * 64) + "\n"
        "pd-ocr-content-sha: " + ("b" * 64) + "\n"
        "---\n\n"
        "# pd-ocr-synth — gaelic\n",
    )

    report = check_required_front_matter(staging)
    assert not report.ok
    assert report.missing_keys == ("pd-ocr-render-tool-version",)
    assert report.empty_keys == ()


def test_empty_required_value_treated_as_failure(tmp_path: Path) -> None:
    """Present-but-empty is just as bad as missing — HF lint rejects both."""

    staging = _build_staging(tmp_path)
    _rewrite_readme(
        staging,
        "---\n"
        "pd-ocr-shape: recognition/v1\n"
        "pd-ocr-source: pd-ocr-synth\n"
        'pd-ocr-recipe-sha: ""\n'
        "pd-ocr-render-tool-version: 0.1.2\n"
        "pd-ocr-content-sha: " + ("c" * 64) + "\n"
        "---\n\n"
        "body\n",
    )

    report = check_required_front_matter(staging)
    assert not report.ok
    assert report.missing_keys == ()
    assert report.empty_keys == ("pd-ocr-recipe-sha",)


def test_multiple_failures_listed_in_stable_order(tmp_path: Path) -> None:
    """Missing + empty keys both reported; order matches ``REQUIRED``."""

    staging = _build_staging(tmp_path)
    _rewrite_readme(
        staging,
        "---\n"
        "pd-ocr-shape: recognition/v1\n"
        "pd-ocr-source: pd-ocr-synth\n"
        'pd-ocr-render-tool-version: "   "\n'
        "---\n\n"
        "body\n",
    )

    report = check_required_front_matter(staging)
    # Missing list keeps the order of REQUIRED_FRONT_MATTER_KEYS so
    # the user sees a stable, easy-to-diff message across runs.
    assert report.missing_keys == ("pd-ocr-recipe-sha", "pd-ocr-content-sha")
    assert report.empty_keys == ("pd-ocr-render-tool-version",)


# ---------------------------------------------------------------------------
# Strict variant
# ---------------------------------------------------------------------------


def test_assert_publish_ready_raises_on_missing_key(tmp_path: Path) -> None:
    """The upload-time strict variant turns failures into typed errors."""

    staging = _build_staging(tmp_path)
    _rewrite_readme(
        staging,
        "---\npd-ocr-shape: recognition/v1\npd-ocr-source: pd-ocr-synth\n---\n\nbody\n",
    )

    with pytest.raises(PreflightError) as excinfo:
        assert_staging_publish_ready(staging)

    msg = str(excinfo.value)
    # Error must name the README path so the user can grep it.
    assert str(staging / README_FILENAME) in msg
    assert "missing front-matter keys" in msg
    # All three of the missing keys present — stable message ordering.
    assert "pd-ocr-recipe-sha" in msg
    assert "pd-ocr-render-tool-version" in msg
    assert "pd-ocr-content-sha" in msg


def test_assert_publish_ready_lists_empty_separately(tmp_path: Path) -> None:
    """Empty values appear under their own bullet so users see *why* it failed."""

    staging = _build_staging(tmp_path)
    _rewrite_readme(
        staging,
        "---\n"
        "pd-ocr-shape: recognition/v1\n"
        "pd-ocr-source: pd-ocr-synth\n"
        "pd-ocr-recipe-sha: " + ("a" * 64) + "\n"
        'pd-ocr-render-tool-version: ""\n'
        "pd-ocr-content-sha: " + ("b" * 64) + "\n"
        "---\n\n"
        "body\n",
    )

    with pytest.raises(PreflightError) as excinfo:
        assert_staging_publish_ready(staging)

    msg = str(excinfo.value)
    assert "empty front-matter values" in msg
    assert "pd-ocr-render-tool-version" in msg
    assert "missing" not in msg.lower() or "empty" in msg.lower()


# ---------------------------------------------------------------------------
# Custom required-set override (forward-looking for --dry-run)
# ---------------------------------------------------------------------------


def test_required_override_skips_keys_not_yet_applied(tmp_path: Path) -> None:
    """``--dry-run`` may want to validate before content-SHA is applied.

    A caller that runs the check between dataset-card write and
    content-SHA embed should still see ``ok`` if it passes a custom
    required set without ``pd-ocr-content-sha``.
    """

    staging = _build_staging(tmp_path)
    # Strip the content-SHA line manually to simulate the
    # "before apply_content_sha_to_readme" state.
    text = (staging / README_FILENAME).read_text(encoding="utf-8")
    rewritten_lines = [ln for ln in text.splitlines() if not ln.startswith("pd-ocr-content-sha:")]
    _rewrite_readme(staging, "\n".join(rewritten_lines) + "\n")

    custom = tuple(k for k in REQUIRED_FRONT_MATTER_KEYS if k != "pd-ocr-content-sha")
    report = check_required_front_matter(staging, required=custom)
    assert report.ok

    # The default set still flags it as missing.
    default_report = check_required_front_matter(staging)
    assert not default_report.ok
    assert default_report.missing_keys == ("pd-ocr-content-sha",)


# ---------------------------------------------------------------------------
# Structural failures — no front matter / unparseable / missing README
# ---------------------------------------------------------------------------


def test_missing_readme_raises_typed_error(tmp_path: Path) -> None:
    staging = _build_staging(tmp_path)
    (staging / README_FILENAME).unlink()

    with pytest.raises(PreflightError) as excinfo:
        check_required_front_matter(staging)
    assert README_FILENAME in str(excinfo.value)


def test_no_front_matter_raises_typed_error(tmp_path: Path) -> None:
    staging = _build_staging(tmp_path)
    _rewrite_readme(staging, "# no front matter here\n\nbody only\n")

    with pytest.raises(PreflightError) as excinfo:
        check_required_front_matter(staging)
    assert "front matter" in str(excinfo.value).lower()


def test_invalid_yaml_front_matter_raises(tmp_path: Path) -> None:
    """A YAML parse error names the README path so the bug is locatable."""

    staging = _build_staging(tmp_path)
    _rewrite_readme(staging, "---\nthis: : is: not: yaml:\n---\n\nbody\n")

    with pytest.raises(PreflightError) as excinfo:
        check_required_front_matter(staging)
    assert str(staging / README_FILENAME) in str(excinfo.value)


def test_non_mapping_front_matter_raises(tmp_path: Path) -> None:
    """If the front matter parses to a list / scalar, that's a writer bug."""

    staging = _build_staging(tmp_path)
    _rewrite_readme(staging, "---\n- a\n- b\n---\n\nbody\n")

    with pytest.raises(PreflightError) as excinfo:
        check_required_front_matter(staging)
    assert "mapping" in str(excinfo.value).lower()
