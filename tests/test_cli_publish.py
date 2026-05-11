"""End-to-end tests for ``pd-ocr-synth publish --dry-run`` (M08).

Per ``docs/specs/10-publishing.md`` § Dry run + the matching M08
deliverable in ``docs/roadmap/08-publishing-hf.md``, the dry-run
surface previews what would be uploaded **without contacting HF**:
target repo, file count, total size, dataset-card preview, content
SHA. These tests round-trip the full pipeline against a hermetic
recipe + render so the CLI exit codes, stdout shape, and
spec-mandated wording are all locked down.

Real upload is intentionally still NOT implemented; we assert that a
no-``--dry-run`` invocation returns exit 1 with the documented
"use --dry-run" hint so callers don't think we silently uploaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pd_ocr_synth.cli import main

# Bundled Bunchló GC font, same one the M07 render tests use. Skipping
# when missing keeps CI green on minimal checkouts.
_BUNDLED_FONT = (
    Path(__file__).resolve().parent.parent / "recipes" / "gaelic" / "fonts" / "bungc" / "bungc.otf"
)


def _require_font() -> Path:
    if not _BUNDLED_FONT.exists():
        pytest.skip("Bundled Gaelic font not available; publish CLI test skipped.")
    return _BUNDLED_FONT


_RECIPE_NO_PUBLISH = """\
schema_version: 1
name: publish-smoke
seed: 7
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {dest}
  count: 4
corpus:
  - type: local
    path: ./seed-words.txt
fonts:
  - path: {font}
    weight: 1.0
rendering:
  font_size_pt: {{ min: 14, max: 16 }}
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: word_crops
  padding_px: 4
"""

_RECIPE_WITH_PUBLISH = (
    _RECIPE_NO_PUBLISH
    + """\
publish:
  hf_dataset:
    repo: ntw8532/pd-ocr-synth-publish-smoke
    license: cc-by-4.0
"""
)

_SEED_WORDS = "\n".join(["ḃeaḋ", "ċeann", "ḋuine", "ḟear", "ġloine"]) + "\n"


def _setup_recipe(tmp_path: Path, *, with_publish: bool, dest: Path | None = None) -> Path:
    font = _require_font()
    target_dest = dest if dest is not None else tmp_path / "trainer-out"
    rp = tmp_path / "recipe.yaml"
    body_template = _RECIPE_WITH_PUBLISH if with_publish else _RECIPE_NO_PUBLISH
    body = body_template.format(font=font, dest=target_dest)
    rp.write_text(body, encoding="utf-8")
    (tmp_path / "seed-words.txt").write_text(_SEED_WORDS, encoding="utf-8")
    return rp


def _do_render(recipe_path: Path, out: Path) -> None:
    rc = main(
        [
            "render",
            str(recipe_path),
            "--count",
            "4",
            "--output",
            str(out),
            "--seed",
            "7",
            "--workers",
            "1",
        ]
    )
    assert rc == 0


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_publish_dry_run_prints_plan_and_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Avoid env leakage so the auth chain probes a temp HOME.
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, with_publish=False, dest=out)
    _do_render(rp, out)
    capsys.readouterr()  # drop render output

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/pd-ocr-synth-test",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    text = captured.out
    assert "Would upload to: alice/pd-ocr-synth-test (public)" in text
    assert "Files:" in text
    # Spec 10 § Dry run shows a Dataset card preview + Content SHA line.
    assert "Dataset card preview:" in text
    assert "Content SHA:" in text
    # No token configured → auth chain printed.
    assert "no token resolved" in text


def test_publish_dry_run_uses_recipe_default_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec 10 § Recipe ``publish:`` block: recipe default applies
    when ``--repo`` is not passed."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, with_publish=True, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    rc = main(["publish", str(rp), "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "ntw8532/pd-ocr-synth-publish-smoke" in captured.out


def test_publish_dry_run_private_flag_overrides_visibility(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, with_publish=False, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--private",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "alice/x (private)" in captured.out


def test_publish_dry_run_reports_token_source_when_env_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``ResolvedToken.source`` ("env") shows up; the value never does."""

    monkeypatch.setenv("HF_TOKEN", "hf_super_secret_token_value_aaaaaaaaaaaa")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, with_publish=False, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "Auth: token resolved from env" in captured.out
    # Defensive: make absolutely sure the token value never leaks into
    # stdout / stderr. This is a key audit invariant.
    assert "hf_super_secret_token_value_aaaaaaaaaaaa" not in captured.out
    assert "hf_super_secret_token_value_aaaaaaaaaaaa" not in captured.err


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_publish_without_repo_or_recipe_default_exits_two(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, with_publish=False, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    rc = main(["publish", str(rp), "--dry-run"])
    err = capsys.readouterr().err
    assert rc == 2  # USAGE_EXIT
    assert "no target repo" in err.lower() or "--repo" in err


def test_publish_missing_local_render_exits_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec 10 § Errors and recovery: missing local render → exit 5."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"  # never rendered
    rp = _setup_recipe(tmp_path, with_publish=False, dest=out)

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--dry-run",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 5
    assert "render" in err.lower()


def test_publish_corrupt_local_output_exits_six(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec 10 § Errors and recovery: corrupt local output → exit 6.

    We simulate corruption by removing ``images/`` after a real render
    so the staging builder's ``StagingError`` path fires.
    """

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, with_publish=False, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    # Drop the images dir → staging build raises StagingError.
    import shutil

    shutil.rmtree(out / "images")

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--dry-run",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 6, err
    assert "images" in err.lower()


def test_publish_real_upload_without_token_exits_seven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Real upload (no ``--dry-run``) requires a token. Spec 01 maps
    publish auth failures to exit 7; spec 10 § Errors and recovery
    pins the same. The auth-error message must name the spec-mandated
    resolution chain so the user knows how to fix it.
    """

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, with_publish=False, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    rc = main(["publish", str(rp), "--repo", "alice/x"])
    err = capsys.readouterr().err
    assert rc == 7
    # Auth-error message names every step of the resolution chain so
    # the user has an actionable fix.
    assert "--token" in err
    assert "HF_TOKEN" in err


def test_publish_real_upload_with_sdk_unavailable_exits_seven(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A real upload on a host without ``huggingface_hub`` installed
    must still fail cleanly at exit 7.

    With the SDK adapter landed, ``make_default_transport`` succeeds
    when ``huggingface_hub`` is importable. To exercise the
    ``[publish]`` extra-not-installed path without uninstalling the
    SDK from the test env (which would break sibling tests sharing the
    worker), we patch ``pd_ocr_synth.publish.cli_runner.make_default_transport``
    to raise :class:`SdkUnavailableError` directly. The CLI runner
    must catch it via the existing :class:`TransportError` branch and
    exit 7 with the install-hint message.
    """

    from pd_ocr_synth.publish import SdkUnavailableError

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_value_aaaaaaaaaaaaaaaaaaaa")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    def _raise(_token: str) -> None:
        raise SdkUnavailableError(
            "the Hugging Face SDK is not installed; "
            "install with `pip install pd-ocr-synth[publish]`, or use "
            "--dry-run to preview the upload plan without an SDK"
        )

    monkeypatch.setattr("pd_ocr_synth.publish.cli_runner.make_default_transport", _raise)

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, with_publish=False, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    rc = main(["publish", str(rp), "--repo", "alice/x"])
    err = capsys.readouterr().err
    assert rc == 7
    # The remediation hint should mention either the publish extra or
    # --dry-run so the user knows how to proceed.
    assert "pd-ocr-synth[publish]" in err or "--dry-run" in err


def test_publish_private_and_public_mutually_exclusive(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rp = _setup_recipe(tmp_path, with_publish=False)
    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--private",
            "--public",
            "--dry-run",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "mutually exclusive" in err


# ---------------------------------------------------------------------------
# --output override
# ---------------------------------------------------------------------------


def test_publish_dry_run_honors_output_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``-o PATH`` lets a publish target a one-off render directory
    (handy when you ran ``render -o /tmp/foo`` for a smoke test).
    """

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    # Recipe destination points somewhere that does not exist yet,
    # but we override with -o to a real render.
    real_render = tmp_path / "actual-render"
    rp = _setup_recipe(
        tmp_path,
        with_publish=False,
        dest=tmp_path / "would-be-empty",
    )
    _do_render(rp, real_render)
    capsys.readouterr()

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--output",
            str(real_render),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "Would upload to: alice/x" in captured.out


# ---------------------------------------------------------------------------
# Structured plan content
# ---------------------------------------------------------------------------


def test_publish_dry_run_lists_content_sha_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Content SHA appears in output and matches the manifest summary
    output from the staging build (rows must be > 0)."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, with_publish=False, dest=out)
    _do_render(rp, out)
    # Sanity: render produced some labels.
    labels = json.loads((out / "labels.json").read_text(encoding="utf-8"))
    assert labels  # non-empty
    capsys.readouterr()

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    # Content SHA shown as the 12-char short prefix.
    assert "Content SHA: " in captured.out
    sha_line = next(line for line in captured.out.splitlines() if line.startswith("Content SHA:"))
    sha_short = sha_line.replace("Content SHA: ", "").strip()
    assert len(sha_short) == 12
    assert all(c in "0123456789abcdef" for c in sha_short)

    # Manifest summary present with the rows count from the render.
    assert "Manifest summary:" in captured.out
    assert f"Rows: {len(labels)}" in captured.out


# ---------------------------------------------------------------------------
# --license override (spec 10 § Recipe ``publish:`` block)
# ---------------------------------------------------------------------------


def test_publish_dry_run_license_override_appears_in_card_preview(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec 10: ``--license`` overrides the recipe's
    ``publish.hf_dataset.license`` and must land in the dry-run's
    "Dataset card preview" block — what the user sees here is what
    the real upload would stamp."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    # Recipe declares ``cc-by-4.0``; we override with ``mit``.
    rp = _setup_recipe(tmp_path, with_publish=True, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    rc = main(
        [
            "publish",
            str(rp),
            "--license",
            "mit",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    # The flag wins over the recipe value.
    assert "license: mit" in captured.out
    assert "license: cc-by-4.0" not in captured.out


def test_publish_dry_run_without_license_flag_uses_recipe_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """No ``--license`` → the recipe's declared license still appears
    in the dry-run preview. Locks the "no flag, recipe wins" half of
    the spec-10 precedence rule."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, with_publish=True, dest=out)
    _do_render(rp, out)
    capsys.readouterr()

    rc = main(["publish", str(rp), "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert "license: cc-by-4.0" in captured.out


# ---------------------------------------------------------------------------
# M09: detection-mode dispatch through ``cmd_publish``
# ---------------------------------------------------------------------------


_RECIPE_DETECTION = """\
schema_version: 1
name: publish-detection-smoke
seed: 11
output:
  format: pd-ocr-trainer/v1
  mode: detection
  destination: {dest}
  count: 1
corpus:
  - type: local
    path: ./seed-pages.txt
fonts:
  - path: {font}
    weight: 1.0
rendering:
  font_size_pt: 16
  dpi: 300
  ink_color: {{ r: 10, g: 10, b: 10 }}
  background_color: {{ r: 240, g: 235, b: 220 }}
layout:
  mode: pages
  padding_px: 6
  line_spacing: 1.2
  paragraph_spacing: 1.0
"""

# Single page: two short paragraphs of two lines each. Lines stay
# narrow so we don't need a wrap-fitter under-test here.
_SEED_PAGES = "ḃeaḋ saoġal mór\nċeann an ḃoṫair\n\nḋuine ḟir oġa\nġloine ṁaṫair óg\n"


def _setup_detection_recipe(tmp_path: Path, dest: Path) -> Path:
    font = _require_font()
    rp = tmp_path / "recipe-detection.yaml"
    rp.write_text(_RECIPE_DETECTION.format(font=font, dest=dest), encoding="utf-8")
    (tmp_path / "seed-pages.txt").write_text(_SEED_PAGES, encoding="utf-8")
    return rp


def test_publish_dry_run_dispatches_detection_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Spec 10 § Format conversion — detection: a recipe with
    ``output.mode: detection`` must route through the detection
    staging path, producing a ``data/page_*.png`` layout and a card
    that announces ``pd-ocr-shape: detection/v1``.

    This is the M09 dispatch contract — the CLI runner picks the
    right staging builder for the recipe's mode rather than blindly
    assuming recognition.
    """

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out-detection"
    rp = _setup_detection_recipe(tmp_path, dest=out)

    # Render the detection-mode output. The detection writer drops
    # ``images/page_*.png`` + ``labels.json`` per spec 08.
    rc = main(
        [
            "render",
            str(rp),
            "--count",
            "1",
            "--output",
            str(out),
            "--seed",
            "11",
            "--workers",
            "1",
        ]
    )
    assert rc == 0
    capsys.readouterr()

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/pd-ocr-synth-detection-smoke",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err

    # The dry-run preview shows the detection-shape front matter.
    assert "pd-ocr-shape: detection/v1" in captured.out
    # Recognition's task category should NOT appear (we'd be lying
    # about the shape if it did).
    assert "text-recognition" not in captured.out
    # Detection's task category should.
    assert "object-detection" in captured.out
