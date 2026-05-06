"""End-to-end tests for ``pd-ocr-synth publish --render-first`` (M08).

Spec 10 § When to publish: "Pass ``--render-first`` to chain them."
Spec 10 § Errors and recovery: missing local render → exit 5 with
"render first or use --render-first".

Implementation lives in ``pd_ocr_synth.publish.cli_runner.cmd_publish``
behind an injectable ``render_first_callable`` parameter, with a
production default that delegates to
:func:`pd_ocr_synth.render.run_recipe`. Tests inject a fake render
callable so they exercise the chaining contract without paying the
real-rendering cost.

Cases covered:

- Without ``--render-first``: the render callable is NOT invoked,
  even if the local output is missing — the missing-output guard
  fires (exit 5) and the user is pointed at the documented
  remediation.
- With ``--render-first``: the render callable IS invoked, populating
  the local output dir; the staging build then runs against that
  output and the publish step (here a dry-run) succeeds with exit 0.
- Render failure short-circuits with exit 5 (RENDER_EXIT) and the
  publish step is NOT reached. This keeps render failures distinct
  from publish-family failures (exit 7 per spec 01).
- ``--render-first`` + ``--dry-run`` is a valid combo: render first,
  then dry-run preview against the freshly-rendered output.
- The default (production) render callable lazy-imports the render
  pipeline; we don't exercise the real run_recipe in these tests
  (covered by M07 tests) but we do verify the wiring by patching
  ``_default_render_first`` on a no-flag invocation and asserting it
  is NOT called.

The fake render callable here writes the same minimal recognition
layout the publish staging builder expects (``images/``,
``labels.json``, ``manifest.jsonl``, ``recipe.snapshot.yaml``).
That mirrors what M07's ``RecognitionWriter`` produces, just enough
for the staging step to succeed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from pd_ocr_synth.cli import main

# ---------------------------------------------------------------------------
# Recipe + fake-render fixtures
# ---------------------------------------------------------------------------


_RECIPE_TEMPLATE = """\
schema_version: 1
name: render-first-smoke
seed: 11
output:
  format: pd-ocr-trainer/v1
  mode: recognition
  destination: {dest}
  count: 2
corpus:
  - type: local
    path: ./seed-words.txt
fonts:
  - path: ./fake.otf
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


def _setup_recipe(tmp_path: Path, dest: Path) -> Path:
    """Materialize a recipe + seed-words file pointing at ``dest``.

    No font is actually required because the real render is replaced
    by a fake callable. The recipe's ``fonts:`` block must still parse
    against the schema, hence the ``./fake.otf`` placeholder; we don't
    invoke the real validator (which would warn about the missing
    file) — the publish CLI loads the recipe via ``load_recipe`` only.
    """

    rp = tmp_path / "recipe.yaml"
    rp.write_text(_RECIPE_TEMPLATE.format(dest=dest), encoding="utf-8")
    (tmp_path / "seed-words.txt").write_text("hello\nworld\n", encoding="utf-8")
    # Touch the placeholder font path so any incidental existence
    # checks don't flake. The publish path itself never opens it.
    (tmp_path / "fake.otf").write_bytes(b"")
    return rp


def _fake_render_layout(output_dir: Path, *, samples: int = 2) -> None:
    """Drop a minimal but valid recognition layout in ``output_dir``.

    Mirrors what ``RecognitionWriter`` produces for two rendered
    samples: ``images/0000000.png``, ``images/0000001.png``,
    ``labels.json``, ``manifest.jsonl``, ``recipe.snapshot.yaml``,
    ``stats.json``. Just enough for ``build_recognition_staging`` to
    succeed; we do NOT verify image pixels here — the publish layer
    treats them as opaque blobs.
    """

    images = output_dir / "images"
    images.mkdir(parents=True, exist_ok=True)
    labels: dict[str, str] = {}
    manifest_rows: list[dict[str, object]] = []
    for idx in range(samples):
        name = f"{idx:07d}.png"
        Image.new("RGB", (8, 8), color=(200, 200, 200)).save(images / name, format="PNG")
        labels[name] = f"word{idx}"
        manifest_rows.append(
            {
                "index": idx,
                "id": Path(name).stem,
                "image": f"images/{name}",
                "text": f"word{idx}",
                "status": "rendered",
                "font": {
                    "name": "fake.otf",
                    "path": "/abs/fake.otf",
                    "size_pt": 14.0,
                },
            }
        )

    (output_dir / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in manifest_rows) + "\n", encoding="utf-8"
    )
    (output_dir / "recipe.snapshot.yaml").write_text(
        "tool_version: 0.0.0\nseed: 11\n", encoding="utf-8"
    )
    (output_dir / "stats.json").write_text("{}\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Wiring: the flag is plumbed; default callable is NOT invoked without it
# ---------------------------------------------------------------------------


def test_publish_without_render_first_does_not_invoke_render_callable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Without the flag, render-first is dormant — even if the user's
    existing render is in place, we never re-invoke render. This locks
    the "publish does not re-render" half of spec 10 § When to publish.
    """

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    # Pre-populate the layout as if a prior `render` ran.
    _fake_render_layout(out)

    invocations: list[Path] = []

    def _spy(recipe_path: Path, output_dir: Path, cache_dir: Path | None) -> None:
        invocations.append(output_dir)

    monkeypatch.setattr("pd_ocr_synth.publish.cli_runner._default_render_first", _spy)

    rc = main(["publish", str(rp), "--repo", "alice/x", "--dry-run"])
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    # Without --render-first, the spy is never reached.
    assert invocations == []


# ---------------------------------------------------------------------------
# Happy path: --render-first invokes render, then publish proceeds
# ---------------------------------------------------------------------------


def test_publish_render_first_invokes_render_then_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--render-first`` creates the local layout the publish step
    needs. After the chain, the publish dry-run prints its plan as
    usual."""

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)
    # NOTE: out is intentionally NOT pre-populated. --render-first
    # must produce it, otherwise publish exits 5.

    invocations: list[tuple[Path, Path]] = []

    def _fake_render(recipe_path: Path, output_dir: Path, cache_dir: Path | None) -> None:
        invocations.append((recipe_path, output_dir))
        _fake_render_layout(output_dir)

    monkeypatch.setattr("pd_ocr_synth.publish.cli_runner._default_render_first", _fake_render)

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--render-first",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    # Render was called exactly once, against the publish dispatch's
    # resolved output_dir.
    assert len(invocations) == 1
    called_recipe_path, called_out = invocations[0]
    assert called_recipe_path == rp
    assert called_out == out
    # Publish dry-run output appears in stdout — the chain reached
    # publish after render.
    assert "Would upload to: alice/x" in captured.out
    assert "Content SHA:" in captured.out


def test_publish_render_first_with_output_override_uses_overridden_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--render-first`` plus ``-o PATH``: render writes to PATH and
    publish reads from PATH, not the recipe's destination. Ensures the
    flag composes with the existing ``-o`` override.
    """

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    recipe_dest = tmp_path / "recipe-dest"
    override = tmp_path / "actual-out"
    rp = _setup_recipe(tmp_path, dest=recipe_dest)

    invocations: list[Path] = []

    def _fake_render(recipe_path: Path, output_dir: Path, cache_dir: Path | None) -> None:
        invocations.append(output_dir)
        _fake_render_layout(output_dir)

    monkeypatch.setattr("pd_ocr_synth.publish.cli_runner._default_render_first", _fake_render)

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--render-first",
            "--output",
            str(override),
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0, captured.err
    assert invocations == [override]
    # Recipe-default destination was bypassed (still doesn't exist).
    assert not recipe_dest.exists()


# ---------------------------------------------------------------------------
# Failure path: render error short-circuits with exit 5
# ---------------------------------------------------------------------------


def test_publish_render_first_render_failure_short_circuits_with_exit_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the render step fails, the runner must NOT proceed to
    publish; exit code is RENDER_EXIT (5) per spec 01, distinct from
    publish-family failures (exit 7).
    """

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)

    def _failing_render(recipe_path: Path, output_dir: Path, cache_dir: Path | None) -> None:
        raise RuntimeError("simulated render failure")

    monkeypatch.setattr("pd_ocr_synth.publish.cli_runner._default_render_first", _failing_render)

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--render-first",
            "--dry-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 5
    assert "render failed" in captured.err.lower()
    # The error message names the underlying failure so the user can
    # debug it.
    assert "simulated render failure" in captured.err
    # Publish step never ran → no "Would upload" line in stdout.
    assert "Would upload" not in captured.out


def test_publish_render_first_render_writes_to_wrong_path_exits_five(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """If the render callable claims success but doesn't actually
    populate the expected directory, publish must still fail loudly
    rather than try to stage from an empty path. The exit code is 5
    (the spec's "missing local render" — same family) and the message
    flags the post-render miss explicitly.
    """

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)

    def _silent_no_op(recipe_path: Path, output_dir: Path, cache_dir: Path | None) -> None:
        # Returns successfully without creating output_dir.
        return None

    monkeypatch.setattr("pd_ocr_synth.publish.cli_runner._default_render_first", _silent_no_op)

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--render-first",
            "--dry-run",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 5
    # Hint distinguishes "after --render-first" vs the bare
    # missing-render case so the user knows to look upstream.
    assert "after --render-first" in err.lower()


# ---------------------------------------------------------------------------
# --render-first composes with the real-upload (non-dry-run) path
# ---------------------------------------------------------------------------


def test_publish_render_first_chains_into_real_upload_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """End-to-end: ``--render-first`` + a real (non-dry-run) publish.
    We replace the SDK transport factory with one that raises
    ``SdkUnavailableError`` so the test stays hermetic — the exit code
    is 7 (publish-family failure, same as the existing
    ``test_publish_real_upload_with_sdk_unavailable_exits_seven``),
    NOT 5. This locks the ordering: render runs first; only after
    render succeeds do we reach the upload step where the SDK error
    fires.
    """

    from pd_ocr_synth.publish import SdkUnavailableError

    monkeypatch.setenv("HF_TOKEN", "hf_test_token_value_aaaaaaaaaaaaaaaaaaaa")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-home"))

    out = tmp_path / "trainer-out"
    rp = _setup_recipe(tmp_path, dest=out)

    rendered: list[Path] = []

    def _fake_render(recipe_path: Path, output_dir: Path, cache_dir: Path | None) -> None:
        rendered.append(output_dir)
        _fake_render_layout(output_dir)

    def _no_sdk(_token: str) -> None:
        raise SdkUnavailableError("sdk not installed (test fake)")

    monkeypatch.setattr("pd_ocr_synth.publish.cli_runner._default_render_first", _fake_render)
    monkeypatch.setattr("pd_ocr_synth.publish.cli_runner.make_default_transport", _no_sdk)

    rc = main(
        [
            "publish",
            str(rp),
            "--repo",
            "alice/x",
            "--render-first",
        ]
    )
    err = capsys.readouterr().err
    # Render WAS called (proves the flag fires before the upload
    # branch); the upload then exits 7 because the SDK fake refuses
    # to construct a transport.
    assert rendered == [out]
    assert rc == 7
    assert "sdk not installed" in err.lower()
