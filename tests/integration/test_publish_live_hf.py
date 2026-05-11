"""Live end-to-end Hugging Face publish test (M08, opt-in).

This is the real-network counterpart to the in-process publish-flow
tests in ``tests/test_publish_orchestrator.py`` and
``tests/test_cli_publish_upload.py``. It exercises the **production**
code path:

    build_recognition_staging
        → publish_recognition(transport=make_default_transport(token))
        → assert (created | uploaded), README has pd-ocr-* keys
        → publish_recognition again
        → assert NO_CHANGE (idempotent no-op)
        → finally: HfApi.delete_repo (cleanup)

Tracked under "Residual M08 work → End-to-end live HF test" in
``docs/roadmap/08-publishing-hf.md``. The corresponding spec entry is
``docs/specs/10-publishing.md`` § Tests ("End-to-end test against a
private 'scratch' repo on HF (gated by an HF_TOKEN env var; skipped on
CI without secrets)").

How to run
----------
The default ``make ci`` (and any plain ``pytest`` invocation) skips
this test. To run it live::

    export PD_OCR_SYNTH_HF_E2E=1
    export HF_TOKEN=hf_...                      # write scope on the namespace
    # optional — overrides the default repo name:
    export PD_OCR_SYNTH_HF_E2E_REPO=me/pd-ocr-synth-livetest-recognition
    uv run pytest -m integration tests/integration/test_publish_live_hf.py -v

Env-var contract
~~~~~~~~~~~~~~~~
``PD_OCR_SYNTH_HF_E2E``
    Master switch. Anything truthy (``"1"``, ``"true"``, etc.) enables
    the test. Unset / empty / ``"0"`` skips. Name matches the spec's
    "gated by an env var" requirement and is namespaced so a sibling
    pd-* repo can reuse the same ``HF_TOKEN`` without colliding.
``HF_TOKEN``
    Required. The token must have **write** scope on the namespace
    used by ``PD_OCR_SYNTH_HF_E2E_REPO``. Resolved via the standard
    publish auth chain (``--token`` > ``HF_TOKEN`` >
    ``~/.cache/huggingface/token``); we only support the env-var slot
    here because there's no CLI flag in a pytest invocation.
``PD_OCR_SYNTH_HF_E2E_REPO``
    Optional. Canonical ``OWNER/NAME`` for the scratch dataset repo.
    Default: ``ConcaveTrillion/pd-ocr-synth-livetest-recognition``.
    The test creates and **deletes** this repo each run, so point it
    at a throwaway namespace.

Convention for future live tests
--------------------------------
Place opt-in tests under ``tests/integration/``. Mark them with
``@pytest.mark.integration`` and gate with::

    @pytest.mark.skipif(
        not _live_enabled(),
        reason="set PD_OCR_SYNTH_HF_E2E=1 + HF_TOKEN to run live",
    )

so they collect cleanly under ``make ci`` (visible as skips, not
errors). The env-var prefix ``PD_OCR_SYNTH_<SUITE>_…`` keeps each
opt-in suite self-contained.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from PIL import Image

from pd_ocr_synth.publish.orchestrator import (
    PublishResult,
    PublishState,
    publish_recognition,
)
from pd_ocr_synth.publish.preflight import (
    REQUIRED_FRONT_MATTER_KEYS,
    check_required_front_matter,
)
from pd_ocr_synth.publish.recognition import build_recognition_staging
from pd_ocr_synth.publish.sdk_transport import make_default_transport

# ---------------------------------------------------------------------------
# Gating helpers
# ---------------------------------------------------------------------------


_TRUTHY = frozenset({"1", "true", "True", "TRUE", "yes", "on"})

_DEFAULT_REPO = "ConcaveTrillion/pd-ocr-synth-livetest-recognition"


def _live_enabled() -> bool:
    """Return True iff both the master switch and a token are set.

    A truthy ``PD_OCR_SYNTH_HF_E2E`` *without* an ``HF_TOKEN`` would
    fail at auth resolution; we treat that as "not configured" and
    skip rather than fail so misconfigured local runs don't look like
    the test broke.
    """

    if os.environ.get("PD_OCR_SYNTH_HF_E2E", "") not in _TRUTHY:
        return False
    return bool(os.environ.get("HF_TOKEN"))


def _resolve_repo_id() -> str:
    return os.environ.get("PD_OCR_SYNTH_HF_E2E_REPO", _DEFAULT_REPO)


# ---------------------------------------------------------------------------
# Fixture: tiny local recognition output (no recipe pipeline)
# ---------------------------------------------------------------------------


def _write_local_output(local: Path) -> None:
    """Materialize a 2-sample local recognition layout.

    Mirrors the helper in ``tests/test_publish_orchestrator.py`` — we
    deliberately do **not** drive ``run_recipe`` here so the live test
    stays fast (a few KB on the wire) and isolated from rendering
    correctness, which has its own coverage.
    """

    images = local / "images"
    images.mkdir(parents=True, exist_ok=True)

    labels: dict[str, str] = {}
    manifest_lines: list[str] = []
    for idx, text in enumerate(["Séadna", "agus"]):
        name = f"{idx:07d}.png"
        Image.new("RGB", (8, 8), color=(200, 200, 200)).save(images / name, format="PNG")
        labels[name] = text
        manifest_lines.append(
            json.dumps(
                {
                    "index": idx,
                    "id": Path(name).stem,
                    "image": f"images/{name}",
                    "text": text,
                    "status": "rendered",
                    "font": {"name": "bungc.otf", "size_pt": 14.0},
                }
            )
        )

    (local / "labels.json").write_text(
        json.dumps(labels, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (local / "manifest.jsonl").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    (local / "recipe.snapshot.yaml").write_text(
        "tool_version: 0.0.0-livetest\nseed: 5\n",
        encoding="utf-8",
    )
    (local / "stats.json").write_text("{}\n", encoding="utf-8")


@pytest.fixture
def _live_repo() -> Iterator[str]:
    """Yield the live test repo id and best-effort delete it afterwards.

    Cleanup uses ``HfApi.delete_repo`` directly (not the transport
    Protocol) — the Protocol is intentionally narrow (see
    ``publish/transport.py`` § "What's *not* on the Protocol") and a
    delete is a test-only concern. We swallow cleanup errors so a
    transient HF outage at teardown doesn't mask a real test failure.
    """

    repo_id = _resolve_repo_id()
    yield repo_id

    token = os.environ.get("HF_TOKEN")
    if not token:
        return
    try:
        from huggingface_hub import HfApi

        HfApi(token=token).delete_repo(repo_id, repo_type="dataset", missing_ok=True)
    except Exception:
        # Intentionally silent: a failed cleanup is a manual janitor
        # job, not a test failure. The repo is small + named so it's
        # easy to spot in the namespace if it lingers.
        pass


# ---------------------------------------------------------------------------
# The actual live test
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not _live_enabled(),
    reason=(
        "live HF E2E test is opt-in: set PD_OCR_SYNTH_HF_E2E=1 and HF_TOKEN (write scope) to run"
    ),
)
def test_live_publish_create_then_idempotent_no_op(
    tmp_path: Path,
    _live_repo: str,
) -> None:
    """End-to-end: build staging → publish → re-publish is a no-op.

    Asserts the spec 10 contract end-to-end against a real HF dataset
    repo:

    1. First publish either CREATES the repo (cold start) or UPLOADS
       to an existing repo (a previous live run lingered). Both are
       valid first-call outcomes; we don't assume which.
    2. The staged README's front matter carries every required
       ``pd-ocr-*`` key plus ``pd-ocr-content-sha`` (per
       ``REQUIRED_FRONT_MATTER_KEYS`` + the post-build SHA pin).
    3. Re-running ``publish_recognition`` with the same staging dir
       returns ``PublishState.NO_CHANGE`` — the spec's "exit 0 with
       'no changes' and do not commit" branch.
    """

    token = os.environ["HF_TOKEN"]  # _live_enabled() guarantees this
    repo_id = _live_repo

    # 1. Build a local recognition output + staging dir.
    local = tmp_path / "local"
    _write_local_output(local)
    staging = tmp_path / "staging"
    staging_result = build_recognition_staging(local, staging)
    assert staging_result.readme_written, (
        "staging build did not write a README — content-SHA pin would "
        "be skipped and the publish would fail pre-flight"
    )
    assert staging_result.content_sha is not None
    expected_sha = staging_result.content_sha

    # 2. Front-matter sanity — locks the spec 10 § Dataset card contract.
    #    Drive through the public preflight helper so the live test fails
    #    the same way the orchestrator's own pre-flight would, were the
    #    builder ever to regress on a required key.
    report = check_required_front_matter(staging)
    assert not report.missing_keys, f"staged README missing keys: {report.missing_keys}"
    assert not report.empty_keys, f"staged README has empty keys: {report.empty_keys}"
    fm = report.front_matter
    for key in REQUIRED_FRONT_MATTER_KEYS:
        assert key in fm, f"staged README missing required front-matter key {key!r}"
    assert fm.get("pd-ocr-content-sha") == expected_sha, (
        "staged README's pd-ocr-content-sha must match the value "
        "compute_content_sha returned during the build"
    )

    # 3. First publish — real network. We don't pin which terminal
    #    state we expect because a prior live run might have lingered;
    #    both CREATED and UPLOADED are valid first-call outcomes.
    transport = make_default_transport(token)
    first: PublishResult = publish_recognition(
        transport,
        repo_id,
        staging,
        commit_message="pd-ocr-synth livetest @initial",
    )
    assert first.state in {PublishState.CREATED, PublishState.UPLOADED}, (
        f"unexpected first-publish state {first.state!r}; expected CREATED or UPLOADED"
    )
    assert first.repo_id == repo_id
    assert first.content_sha == expected_sha
    assert first.commit_sha != "", "real upload must return a non-empty commit SHA"

    # 4. Re-publish without local changes — must be a no-op.
    second: PublishResult = publish_recognition(
        transport,
        repo_id,
        staging,
        commit_message="pd-ocr-synth livetest @should-not-fire",
    )
    assert second.state is PublishState.NO_CHANGE, (
        f"second publish was {second.state!r}; expected NO_CHANGE "
        "(idempotency check should match the remote pd-ocr-content-sha)"
    )
    assert second.is_no_change
    assert second.content_sha == expected_sha
    assert second.commit_sha == "", (
        "NO_CHANGE branch must not synthesize a commit SHA; see PublishResult.commit_sha contract"
    )


# ---------------------------------------------------------------------------
# Collection sanity (always runs, no network)
# ---------------------------------------------------------------------------


def test_live_test_skipped_without_env() -> None:
    """Belt-and-braces: prove the gating helper returns False in default CI.

    Without this, a refactor that accidentally inverts ``_live_enabled``
    (or hardcodes the truthy set wrong) would only get caught in a live
    run — at which point the live test would *try* to run with no token.
    Cheap to assert here; runs as a normal unit test under ``make ci``.
    """

    # Save and clear the env so we're testing the helper, not the
    # environment we happen to inherit. We restore in finally so a
    # developer running with PD_OCR_SYNTH_HF_E2E set locally still gets
    # their live run in the next test.
    saved = {key: os.environ.get(key) for key in ("PD_OCR_SYNTH_HF_E2E", "HF_TOKEN")}
    try:
        os.environ.pop("PD_OCR_SYNTH_HF_E2E", None)
        os.environ.pop("HF_TOKEN", None)
        assert _live_enabled() is False

        # Switch on without a token — still skipped.
        os.environ["PD_OCR_SYNTH_HF_E2E"] = "1"
        assert _live_enabled() is False

        # Token without the master switch — still skipped.
        os.environ.pop("PD_OCR_SYNTH_HF_E2E", None)
        os.environ["HF_TOKEN"] = "hf_dummy"
        assert _live_enabled() is False

        # Both set — enabled.
        os.environ["PD_OCR_SYNTH_HF_E2E"] = "1"
        os.environ["HF_TOKEN"] = "hf_dummy"
        assert _live_enabled() is True

        # Falsy master-switch values stay disabled.
        for falsy in ("", "0", "false", "no"):
            os.environ["PD_OCR_SYNTH_HF_E2E"] = falsy
            assert _live_enabled() is False, f"value {falsy!r} should disable"
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_live_repo_id_default_is_namespaced() -> None:
    """The default repo id is in a known namespace so a misconfigured
    run can't accidentally write into a third-party namespace."""

    saved = os.environ.pop("PD_OCR_SYNTH_HF_E2E_REPO", None)
    try:
        repo = _resolve_repo_id()
        assert repo == _DEFAULT_REPO
        owner, _, name = repo.partition("/")
        assert owner and name, "default repo id must be canonical OWNER/NAME"
        assert "livetest" in name, (
            "default repo name must mention 'livetest' so a stray repo "
            "in the owner namespace is obviously the test fixture"
        )
    finally:
        if saved is not None:
            os.environ["PD_OCR_SYNTH_HF_E2E_REPO"] = saved


def test_resolve_repo_id_honors_env_override() -> None:
    """Custom ``PD_OCR_SYNTH_HF_E2E_REPO`` wins over the default."""

    saved = os.environ.get("PD_OCR_SYNTH_HF_E2E_REPO")
    try:
        os.environ["PD_OCR_SYNTH_HF_E2E_REPO"] = "alice/scratch-pd-ocr"
        assert _resolve_repo_id() == "alice/scratch-pd-ocr"
    finally:
        if saved is None:
            os.environ.pop("PD_OCR_SYNTH_HF_E2E_REPO", None)
        else:
            os.environ["PD_OCR_SYNTH_HF_E2E_REPO"] = saved
