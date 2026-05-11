"""Unit tests for ``pd_ocr_synth.publish.orchestrator`` (M08).

Drives the full upload sequence — pre-flight, content-SHA, idempotency,
``create_repo`` (when needed), ``upload_folder``, optional
``create_tag`` — against :class:`FakeTransport`. No network, no
``huggingface_hub`` import. Tests assert both on the returned
:class:`PublishResult` *and* on the operation order recorded by the
fake transport's ``calls`` list — the latter locks the spec sequence
("idempotency check before create; create before upload; tag after
upload") so a refactor that subtly reorders the orchestrator can't
ship green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from pd_ocr_synth.publish.content_sha import (
    CONTENT_SHA_KEY,
    apply_content_sha_to_readme,
    compute_content_sha,
)
from pd_ocr_synth.publish.orchestrator import (
    PublishError,
    PublishResult,
    PublishState,
    publish_recognition,
)
from pd_ocr_synth.publish.recognition import build_recognition_staging
from pd_ocr_synth.publish.transport import (
    FakeTransport,
    TransportError,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_local_output(local: Path) -> None:
    """Materialize a minimal but complete recognition local layout.

    Mirrors the helper in ``test_publish_recognition.py`` but slimmed
    down to two samples — these tests don't care about provenance
    columns, only the upload flow.
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
        "tool_version: 0.0.0\nseed: 5\n",
        encoding="utf-8",
    )
    (local / "stats.json").write_text("{}\n", encoding="utf-8")


def _build_staging(tmp_path: Path) -> Path:
    """Build a real, preflight-passing staging dir for upload tests."""

    local = tmp_path / "local"
    _write_local_output(local)
    staging = tmp_path / "staging"
    result = build_recognition_staging(local, staging)
    assert result.readme_written, "fixture invariant: README must be present for upload"
    return staging


def _op_names(transport: FakeTransport) -> list[str]:
    return [name for name, _ in transport.calls]


# ---------------------------------------------------------------------------
# Happy path: first publish (CREATED state)
# ---------------------------------------------------------------------------


def test_first_publish_creates_repo_and_uploads(tmp_path: Path) -> None:
    """The repo doesn't exist; ``allow_create=True`` (default) means
    the orchestrator creates it and uploads. State is CREATED."""

    transport = FakeTransport()
    staging = _build_staging(tmp_path)

    result = publish_recognition(
        transport,
        "alice/x",
        staging,
        commit_message="pd-ocr-synth render @abc123",
    )

    assert isinstance(result, PublishResult)
    assert result.state is PublishState.CREATED
    assert result.repo_id == "alice/x"
    assert result.commit_sha != ""
    assert result.commit_url.startswith("https://huggingface.co/datasets/alice/x/")
    assert result.tag is None
    # Content-SHA was actually computed (non-empty hex digest).
    assert len(result.content_sha) == 64

    # Operation sequence: existence probe, missing → create_repo,
    # then upload. No card-data read on the missing-repo branch.
    assert _op_names(transport) == ["repo_exists", "create_repo", "upload_folder"]

    # The repo now exists in the fake with our staged files.
    repo = transport.repos["alice/x"]
    assert "README.md" in repo.files
    assert "metadata.jsonl" in repo.files
    assert "data/0000000.png" in repo.files


def test_first_publish_honors_private_flag(tmp_path: Path) -> None:
    """``private=True`` must propagate to ``create_repo``."""

    transport = FakeTransport()
    staging = _build_staging(tmp_path)

    publish_recognition(
        transport,
        "alice/x",
        staging,
        commit_message="initial",
        private=True,
    )

    create_call = next(call for call in transport.calls if call[0] == "create_repo")
    assert create_call[1]["private"] is True
    assert transport.repos["alice/x"].private is True


# ---------------------------------------------------------------------------
# Happy path: re-publish with changes (UPLOADED state)
# ---------------------------------------------------------------------------


def test_republish_with_changed_content_uploads_without_creating(tmp_path: Path) -> None:
    """A repo that already exists but carries a stale content-SHA gets
    a plain upload — no ``create_repo`` call."""

    transport = FakeTransport()
    transport.seed_repo(
        "alice/x",
        card_data={CONTENT_SHA_KEY: "0" * 64, "license": "cc-by-4.0"},
    )
    staging = _build_staging(tmp_path)

    result = publish_recognition(
        transport,
        "alice/x",
        staging,
        commit_message="re-publish",
    )

    assert result.state is PublishState.UPLOADED
    assert result.commit_sha != ""

    op_names = _op_names(transport)
    assert "create_repo" not in op_names
    # Sequence: repo_exists → read_remote_card_data → upload_folder.
    assert op_names == ["repo_exists", "read_remote_card_data", "upload_folder"]


# ---------------------------------------------------------------------------
# Idempotency short-circuit (NO_CHANGE state)
# ---------------------------------------------------------------------------


def test_republish_with_matching_sha_is_no_op(tmp_path: Path) -> None:
    """Spec 10 § Idempotency: matching content-SHA → exit 0 with no
    upload. The orchestrator must not call ``upload_folder``."""

    staging = _build_staging(tmp_path)
    # Recompute the SHA the same way the orchestrator will: this is
    # what the fake transport must already carry to trigger NO_CHANGE.
    local_sha = compute_content_sha(staging)

    transport = FakeTransport()
    transport.seed_repo(
        "alice/x",
        card_data={CONTENT_SHA_KEY: local_sha, "license": "cc-by-4.0"},
    )

    result = publish_recognition(
        transport,
        "alice/x",
        staging,
        commit_message="should-not-be-used",
    )

    assert result.state is PublishState.NO_CHANGE
    assert result.is_no_change is True
    assert result.commit_sha == ""
    assert result.commit_url == ""
    assert result.tag is None
    assert result.content_sha == local_sha

    # The fake transport's ``upload_folder`` was never called.
    assert "upload_folder" not in _op_names(transport)
    # And no repo was created (existing one was seeded via seed_repo).
    assert "create_repo" not in _op_names(transport)


def test_round_trip_second_publish_is_no_op(tmp_path: Path) -> None:
    """The realistic round-trip: first publish creates+uploads; an
    immediate re-publish over an unchanged staging dir is NO_CHANGE.
    This is the exact CI loop spec 10 promises ('safe to invoke from
    CI on every run')."""

    transport = FakeTransport()
    staging = _build_staging(tmp_path)

    first = publish_recognition(transport, "alice/x", staging, commit_message="initial")
    second = publish_recognition(transport, "alice/x", staging, commit_message="should-not-fire")

    assert first.state is PublishState.CREATED
    assert second.state is PublishState.NO_CHANGE
    assert second.content_sha == first.content_sha

    # Only one upload commit on the repo, despite two publish calls.
    assert len(transport.repos["alice/x"].commits) == 1


# ---------------------------------------------------------------------------
# allow_create gating
# ---------------------------------------------------------------------------


def test_missing_repo_with_allow_create_false_raises(tmp_path: Path) -> None:
    """Spec 10 § Errors and recovery: ``--no-create`` on a missing
    repo must fail (mapped to exit 7 by the runner)."""

    transport = FakeTransport()
    staging = _build_staging(tmp_path)

    with pytest.raises(PublishError, match="does not exist and allow_create=False"):
        publish_recognition(
            transport,
            "alice/x",
            staging,
            commit_message="initial",
            allow_create=False,
        )

    # We did NOT proceed to create or upload.
    op_names = _op_names(transport)
    assert "create_repo" not in op_names
    assert "upload_folder" not in op_names


def test_existing_repo_with_allow_create_false_still_uploads(tmp_path: Path) -> None:
    """``allow_create=False`` only blocks the missing-repo path; an
    existing repo with stale content still uploads."""

    transport = FakeTransport()
    transport.seed_repo("alice/x", card_data={CONTENT_SHA_KEY: "0" * 64})
    staging = _build_staging(tmp_path)

    result = publish_recognition(
        transport,
        "alice/x",
        staging,
        commit_message="re-publish",
        allow_create=False,
    )

    assert result.state is PublishState.UPLOADED


# ---------------------------------------------------------------------------
# Tagging
# ---------------------------------------------------------------------------


def test_tag_is_created_after_upload(tmp_path: Path) -> None:
    """``--tag v1`` should call ``create_tag`` *after* a successful
    upload, never before."""

    transport = FakeTransport()
    staging = _build_staging(tmp_path)

    result = publish_recognition(
        transport,
        "alice/x",
        staging,
        commit_message="initial",
        tag="v2026.05.06",
    )

    assert result.tag == "v2026.05.06"
    op_names = _op_names(transport)
    # ``create_tag`` lands strictly after ``upload_folder`` — never
    # before. Locking the order rules out a future refactor that
    # tags first and then uploads (which would tag a missing commit).
    upload_idx = op_names.index("upload_folder")
    tag_idx = op_names.index("create_tag")
    assert tag_idx > upload_idx
    assert "v2026.05.06" in transport.repos["alice/x"].tags


def test_no_tag_means_no_create_tag_call(tmp_path: Path) -> None:
    """Default ``tag=None`` must not invoke ``create_tag`` at all."""

    transport = FakeTransport()
    staging = _build_staging(tmp_path)

    result = publish_recognition(transport, "alice/x", staging, commit_message="initial")

    assert result.tag is None
    assert "create_tag" not in _op_names(transport)


def test_tag_is_skipped_on_no_change(tmp_path: Path) -> None:
    """A NO_CHANGE result must NOT create a tag — there's no new
    commit to tag, and re-tagging the existing commit would be a
    side-effect users don't expect from an "exit 0 no changes" flow.
    """

    staging = _build_staging(tmp_path)
    local_sha = compute_content_sha(staging)

    transport = FakeTransport()
    transport.seed_repo(
        "alice/x",
        card_data={CONTENT_SHA_KEY: local_sha},
    )

    result = publish_recognition(
        transport,
        "alice/x",
        staging,
        commit_message="should-not-be-used",
        tag="v1",
    )

    assert result.state is PublishState.NO_CHANGE
    assert result.tag is None
    assert "create_tag" not in _op_names(transport)


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_transport_error_during_upload_propagates(tmp_path: Path) -> None:
    """A mid-upload network failure surfaces as TransportError; the
    runner maps it to exit 7. We do NOT swallow / repackage it.
    """

    transport = FakeTransport(raise_on_upload=True)
    staging = _build_staging(tmp_path)

    with pytest.raises(TransportError, match="raise_on_upload"):
        publish_recognition(transport, "alice/x", staging, commit_message="initial")


def test_preflight_error_propagates_when_staging_lacks_readme(tmp_path: Path) -> None:
    """A staging dir that doesn't pass preflight (e.g. missing README)
    must raise before any transport call lands. Tests this by deleting
    the README the staging builder produced, then re-running the
    orchestrator.
    """

    from pd_ocr_synth.publish.preflight import PreflightError

    transport = FakeTransport()
    staging = _build_staging(tmp_path)
    (staging / "README.md").unlink()

    with pytest.raises(PreflightError):
        publish_recognition(transport, "alice/x", staging, commit_message="initial")

    # No transport calls at all — preflight runs before any.
    assert transport.calls == []


def test_orchestrator_ignores_pinned_readme_sha_value_for_recompute(
    tmp_path: Path,
) -> None:
    """The README's ``pd-ocr-content-sha`` line is *informational*; it
    doesn't feed back into the digest. The orchestrator recomputes
    the digest from the staging dir (with that line stripped before
    hashing — see ``compute_content_sha``'s idempotency contract) and
    compares against the remote.

    Verify by tampering with the README's pinned SHA: the recompute
    must still produce the canonical digest, so a remote that carries
    that canonical digest still triggers NO_CHANGE despite the
    hand-edit. This locks the "the SHA line in the README is a
    side-effect, not an input" property the orchestrator's
    idempotency loop depends on.
    """

    transport = FakeTransport()
    staging = _build_staging(tmp_path)
    correct_sha = compute_content_sha(staging)

    # Tamper: rewrite the README to claim a bogus SHA. Since the
    # SHA line is stripped before hashing, the digest stays the same.
    apply_content_sha_to_readme(staging, "f" * 64)
    assert compute_content_sha(staging) == correct_sha

    # Seed the fake with the canonical SHA — the orchestrator should
    # still recognize NO_CHANGE despite the lying README.
    transport.seed_repo(
        "alice/x",
        card_data={CONTENT_SHA_KEY: correct_sha},
    )

    result = publish_recognition(transport, "alice/x", staging, commit_message="should-not-be-used")

    assert result.state is PublishState.NO_CHANGE
    assert result.content_sha == correct_sha
    # And the digest is NOT the bogus value pinned in the README.
    assert result.content_sha != "f" * 64
    # No upload happened.
    assert "upload_folder" not in _op_names(transport)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_orchestrator_names_reexported_from_publish_package() -> None:
    """Down-stream code (the CLI runner) imports from
    ``pd_ocr_synth.publish``; make sure the new names land there."""

    from pd_ocr_synth import publish

    assert publish.publish_recognition is publish_recognition
    assert publish.PublishResult is PublishResult
    assert publish.PublishState is PublishState
    assert publish.PublishError is PublishError


def test_publish_state_values_are_string_compatible() -> None:
    """StrEnum compatibility, mirroring IdempotencyState."""

    assert PublishState.NO_CHANGE == "no_change"
    assert PublishState.CREATED == "created"
    assert PublishState.UPLOADED == "uploaded"


def test_publish_result_is_frozen(tmp_path: Path) -> None:
    """The runner logs results; freezing rules out a subtle log-then-
    mutate-then-re-read bug.
    """

    transport = FakeTransport()
    staging = _build_staging(tmp_path)
    result = publish_recognition(transport, "alice/x", staging, commit_message="initial")

    with pytest.raises(AttributeError):
        result.state = PublishState.NO_CHANGE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Defensive: commit_message is forwarded verbatim
# ---------------------------------------------------------------------------


def test_commit_message_is_forwarded_verbatim(tmp_path: Path) -> None:
    """Spec 10 § Versioning: ``--message`` overrides the auto-generated
    commit message. The orchestrator must forward the caller's string
    unchanged so the CLI's override path actually overrides.
    """

    transport = FakeTransport()
    staging = _build_staging(tmp_path)

    publish_recognition(
        transport,
        "alice/x",
        staging,
        commit_message="custom human message: shipping v2",
    )

    upload_call = next(call for call in transport.calls if call[0] == "upload_folder")
    assert upload_call[1]["commit_message"] == "custom human message: shipping v2"


# ---------------------------------------------------------------------------
# Result fields the runner depends on
# ---------------------------------------------------------------------------


def _result_field_names() -> set[str]:
    return set(PublishResult.__dataclass_fields__)


def test_result_fields_match_runner_contract() -> None:
    """Lock the field set so a future addition that the runner needs
    is intentional. Tests both sides of the contract.
    """

    assert _result_field_names() == {
        "state",
        "repo_id",
        "content_sha",
        "commit_sha",
        "commit_url",
        "tag",
    }


def test_result_carries_repo_id_for_logging(tmp_path: Path) -> None:
    """Mirror :class:`IdempotencyDecision`: echo the repo so log
    aggregators don't have to thread it through separately.
    """

    transport = FakeTransport()
    staging = _build_staging(tmp_path)

    result = publish_recognition(transport, "alice/x", staging, commit_message="initial")

    assert result.repo_id == "alice/x"


# ---------------------------------------------------------------------------
# Determinism: same staging dir → same content_sha across calls
# ---------------------------------------------------------------------------


def test_two_orchestrator_calls_over_unchanged_staging_agree_on_sha(
    tmp_path: Path,
) -> None:
    """The content-SHA is what makes the idempotency loop work; if
    two orchestrator calls over the same staging dir disagreed on the
    digest, NO_CHANGE would be unreachable in real life. Lock the
    determinism here at the orchestrator level (the content_sha
    module already locks it at the leaf level).
    """

    transport_a = FakeTransport()
    transport_b = FakeTransport()
    staging = _build_staging(tmp_path)

    result_a = publish_recognition(transport_a, "alice/x", staging, commit_message="a")
    result_b = publish_recognition(transport_b, "bob/y", staging, commit_message="b")

    assert result_a.content_sha == result_b.content_sha


# ---------------------------------------------------------------------------
# Sequence: idempotency check happens before any state-changing call
# ---------------------------------------------------------------------------


def test_idempotency_check_runs_before_state_changes(tmp_path: Path) -> None:
    """The spec sequence is: check → maybe create → upload. We must
    never call ``create_repo`` or ``upload_folder`` before the
    idempotency check has run, otherwise NO_CHANGE could be missed
    on a re-publish (bug we want to lock out).
    """

    transport = FakeTransport()
    staging = _build_staging(tmp_path)

    publish_recognition(transport, "alice/x", staging, commit_message="initial")

    op_names = _op_names(transport)
    repo_exists_idx = op_names.index("repo_exists")
    # ``create_repo`` and ``upload_folder`` (if present) must come
    # AFTER the existence probe.
    for op in ("create_repo", "upload_folder"):
        if op in op_names:
            assert op_names.index(op) > repo_exists_idx, (
                f"{op} must run after repo_exists; full sequence: {op_names!r}"
            )
