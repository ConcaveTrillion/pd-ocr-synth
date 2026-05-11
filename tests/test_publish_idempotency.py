"""Unit tests for ``pd_ocr_synth.publish.idempotency`` (M08).

Covers the three-step procedure from ``docs/specs/10-publishing.md``
§ Idempotency:

1. (Caller-supplied) compute local content SHA.
2. Read remote ``card_data.pd-ocr-content-sha``.
3. If equal → no-op; otherwise → upload.

We drive every test against :class:`FakeTransport` so the suite stays
network-free. The fake's ``seed_repo`` lets us model "the repo already
has a published commit with content-SHA X" so the up-to-date branch is
exercisable without first running an upload — but we also exercise the
post-upload round-trip path where ``upload_folder`` refreshes
``card_data`` from the staged README, because that's the real-world
shape the runner will hit on a re-run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.publish.content_sha import CONTENT_SHA_KEY
from pd_ocr_synth.publish.idempotency import (
    IdempotencyDecision,
    IdempotencyState,
    check_idempotency,
)
from pd_ocr_synth.publish.transport import FakeTransport, TransportError

_LOCAL_SHA = "deadbeefcafebabe0123456789abcdef0123456789abcdef0123456789abcdef"
_OTHER_SHA = "00000000000000000000000000000000000000000000000000000000feedface"


# ---------------------------------------------------------------------------
# Happy paths — the three states
# ---------------------------------------------------------------------------


def test_repo_missing_when_transport_says_repo_does_not_exist() -> None:
    """First publish: repo doesn't exist yet. The runner falls
    through to ``create_repo`` + first upload."""

    transport = FakeTransport()

    decision = check_idempotency(transport, "alice/x", _LOCAL_SHA)

    assert decision.state is IdempotencyState.REPO_MISSING
    assert decision.repo_id == "alice/x"
    assert decision.local_sha == _LOCAL_SHA
    assert decision.remote_sha is None
    assert decision.is_up_to_date is False


def test_repo_missing_short_circuits_before_read_remote_card() -> None:
    """Defensive: if the repo doesn't exist, we must not waste a
    ``read_remote_card_data`` call (the real SDK would 404). The
    runner ordering is ``repo_exists`` → maybe ``read``, never read
    first."""

    transport = FakeTransport()

    check_idempotency(transport, "alice/x", _LOCAL_SHA)

    op_names = [name for name, _ in transport.calls]
    assert op_names == ["repo_exists"]


def test_up_to_date_when_remote_sha_matches() -> None:
    """The spec's "exit 0 with no changes" branch: remote card has
    a ``pd-ocr-content-sha`` equal to the local digest."""

    transport = FakeTransport()
    transport.seed_repo(
        "alice/x",
        card_data={CONTENT_SHA_KEY: _LOCAL_SHA, "license": "cc-by-4.0"},
    )

    decision = check_idempotency(transport, "alice/x", _LOCAL_SHA)

    assert decision.state is IdempotencyState.UP_TO_DATE
    assert decision.is_up_to_date is True
    assert decision.local_sha == _LOCAL_SHA
    assert decision.remote_sha == _LOCAL_SHA


def test_changed_when_remote_sha_differs() -> None:
    """A genuine content change: remote SHA exists but doesn't match
    ours. The runner proceeds with upload."""

    transport = FakeTransport()
    transport.seed_repo("alice/x", card_data={CONTENT_SHA_KEY: _OTHER_SHA})

    decision = check_idempotency(transport, "alice/x", _LOCAL_SHA)

    assert decision.state is IdempotencyState.CHANGED
    assert decision.is_up_to_date is False
    assert decision.local_sha == _LOCAL_SHA
    assert decision.remote_sha == _OTHER_SHA


def test_changed_when_remote_card_has_no_content_sha_key() -> None:
    """A card written by some other tool / a manually-curated repo
    won't have our conventional key. Treat as ``changed``: we have
    no evidence the remote matches, so we upload."""

    transport = FakeTransport()
    transport.seed_repo("alice/x", card_data={"license": "cc-by-4.0"})

    decision = check_idempotency(transport, "alice/x", _LOCAL_SHA)

    assert decision.state is IdempotencyState.CHANGED
    assert decision.remote_sha is None


def test_changed_when_remote_card_data_is_empty() -> None:
    """A brand-new repo created by ``create_repo`` but with no README
    yet has empty ``card_data``. That's the "I made the repo, never
    uploaded" interim state — treat as ``changed`` so the runner
    pushes the first README."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)

    decision = check_idempotency(transport, "alice/x", _LOCAL_SHA)

    assert decision.state is IdempotencyState.CHANGED
    assert decision.remote_sha is None


# ---------------------------------------------------------------------------
# Defensive: input validation
# ---------------------------------------------------------------------------


def test_empty_local_sha_raises_value_error() -> None:
    """An empty digest is a programmer-side bug; raising loudly keeps
    a forgotten ``compute_content_sha`` call from masquerading as a
    silent no-op."""

    transport = FakeTransport()
    with pytest.raises(ValueError, match="non-empty local_content_sha"):
        check_idempotency(transport, "alice/x", "")


def test_non_string_remote_sha_treated_as_changed() -> None:
    """If a malformed README parses ``pd-ocr-content-sha`` to a non-
    string (e.g. an unquoted hex got coerced to int), we don't crash
    and we don't pretend it matches — we fall through to ``changed``
    with ``remote_sha=None``."""

    transport = FakeTransport()
    transport.seed_repo("alice/x", card_data={CONTENT_SHA_KEY: 12345})

    decision = check_idempotency(transport, "alice/x", _LOCAL_SHA)

    assert decision.state is IdempotencyState.CHANGED
    assert decision.remote_sha is None


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


def test_transport_error_during_repo_exists_propagates() -> None:
    """Network / auth failure during the existence probe must
    propagate so the runner can map it to exit 7. We do NOT swallow
    it as ``repo_missing`` (that would mask a real auth bug)."""

    transport = FakeTransport(raise_on_repo_exists=True)
    with pytest.raises(TransportError, match="raise_on_repo_exists"):
        check_idempotency(transport, "alice/x", _LOCAL_SHA)


# ---------------------------------------------------------------------------
# Round-trip: real upload semantics
# ---------------------------------------------------------------------------


def test_round_trip_after_upload_yields_up_to_date(tmp_path: Path) -> None:
    """End-to-end: stage a folder with a README carrying ``pd-ocr-
    content-sha: <X>``, upload it, then call ``check_idempotency``
    with the same SHA. The fake's ``upload_folder`` refreshes
    ``card_data`` from the README front matter (mirrors real HF), so
    the post-upload check must report ``up_to_date``."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "0000000.png").write_bytes(b"fake-png")
    (tmp_path / "metadata.jsonl").write_text(
        '{"file_name": "data/0000000.png", "text": "Séadna"}\n',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text(
        "---\n"
        "license: cc-by-4.0\n"
        "pd-ocr-shape: recognition/v1\n"
        f"pd-ocr-content-sha: {_LOCAL_SHA}\n"
        "---\n"
        "# Body\n",
        encoding="utf-8",
    )
    transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="initial")

    decision = check_idempotency(transport, "alice/x", _LOCAL_SHA)
    assert decision.state is IdempotencyState.UP_TO_DATE


def test_round_trip_with_local_change_yields_changed(tmp_path: Path) -> None:
    """A second publish where the local SHA has changed must be
    reported as ``changed`` so the runner re-uploads."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)

    (tmp_path / "README.md").write_text(
        f"---\npd-ocr-content-sha: {_OTHER_SHA}\n---\n",
        encoding="utf-8",
    )
    transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="v1")

    # Local digest has moved on — this is what a real edit-and-rerun
    # looks like.
    decision = check_idempotency(transport, "alice/x", _LOCAL_SHA)

    assert decision.state is IdempotencyState.CHANGED
    assert decision.remote_sha == _OTHER_SHA


# ---------------------------------------------------------------------------
# Decision dataclass contract
# ---------------------------------------------------------------------------


def test_decision_is_frozen_so_runner_logging_cannot_mutate() -> None:
    """The runner is expected to log decisions; a frozen dataclass
    rules out a subtle bug where a log helper mutates a field that
    later gets re-read for control flow.

    ``frozen=True`` raises :class:`dataclasses.FrozenInstanceError`,
    which is itself a subclass of :class:`AttributeError` — we assert
    against the latter so the test is resilient to either being
    surfaced (Python has historically tightened the hierarchy).
    """

    transport = FakeTransport()
    decision = check_idempotency(transport, "alice/x", _LOCAL_SHA)

    with pytest.raises(AttributeError):
        decision.state = IdempotencyState.UP_TO_DATE  # type: ignore[misc]


def test_idempotency_state_values_are_string_compatible() -> None:
    """Subclass-of-str is a deliberate convenience for callers that
    want to compare against the literal string (e.g. in log
    formatting / JSON serialization). Lock that in."""

    assert IdempotencyState.UP_TO_DATE == "up_to_date"
    assert IdempotencyState.CHANGED == "changed"
    assert IdempotencyState.REPO_MISSING == "repo_missing"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_idempotency_names_reexported_from_publish_package() -> None:
    """Down-stream code (the runner) imports from
    ``pd_ocr_synth.publish``; make sure the new names land there."""

    from pd_ocr_synth import publish

    assert publish.check_idempotency is check_idempotency
    assert publish.IdempotencyDecision is IdempotencyDecision
    assert publish.IdempotencyState is IdempotencyState


def test_decision_carries_repo_id_for_logging() -> None:
    """The runner's log line will include the repo id; making it part
    of the decision keeps callers from threading it through
    separately."""

    transport = FakeTransport()
    transport.seed_repo("alice/x", card_data={CONTENT_SHA_KEY: _LOCAL_SHA})

    decision = check_idempotency(transport, "alice/x", _LOCAL_SHA)
    assert decision.repo_id == "alice/x"
