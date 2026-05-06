"""Unit tests for ``pd_ocr_synth.publish.transport`` (M08).

The transport module defines the seam between the publish CLI runner
and the Hugging Face SDK; the runner depends on the ``HfTransport``
Protocol and the test suite drives a ``FakeTransport`` so every M08
deliverable past this point can be exercised without a real network /
real ``huggingface_hub`` import.

These tests cover the fake's contract directly: that it implements the
Protocol, that it records calls, that error injection works, that
``upload_folder`` snapshots files and refreshes ``card_data``, and
that tag conflicts surface as :class:`TransportError`. The runner-side
tests live alongside the runner once the upload chunk lands.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.publish.transport import (
    CommitInfo,
    FakeTransport,
    HfTransport,
    TransportError,
)

# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_fake_transport_satisfies_hf_transport_protocol() -> None:
    """The runtime ``isinstance`` check is the cheap proof that the
    fake's method shapes match the Protocol. If a method gets renamed
    on either side this fires immediately."""

    transport = FakeTransport()
    assert isinstance(transport, HfTransport)


# ---------------------------------------------------------------------------
# repo_exists / create_repo
# ---------------------------------------------------------------------------


def test_repo_exists_returns_false_for_unseeded_repo() -> None:
    transport = FakeTransport()
    assert transport.repo_exists("alice/x") is False
    # Probe was recorded.
    assert transport.calls == [("repo_exists", {"repo_id": "alice/x"})]


def test_repo_exists_returns_true_after_create_repo() -> None:
    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    assert transport.repo_exists("alice/x") is True


def test_create_repo_records_visibility() -> None:
    transport = FakeTransport()
    transport.create_repo("alice/secret", private=True)
    repo = transport.repos["alice/secret"]
    assert repo.private is True


def test_create_repo_is_idempotent_with_exist_ok_default() -> None:
    """Default ``exist_ok=True`` matches HF's contract — a second call
    against the existing repo is a no-op, not a failure."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    # No exception.
    transport.create_repo("alice/x", private=False)


def test_create_repo_raises_when_exist_ok_false() -> None:
    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    with pytest.raises(TransportError, match="already exists"):
        transport.create_repo("alice/x", private=False, exist_ok=False)


def test_create_repo_does_not_change_visibility_on_re_create() -> None:
    """Important: a re-``create_repo`` with a different ``private``
    flag does NOT silently flip the existing repo's visibility. The
    real HF API behaves the same way; we mirror it so a runner bug
    doesn't accidentally rely on the wrong semantics."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    transport.create_repo("alice/x", private=True)  # would-be flip
    assert transport.repos["alice/x"].private is False


def test_repo_exists_can_simulate_network_failure() -> None:
    transport = FakeTransport(raise_on_repo_exists=True)
    with pytest.raises(TransportError, match="raise_on_repo_exists"):
        transport.repo_exists("alice/x")


# ---------------------------------------------------------------------------
# read_remote_card_data
# ---------------------------------------------------------------------------


def test_read_remote_card_data_returns_seeded_card() -> None:
    """The idempotency check reads the latest commit's card data; the
    fake returns whatever was seeded so a test can compare against a
    known content-SHA."""

    transport = FakeTransport()
    transport.seed_repo(
        "alice/x",
        card_data={"pd-ocr-content-sha": "abc123", "license": "cc-by-4.0"},
    )
    card = transport.read_remote_card_data("alice/x")
    assert card["pd-ocr-content-sha"] == "abc123"
    assert card["license"] == "cc-by-4.0"


def test_read_remote_card_data_returns_empty_dict_when_card_unset() -> None:
    """A brand-new repo with no published README has no card data;
    the runner needs to distinguish that from a repo it can't reach."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    assert transport.read_remote_card_data("alice/x") == {}


def test_read_remote_card_data_raises_for_missing_repo() -> None:
    """Querying card data on a non-existent repo is a :class:`TransportError`,
    not an empty mapping. The runner needs the distinction."""

    transport = FakeTransport()
    with pytest.raises(TransportError, match="does not exist"):
        transport.read_remote_card_data("alice/missing")


def test_read_remote_card_data_returns_a_copy_not_a_view() -> None:
    """Defensive: mutating the returned dict must not corrupt the
    fake's internal state. Callers (e.g. the runner) may reasonably
    pop / set keys for normalization."""

    transport = FakeTransport()
    transport.seed_repo("alice/x", card_data={"pd-ocr-content-sha": "abc"})
    card = transport.read_remote_card_data("alice/x")
    card["pd-ocr-content-sha"] = "tampered"  # type: ignore[index]
    fresh = transport.read_remote_card_data("alice/x")
    assert fresh["pd-ocr-content-sha"] == "abc"


# ---------------------------------------------------------------------------
# upload_folder
# ---------------------------------------------------------------------------


def _write_staging(root: Path) -> None:
    """Build a minimal staging-shaped folder for upload tests."""

    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data" / "0000000.png").write_bytes(b"fake-png-bytes")
    (root / "metadata.jsonl").write_text(
        '{"file_name": "data/0000000.png", "text": "Séadna"}\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "---\n"
        "license: cc-by-4.0\n"
        "pd-ocr-shape: recognition/v1\n"
        "pd-ocr-content-sha: deadbeefcafebabe0123456789abcdef0123456789abcdef0123456789abcdef\n"
        "---\n"
        "# Body\n",
        encoding="utf-8",
    )


def test_upload_folder_snapshots_files_into_repo(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    _write_staging(tmp_path)

    info = transport.upload_folder(
        "alice/x",
        folder_path=tmp_path,
        commit_message="initial publish",
    )

    assert isinstance(info, CommitInfo)
    assert info.commit_sha  # non-empty
    assert info.commit_url.startswith("https://huggingface.co/datasets/alice/x/commit/")

    repo = transport.repos["alice/x"]
    # File contents preserved verbatim.
    assert repo.files["data/0000000.png"] == b"fake-png-bytes"
    assert "metadata.jsonl" in repo.files
    assert "README.md" in repo.files


def test_upload_folder_refreshes_card_data_from_uploaded_readme(tmp_path: Path) -> None:
    """``card_data`` IS the README front matter, just like real HF.
    After upload, the next ``read_remote_card_data`` must reflect what
    was just uploaded — that's the round-trip the idempotency check
    depends on."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    _write_staging(tmp_path)

    transport.upload_folder(
        "alice/x",
        folder_path=tmp_path,
        commit_message="initial publish",
    )

    card = transport.read_remote_card_data("alice/x")
    assert (
        card["pd-ocr-content-sha"]
        == "deadbeefcafebabe0123456789abcdef0123456789abcdef0123456789abcdef"
    )
    assert card["license"] == "cc-by-4.0"


def test_upload_folder_replaces_files_does_not_merge(tmp_path: Path) -> None:
    """``upload_large_folder`` is a "the folder == the repo" semantic,
    not a layered patch. A second upload that omits a previously-
    present file must remove it."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)

    first = tmp_path / "first"
    first.mkdir()
    (first / "old.txt").write_bytes(b"old")
    (first / "shared.txt").write_bytes(b"v1")
    transport.upload_folder("alice/x", folder_path=first, commit_message="v1")

    second = tmp_path / "second"
    second.mkdir()
    (second / "shared.txt").write_bytes(b"v2")
    (second / "new.txt").write_bytes(b"new")
    transport.upload_folder("alice/x", folder_path=second, commit_message="v2")

    repo = transport.repos["alice/x"]
    assert "old.txt" not in repo.files
    assert repo.files["shared.txt"] == b"v2"
    assert repo.files["new.txt"] == b"new"


def test_upload_folder_records_each_commit(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    _write_staging(tmp_path)

    info1 = transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="c1")
    info2 = transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="c2")

    repo = transport.repos["alice/x"]
    assert [c.commit_sha for c in repo.commits] == [info1.commit_sha, info2.commit_sha]
    # SHAs are distinct.
    assert info1.commit_sha != info2.commit_sha


def test_upload_folder_raises_for_missing_repo(tmp_path: Path) -> None:
    transport = FakeTransport()
    _write_staging(tmp_path)
    with pytest.raises(TransportError, match="does not exist"):
        transport.upload_folder(
            "alice/missing",
            folder_path=tmp_path,
            commit_message="x",
        )


def test_upload_folder_raises_for_missing_folder(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    with pytest.raises(TransportError, match="not a directory"):
        transport.upload_folder(
            "alice/x",
            folder_path=tmp_path / "no-such-dir",
            commit_message="x",
        )


def test_upload_folder_can_simulate_network_failure(tmp_path: Path) -> None:
    transport = FakeTransport(raise_on_upload=True)
    transport.create_repo("alice/x", private=False)
    _write_staging(tmp_path)
    with pytest.raises(TransportError, match="raise_on_upload"):
        transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="x")
    # Crucially: a failed upload must not corrupt the repo's files
    # or commit history.
    repo = transport.repos["alice/x"]
    assert repo.files == {}
    assert repo.commits == []


def test_upload_folder_records_all_files_recursively(tmp_path: Path) -> None:
    """Nested directories are walked and stored with POSIX-style
    relative paths so the keys match what the real upload sees."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    nested = tmp_path / "data" / "subdir"
    nested.mkdir(parents=True)
    (nested / "a.png").write_bytes(b"a")
    (tmp_path / "README.md").write_text("---\n---\n", encoding="utf-8")

    transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="x")
    repo = transport.repos["alice/x"]
    assert "data/subdir/a.png" in repo.files
    assert "README.md" in repo.files


# ---------------------------------------------------------------------------
# create_tag
# ---------------------------------------------------------------------------


def test_create_tag_pins_the_latest_commit(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    _write_staging(tmp_path)
    info = transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="x")

    transport.create_tag("alice/x", tag="v1.0")
    repo = transport.repos["alice/x"]
    assert repo.tags["v1.0"] == info.commit_sha


def test_create_tag_idempotent_at_same_sha(tmp_path: Path) -> None:
    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    _write_staging(tmp_path)
    transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="x")

    transport.create_tag("alice/x", tag="v1.0")
    # Same SHA → no error.
    transport.create_tag("alice/x", tag="v1.0")


def test_create_tag_conflict_raises(tmp_path: Path) -> None:
    """Tagging an existing tag at a *different* SHA is an error."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    _write_staging(tmp_path)
    transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="c1")
    transport.create_tag("alice/x", tag="v1.0")

    # New commit moves HEAD; tagging v1.0 again would now point
    # somewhere else.
    transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="c2")
    with pytest.raises(TransportError, match="already exists"):
        transport.create_tag("alice/x", tag="v1.0")


def test_create_tag_raises_for_missing_repo() -> None:
    transport = FakeTransport()
    with pytest.raises(TransportError, match="does not exist"):
        transport.create_tag("alice/missing", tag="v1.0")


def test_create_tag_raises_when_no_commits_yet() -> None:
    """Spec: a tag pins a commit; no commits → no tag."""

    transport = FakeTransport()
    transport.create_repo("alice/x", private=False)
    with pytest.raises(TransportError, match="no commits"):
        transport.create_tag("alice/x", tag="v1.0")


# ---------------------------------------------------------------------------
# Call recording
# ---------------------------------------------------------------------------


def test_calls_records_in_order_for_runner_assertions(tmp_path: Path) -> None:
    """Tests for the upload runner (next chunk) will assert that we
    create_repo BEFORE upload_folder, so the call recorder needs to
    preserve order."""

    transport = FakeTransport()
    _write_staging(tmp_path)

    transport.repo_exists("alice/x")
    transport.create_repo("alice/x", private=False)
    transport.upload_folder("alice/x", folder_path=tmp_path, commit_message="x")
    transport.create_tag("alice/x", tag="v1.0")

    op_names = [name for name, _ in transport.calls]
    assert op_names == ["repo_exists", "create_repo", "upload_folder", "create_tag"]


def test_seed_repo_does_not_show_up_in_calls() -> None:
    """``seed_repo`` is a test helper, not a Protocol method. It
    must not pollute ``transport.calls`` (which tests use to assert
    on the runner's behavior)."""

    transport = FakeTransport()
    transport.seed_repo("alice/x", card_data={"pd-ocr-content-sha": "abc"})
    assert transport.calls == []


# ---------------------------------------------------------------------------
# Transport-level exports surface from the package
# ---------------------------------------------------------------------------


def test_transport_types_reexported_from_publish_package() -> None:
    """Down-stream code imports from ``pd_ocr_synth.publish``; make
    sure the new names land there too."""

    from pd_ocr_synth import publish

    assert publish.HfTransport is HfTransport
    assert publish.FakeTransport is FakeTransport
    assert publish.TransportError is TransportError
    assert publish.CommitInfo is CommitInfo
