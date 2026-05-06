"""Transport interface for the Hugging Face publish step (M08).

Per ``docs/specs/10-publishing.md`` § Tooling used, the upload path
needs five primitive operations:

1. ``HfApi.create_repo`` — first-time repo creation.
2. ``HfApi.upload_large_folder`` — chunked, resumable folder upload.
3. ``HfApi.list_repo_commits`` — read the latest commit's ``card_data``
   for the idempotency check.
4. ``HfApi.create_tag`` — optional ``--tag`` after upload.
5. ``HfApi.repo_info`` (or equivalent) — does the repo exist at all?

Rather than wire ``huggingface_hub`` directly into the CLI runner, we
hide those calls behind a Protocol. Two motivations:

- **Offline tests.** Every M08 deliverable that follows this seam
  (idempotency check, upload commit-message format, exit-code mapping
  for auth + network failures) needs to be testable without a real HF
  token or network. A fake transport makes all of that hermetic.
- **Single import boundary for the SDK.** Once the concrete
  :class:`HfHubTransport` adapter lands in a later chunk, it will be
  the *only* file that imports ``huggingface_hub``. The CLI runner
  stays SDK-agnostic and can be exercised under fakes in unit tests
  while the adapter is exercised under SDK mocks in its own tests.

This module is **pure typing + a fake double**; no SDK import here.
The concrete adapter lands in a later chunk along with the upload
orchestrator.

## What's *not* on the Protocol

We deliberately do not expose:

- File-level upload (``upload_file``). The upload story is folder-at-
  a-time per the spec; a file-level method would invite ad-hoc usage
  that bypasses the resumable chunking.
- Authentication wiring. The ``token`` is supplied at transport
  construction time, not per-call — that's the contract
  ``huggingface_hub.HfApi(token=...)`` already follows and it keeps
  call-sites short.
- Branch / refs management. The spec is silent on multi-branch
  publishes; we publish to ``main`` and never elsewhere.

Adding any of those is fine when a milestone calls for it; the
Protocol is intentionally narrow so a future addition is a deliberate
extension rather than a slow drift.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class TransportError(Exception):
    """Raised for any transport-level failure.

    Distinct exception type so the CLI runner maps it cleanly to the
    publish auth/network exit code (per
    ``docs/specs/10-publishing.md`` § Errors and recovery: auth
    errors → exit 7).

    Concrete adapters wrap their library-specific exceptions in this
    type so the runner does not need to catch
    ``huggingface_hub.errors.HfHubHTTPError`` directly.
    """


@dataclass(frozen=True)
class CommitInfo:
    """Minimal description of a created HF commit.

    Returned by :meth:`HfTransport.upload_folder`. We carry only the
    fields the CLI runner actually surfaces today (commit SHA + the
    commit URL the user can click). A future deliverable that needs
    the parent SHA / commit message echo can add fields here without
    breaking call-sites.

    Attributes
    ----------
    commit_sha:
        The 40-char hex SHA the upload created. Empty string is
        permitted only for the no-op idempotency case where the
        runner short-circuited before calling :meth:`upload_folder`.
    commit_url:
        Human-readable URL the user can paste into a browser. May be
        empty if the underlying SDK does not expose one (in which case
        the runner falls back to a synthesized ``hf.co/<repo>/commit
        /<sha>`` link).
    """

    commit_sha: str
    commit_url: str = ""


@runtime_checkable
class HfTransport(Protocol):
    """The narrow interface the publish runner depends on.

    Implementations:

    - :class:`FakeTransport` — in-memory; used by tests + dry-run
      smoke tests.
    - ``HfHubTransport`` — concrete, wraps ``huggingface_hub.HfApi``;
      lands in a later chunk.

    Method shapes mirror ``huggingface_hub.HfApi`` keyword conventions
    (``repo_id``, ``private``, etc.) so the eventual adapter is a thin
    forwarder rather than a translator.
    """

    def repo_exists(self, repo_id: str) -> bool:
        """Return ``True`` iff a dataset repo with ``repo_id`` exists.

        ``repo_id`` is the canonical ``OWNER/NAME`` form. Implementations
        must distinguish "exists" from any other failure: a network
        error during the check should raise :class:`TransportError`,
        not silently return ``False``.
        """
        ...

    def create_repo(
        self,
        repo_id: str,
        *,
        private: bool,
        exist_ok: bool = True,
    ) -> None:
        """Create a dataset repo at ``repo_id``.

        ``exist_ok=True`` (the default) makes a second call against an
        existing repo a no-op; this matches HF's own contract and lets
        the runner call ``create_repo`` unconditionally before upload
        without first probing :meth:`repo_exists`.
        """
        ...

    def read_remote_card_data(
        self,
        repo_id: str,
        *,
        revision: str = "main",
    ) -> Mapping[str, Any]:
        """Return the latest commit's README YAML front matter.

        Used for the idempotency check: the runner compares
        ``card_data["pd-ocr-content-sha"]`` against the locally
        computed digest and short-circuits when they match.

        Implementations may legitimately return an empty mapping for a
        brand-new repo with no README. Failure to *reach* the repo
        (auth / network) should raise :class:`TransportError`; a
        successfully-fetched but missing-card case is just an empty
        dict.
        """
        ...

    def upload_folder(
        self,
        repo_id: str,
        *,
        folder_path: Path,
        commit_message: str,
        revision: str = "main",
    ) -> CommitInfo:
        """Upload every file under ``folder_path`` as one commit.

        Mirrors ``HfApi.upload_large_folder``: chunks large folders,
        resumes on interruption. The runner does not pass an explicit
        ``ignore_patterns`` because the staging dir is already
        post-filtered.
        """
        ...

    def create_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        revision: str = "main",
        message: str | None = None,
    ) -> None:
        """Create a git tag at ``revision`` of ``repo_id``.

        Used by ``pd-ocr-synth publish --tag <version>``. Tag conflicts
        (tag already exists with a different SHA) raise
        :class:`TransportError`; idempotent re-tagging at the same SHA
        is allowed by the HF API and treated as success here.
        """
        ...


# ---------------------------------------------------------------------------
# In-memory fake
# ---------------------------------------------------------------------------


@dataclass
class _FakeRepo:
    """One row in :class:`FakeTransport`'s in-memory repo registry.

    Mutable: tests reach in to seed initial state (e.g. pre-existing
    ``card_data`` for idempotency tests) and to assert post-conditions
    (e.g. did the runner upload the expected files?).
    """

    private: bool
    card_data: dict[str, Any] = field(default_factory=dict)
    files: dict[str, bytes] = field(default_factory=dict)
    commits: list[CommitInfo] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)


class FakeTransport:
    """In-memory :class:`HfTransport` for tests + dry-run smoke tests.

    Records every call as an attribute on the instance so tests can
    assert against operation order ("did we ``create_repo`` before
    ``upload_folder``?") without a mocking library.

    Network and auth behavior are simulated via the constructor flags:

    - ``raise_on_repo_exists`` — make :meth:`repo_exists` raise
      :class:`TransportError` (e.g. simulate a network outage during
      the existence probe).
    - ``raise_on_upload`` — make :meth:`upload_folder` raise
      :class:`TransportError` (e.g. simulate a mid-upload network
      failure).

    The fake stores file *contents* (bytes), not just paths, so a
    future test that wants to assert e.g. that the uploaded README has
    a specific ``pd-ocr-content-sha`` line can just decode
    ``transport.repos["alice/x"].files["README.md"]`` and parse it.
    """

    def __init__(
        self,
        *,
        raise_on_repo_exists: bool = False,
        raise_on_upload: bool = False,
    ) -> None:
        self.repos: dict[str, _FakeRepo] = {}
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self._raise_on_repo_exists = raise_on_repo_exists
        self._raise_on_upload = raise_on_upload
        # Monotonic commit counter so SHAs are deterministic across
        # tests but distinct within one run. Simulates HF's hex SHAs
        # without pulling in a real hash — tests should not depend on
        # the digest *value*, only on its presence / equality across
        # calls.
        self._next_commit = 0

    # ------------------------------------------------------------------
    # Test seam helpers (not part of the Protocol).
    # ------------------------------------------------------------------

    def seed_repo(
        self,
        repo_id: str,
        *,
        private: bool = False,
        card_data: Mapping[str, Any] | None = None,
        files: Mapping[str, bytes] | None = None,
    ) -> None:
        """Install a pre-existing repo into the fake.

        Tests use this to model "the repo already has a published
        commit with content-SHA X" so the idempotency check can
        compare against a known value without first running an
        upload.
        """

        self.repos[repo_id] = _FakeRepo(
            private=private,
            card_data=dict(card_data or {}),
            files=dict(files or {}),
        )

    # ------------------------------------------------------------------
    # HfTransport implementation.
    # ------------------------------------------------------------------

    def repo_exists(self, repo_id: str) -> bool:
        self.calls.append(("repo_exists", {"repo_id": repo_id}))
        if self._raise_on_repo_exists:
            raise TransportError(
                f"fake transport refusing repo_exists({repo_id}) (raise_on_repo_exists=True)"
            )
        return repo_id in self.repos

    def create_repo(
        self,
        repo_id: str,
        *,
        private: bool,
        exist_ok: bool = True,
    ) -> None:
        self.calls.append(
            (
                "create_repo",
                {"repo_id": repo_id, "private": private, "exist_ok": exist_ok},
            )
        )
        if repo_id in self.repos:
            if not exist_ok:
                raise TransportError(
                    f"fake transport: repo {repo_id} already exists and exist_ok=False"
                )
            # ``exist_ok`` semantics: keep the existing repo as-is; do
            # not flip its visibility. The real HF API behaves the
            # same way.
            return
        self.repos[repo_id] = _FakeRepo(private=private)

    def read_remote_card_data(
        self,
        repo_id: str,
        *,
        revision: str = "main",
    ) -> Mapping[str, Any]:
        self.calls.append(
            (
                "read_remote_card_data",
                {"repo_id": repo_id, "revision": revision},
            )
        )
        repo = self.repos.get(repo_id)
        if repo is None:
            # Mirror HF: querying card_data on a missing repo is an
            # error, not an empty mapping. Distinguishes "no card yet
            # in an existing repo" (empty dict) from "no repo at all"
            # (TransportError).
            raise TransportError(f"fake transport: repo {repo_id} does not exist")
        return dict(repo.card_data)

    def upload_folder(
        self,
        repo_id: str,
        *,
        folder_path: Path,
        commit_message: str,
        revision: str = "main",
    ) -> CommitInfo:
        self.calls.append(
            (
                "upload_folder",
                {
                    "repo_id": repo_id,
                    "folder_path": str(folder_path),
                    "commit_message": commit_message,
                    "revision": revision,
                },
            )
        )
        if self._raise_on_upload:
            raise TransportError(
                f"fake transport refusing upload_folder({repo_id}) (raise_on_upload=True)"
            )
        repo = self.repos.get(repo_id)
        if repo is None:
            raise TransportError(
                f"fake transport: repo {repo_id} does not exist; call create_repo first"
            )

        folder = Path(folder_path)
        if not folder.is_dir():
            raise TransportError(f"fake transport: folder_path {folder} is not a directory")

        # Snapshot every file under the folder so subsequent reads see
        # exactly what would have been uploaded. We replace, not merge,
        # because HF's ``upload_large_folder`` is "the folder == the
        # repo", not a layered patch.
        new_files: dict[str, bytes] = {}
        for path in sorted(folder.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(folder).as_posix()
            new_files[rel] = path.read_bytes()
        repo.files = new_files

        # If the upload includes a README with a ``pd-ocr-*`` front
        # matter block, refresh the repo's ``card_data`` so the next
        # ``read_remote_card_data`` call sees the just-uploaded values.
        # This mirrors HF's behavior: ``card_data`` on a repo IS the
        # parsed README front matter.
        readme_bytes = new_files.get("README.md")
        if readme_bytes is not None:
            repo.card_data = _parse_readme_front_matter(readme_bytes)

        # Build a unique 40-char hex-shaped SHA without ever
        # truncating the counter: pad the counter into the trailing 30
        # hex digits so two distinct uploads always produce distinct
        # SHAs (the bug we shipped first was a ``[:40]`` truncation
        # that lost the counter when ``_next_commit`` was small).
        commit = CommitInfo(
            commit_sha=f"fakecmt{self._next_commit:033x}",
            commit_url=f"https://huggingface.co/datasets/{repo_id}/commit/fake{self._next_commit}",
        )
        self._next_commit += 1
        repo.commits.append(commit)
        return commit

    def create_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        revision: str = "main",
        message: str | None = None,
    ) -> None:
        self.calls.append(
            (
                "create_tag",
                {
                    "repo_id": repo_id,
                    "tag": tag,
                    "revision": revision,
                    "message": message,
                },
            )
        )
        repo = self.repos.get(repo_id)
        if repo is None:
            raise TransportError(f"fake transport: repo {repo_id} does not exist")
        if not repo.commits:
            raise TransportError(f"fake transport: cannot tag {repo_id} — no commits yet")
        target_sha = repo.commits[-1].commit_sha
        existing = repo.tags.get(tag)
        if existing is not None and existing != target_sha:
            raise TransportError(
                f"fake transport: tag {tag!r} already exists at {existing[:12]} "
                f"(would have moved to {target_sha[:12]})"
            )
        repo.tags[tag] = target_sha


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_readme_front_matter(readme_bytes: bytes) -> dict[str, Any]:
    """Tiny YAML front-matter parser used by :class:`FakeTransport`.

    The fake doesn't need to be bullet-proof — tests control the
    README it sees — but it must agree with the real preflight parser
    on the happy path so a fake-transport-driven test exercises the
    same idempotency contract a real one would. We reuse
    :func:`pd_ocr_synth.publish.preflight._parse_front_matter` via a
    thin wrapper that swallows :class:`PreflightError` (a missing /
    unparseable front matter is just an empty card on the wire).
    """

    # Local import keeps the module's import graph tight: ``transport``
    # is otherwise leaf-level, and only the fake uses preflight.
    from pd_ocr_synth.publish.preflight import PreflightError, _parse_front_matter

    try:
        return _parse_front_matter(
            readme_bytes.decode("utf-8", errors="replace"),
            Path("<fake-transport-readme>"),
        )
    except PreflightError:
        return {}
