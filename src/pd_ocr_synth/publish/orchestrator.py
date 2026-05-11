"""Upload orchestration for ``pd-ocr-synth publish`` (M08).

This module ties the leaf primitives that already shipped — staging
build, content-SHA, preflight, idempotency check, transport — into the
single end-to-end upload call the CLI runner will dispatch to once the
real-upload path replaces the current "use --dry-run" stub.

Per ``docs/specs/10-publishing.md`` the upload sequence is, in order:

1. **Pre-flight** the staging dir (already built upstream by
   :func:`pd_ocr_synth.publish.recognition.build_recognition_staging`):
   the README must carry the documented ``pd-ocr-*`` keys including
   the freshly-embedded ``pd-ocr-content-sha``.
2. **Compute the local content SHA** from the staging dir. For a
   staging dir built by :func:`build_recognition_staging` the digest
   is already pinned in the README, so we recompute over the
   strip-then-rehash version (the ``compute_content_sha`` helper does
   the strip itself) — recomputing rather than parsing keeps this
   path resilient to a hand-edited README and matches the contract
   the idempotency check uses.
3. **Idempotency check** against the transport. If the repo already
   carries our digest in ``card_data.pd-ocr-content-sha`` the publish
   is a no-op and we return early with state ``"no_change"``. Spec 10
   § Idempotency: "If equal → exit 0 with 'no changes' and do not
   commit."
4. **Create the repo** if it doesn't exist yet, unless ``allow_create``
   is False. Spec 10 § Errors and recovery: "Repo doesn't exist →
   Auto-create unless ``--no-create``." When ``allow_create`` is
   False on a missing repo we raise :class:`PublishError` with state
   ``"repo_missing"`` so the CLI runner can map it to the documented
   exit-7 publish failure.
5. **Upload** the staging folder via :meth:`HfTransport.upload_folder`.
   The transport returns a :class:`CommitInfo` we surface verbatim.
6. **Optionally tag** the resulting commit when ``tag`` is supplied.
   Spec 10 § Versioning: "Each publish creates one commit. Tag it
   explicitly when you want a pin." The tag is created *after* a
   successful upload, never before — a tag pointing at a missing /
   half-uploaded commit would be worse than no tag at all.

The orchestrator is pure :class:`HfTransport` orchestration: every
network-shaped operation flows through the Protocol. There is **no
``huggingface_hub`` import here** and tests drive against
:class:`pd_ocr_synth.publish.transport.FakeTransport`. The concrete
SDK-backed transport adapter lands in a later chunk and is the only
file that imports the SDK.

## Why ``PublishResult`` instead of "just the commit SHA"

The CLI runner needs to print a different message for each terminal
state (no-change, created-and-uploaded, plain-upload, tagged) and the
``DATASETS.md``-style operator log wants the digest + URL too. A
single dataclass keeps every call site reading from the same object
rather than passing positional tuples around.

## Why a separate ``allow_create`` flag instead of inferring from CLI

The CLI parses ``--no-create`` into a boolean; threading the flag
through verbatim keeps the orchestrator agnostic of how the user
toggled it. A future programmatic caller (e.g. a Python notebook
publishing without the CLI) gets the same default ``allow_create=True``
that matches the spec's documented "auto-create unless ``--no-create``"
behavior.

## What this module does *not* do

- It does **not** build the staging dir. The caller must have already
  run :func:`build_recognition_staging` (or equivalent) so the
  staging dir on disk is upload-ready.
- It does **not** resolve the HF token. Token resolution lives in
  :mod:`pd_ocr_synth.publish.auth` and is the CLI runner's
  responsibility *before* dispatching to this orchestrator. Once
  resolved, the token is wired into the transport at construction
  time, not threaded through here.
- It does **not** map exit codes. The CLI runner translates
  :class:`PublishError` (and any propagated
  :class:`pd_ocr_synth.publish.transport.TransportError` /
  :class:`pd_ocr_synth.publish.preflight.PreflightError` /
  :class:`pd_ocr_synth.publish.content_sha.ContentShaError`) onto
  spec 01's exit codes. Embedding exit codes here would couple this
  leaf module to the CLI dispatcher.
- It does **not** decide what to upload. The staging dir on disk
  *is* the contract: every regular file under it goes up, the
  preflight has already verified the layout, and the orchestrator
  does not second-guess that decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pd_ocr_synth.publish.content_sha import compute_content_sha
from pd_ocr_synth.publish.idempotency import (
    IdempotencyState,
    check_idempotency,
)
from pd_ocr_synth.publish.preflight import assert_staging_publish_ready
from pd_ocr_synth.publish.transport import (
    CommitInfo,
    HfTransport,
)


class PublishError(Exception):
    """Raised for orchestration-level failures the CLI runner maps to exit 7.

    Distinct from
    :class:`pd_ocr_synth.publish.transport.TransportError` (which is
    network/auth-shaped) and from
    :class:`pd_ocr_synth.publish.preflight.PreflightError` (which is
    staging-shape-shaped). A :class:`PublishError` indicates a
    *policy* violation — e.g. the repo doesn't exist and the caller
    opted out of auto-creation. The CLI maps all three to exit 7
    today; keeping them distinct preserves the option to differentiate
    later (e.g. a non-zero but non-7 code for "repo missing &
    --no-create").
    """


class PublishState(StrEnum):
    """Terminal state of a :func:`publish_recognition` call.

    ``StrEnum`` so callers (the CLI runner, tests) can compare against
    the literal strings without importing the enum, mirroring
    :class:`pd_ocr_synth.publish.idempotency.IdempotencyState`.

    Members
    -------
    NO_CHANGE:
        Idempotency check found a matching ``pd-ocr-content-sha`` on
        the remote. No ``upload_folder`` happened. The runner exits 0
        with the spec's "no changes" message.
    CREATED:
        The repo did not exist; the orchestrator created it *and*
        uploaded. ``commit_sha`` carries the first commit's SHA. The
        runner prints both "created repo" and "uploaded" lines.
    UPLOADED:
        The repo already existed and the local content differed; we
        uploaded and got back a new commit. Most-common branch on a
        re-publish that has actual changes.
    """

    NO_CHANGE = "no_change"
    CREATED = "created"
    UPLOADED = "uploaded"


@dataclass(frozen=True, slots=True)
class PublishResult:
    """What :func:`publish_recognition` did, end to end.

    Frozen so the runner can log without worrying about downstream
    mutation; ``slots=True`` keeps it cheap for the inevitable case
    where a single CLI invocation builds many of these (e.g. a future
    multi-recipe publish loop).

    Attributes
    ----------
    state:
        See :class:`PublishState`. Drives the runner's branch on
        which "would you like fries with that" log line to print.
    repo_id:
        Echoed verbatim. Useful for log aggregation that wants to
        match the result to its originating publish call.
    content_sha:
        The digest the orchestrator computed locally. On the
        ``NO_CHANGE`` branch this equals the remote's
        ``pd-ocr-content-sha`` by construction.
    commit_sha:
        The 40-char SHA of the upload commit, or empty string if no
        upload happened (``NO_CHANGE``). Tests treat empty-string as
        the "no commit" sentinel.
    commit_url:
        Human-clickable URL the transport returned, or empty string.
        The CLI runner falls back to a synthesized
        ``hf.co/datasets/<repo>/commit/<sha>`` link when this is
        empty so users always get *some* link.
    tag:
        The tag created against the upload commit, or ``None`` if no
        ``tag`` was passed. Echoed so the runner can include it in
        the success line without re-reading the call args.
    """

    state: PublishState
    repo_id: str
    content_sha: str
    commit_sha: str
    commit_url: str
    tag: str | None

    @property
    def is_no_change(self) -> bool:
        """True iff the publish was an idempotent no-op.

        Convenience for the runner's short-circuit; equivalent to
        ``result.state is PublishState.NO_CHANGE`` but reads more
        naturally at call sites.
        """

        return self.state is PublishState.NO_CHANGE


def publish_recognition(
    transport: HfTransport,
    repo_id: str,
    staging_dir: Path,
    *,
    commit_message: str,
    private: bool = False,
    allow_create: bool = True,
    tag: str | None = None,
) -> PublishResult:
    """Upload an HF imagefolder staging dir, idempotent against repo state.

    Implements the spec's upload sequence end-to-end:
    pre-flight → content-SHA → idempotency → maybe ``create_repo`` →
    ``upload_folder`` → maybe ``create_tag``.

    The function is **pure transport orchestration** — every
    network-shaped operation flows through the
    :class:`HfTransport` Protocol, and tests drive the whole flow
    against :class:`pd_ocr_synth.publish.transport.FakeTransport`. No
    ``huggingface_hub`` import.

    Parameters
    ----------
    transport:
        Anything satisfying :class:`HfTransport`. In production this
        will be the SDK-backed adapter; in tests it's
        :class:`pd_ocr_synth.publish.transport.FakeTransport`.
    repo_id:
        Canonical ``OWNER/NAME``. Repo-id validation is not the
        orchestrator's job — the CLI runner / a future small chunk
        owns that. Passing a malformed value here will surface the
        SDK's own error via :class:`TransportError`.
    staging_dir:
        Already-built staging directory (per
        :func:`pd_ocr_synth.publish.recognition.build_recognition_staging`).
        Must exist and pass :func:`assert_staging_publish_ready`.
    commit_message:
        The commit message for the upload commit. The caller
        supplies a fully-formatted string — the spec's default
        format (``pd-ocr-synth render @<recipe-sha>``) is built
        elsewhere; this function just forwards it. ``--message
        <override>`` from spec 10 plumbs through unchanged.
    private:
        Visibility for newly-created repos. Ignored when the repo
        already exists (HF's :meth:`create_repo` honors
        ``exist_ok=True`` by leaving the existing visibility alone;
        that's the right default for a publish step that should
        never silently flip a public repo to private).
    allow_create:
        When True (the default, matching spec 10's "auto-create
        unless ``--no-create``"), a missing repo is created before
        upload. When False, a missing repo raises
        :class:`PublishError` so the CLI runner can map it to exit 7.
    tag:
        Optional version tag (per spec 10 § Versioning). Created
        *after* a successful upload; never against a missing or
        half-finished commit. ``None`` skips tag creation.

    Returns
    -------
    PublishResult
        Structured terminal state + artifacts. The runner prints a
        per-state log line and exits 0.

    Raises
    ------
    pd_ocr_synth.publish.preflight.PreflightError
        Staging dir is structurally invalid (missing README, missing
        ``pd-ocr-*`` keys, etc). The runner maps this to exit 6.
    pd_ocr_synth.publish.content_sha.ContentShaError
        Could not hash the staging dir.
    PublishError
        ``allow_create=False`` and the repo doesn't exist.
    pd_ocr_synth.publish.transport.TransportError
        Any transport-level failure (network, auth, conflict). The
        runner maps this to exit 7. We do not catch / repackage
        TransportError here — keeping it in flight preserves the
        original error message for the user.
    """

    # Step 1: pre-flight. The staging dir might have been built by an
    # earlier render pass and then locally edited; we re-run the
    # check rather than trusting a stale invariant. Cheap (single
    # README parse).
    assert_staging_publish_ready(staging_dir)

    # Step 2: content-SHA. We recompute over the staging dir rather
    # than parsing the README's pinned digest so a hand-edited README
    # cannot lie to the idempotency check.
    local_content_sha = compute_content_sha(staging_dir)

    # Step 3: idempotency. Three branches: ``UP_TO_DATE`` short-
    # circuits to ``NO_CHANGE``; ``REPO_MISSING`` falls through to
    # the ``allow_create`` branch; ``CHANGED`` falls through to a
    # plain upload.
    decision = check_idempotency(transport, repo_id, local_content_sha)

    if decision.state is IdempotencyState.UP_TO_DATE:
        # Spec 10: "exit 0 with 'no changes' and do not commit." We
        # don't synthesize a commit_sha here — the caller can use
        # ``decision.remote_sha`` if it wants to identify the unchanged
        # commit, but the orchestrator's job is to report "no upload
        # happened" as cleanly as possible.
        return PublishResult(
            state=PublishState.NO_CHANGE,
            repo_id=repo_id,
            content_sha=local_content_sha,
            commit_sha="",
            commit_url="",
            tag=None,
        )

    state: PublishState
    if decision.state is IdempotencyState.REPO_MISSING:
        if not allow_create:
            raise PublishError(
                f"repo {repo_id} does not exist and allow_create=False; "
                "drop --no-create or pre-create the repo manually"
            )
        # Step 4: create_repo. ``exist_ok=True`` is belt-and-braces:
        # the idempotency check just told us the repo doesn't exist,
        # but a parallel publisher could have raced us between then
        # and now. The HF API is itself idempotent on
        # ``exist_ok=True`` so this is the safe default.
        transport.create_repo(repo_id, private=private, exist_ok=True)
        state = PublishState.CREATED
    else:
        # ``IdempotencyState.CHANGED`` is the third and final state;
        # asserting it explicitly catches any future enum addition
        # that forgets to update this branch. We use a plain
        # comparison (not ``assert``) so ``-O`` doesn't strip the
        # check.
        if decision.state is not IdempotencyState.CHANGED:
            raise PublishError(f"unexpected idempotency state {decision.state!r} for {repo_id}")
        state = PublishState.UPLOADED

    # Step 5: upload. The transport's ``upload_folder`` mirrors HF's
    # ``upload_large_folder`` — it chunks and resumes. We surface the
    # CommitInfo verbatim.
    commit: CommitInfo = transport.upload_folder(
        repo_id,
        folder_path=staging_dir,
        commit_message=commit_message,
    )

    # Step 6: optional tag. Only after a successful upload — a tag
    # against a missing / half-uploaded commit is worse than no tag.
    applied_tag: str | None = None
    if tag is not None:
        transport.create_tag(repo_id, tag=tag)
        applied_tag = tag

    return PublishResult(
        state=state,
        repo_id=repo_id,
        content_sha=local_content_sha,
        commit_sha=commit.commit_sha,
        commit_url=commit.commit_url,
        tag=applied_tag,
    )
