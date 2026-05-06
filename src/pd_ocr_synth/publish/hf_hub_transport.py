"""Concrete :class:`HfTransport` adapter wrapping ``huggingface_hub`` (M08).

This module is the **only** file in ``pd_ocr_synth`` that imports
``huggingface_hub``. Every other publish module — orchestrator, CLI
runner, idempotency check, dataset card, content SHA — drives against
the :class:`pd_ocr_synth.publish.transport.HfTransport` Protocol so
they remain unit-testable without the SDK and without the network.

The adapter is a thin translator: each Protocol method forwards to the
corresponding ``HfApi`` call documented in ``docs/specs/10-publishing.md``
§ Tooling used. The mapping is:

============================  ==============================================
Protocol method               ``HfApi`` call
============================  ==============================================
``repo_exists``               ``repo_exists(repo_id, repo_type='dataset')``
``create_repo``               ``create_repo(repo_id, repo_type='dataset',
                              private=..., exist_ok=...)``
``read_remote_card_data``     ``dataset_info(repo_id, revision=...)``
                              then ``.card_data`` → ``dict``
``upload_folder``             ``upload_large_folder(repo_id,
                              folder_path=str(p), repo_type='dataset',
                              revision=...)`` then
                              ``list_repo_commits(...)`` for the SHA
``create_tag``                ``create_tag(repo_id, repo_type='dataset',
                              tag=..., revision=...,
                              tag_message=message)``
============================  ==============================================

Why ``upload_large_folder`` rather than ``upload_folder``: spec 10
mandates the resumable variant. The trade-off is that
``upload_large_folder`` does not accept a ``commit_message``; the
chunked uploads use HF's auto-generated messages. We honor the
caller-supplied ``commit_message`` by including it in our own log
output but cannot stamp it on the on-HF commit. Recovering the latest
commit SHA after upload is done via
``HfApi.list_repo_commits(...)[0]``; if that probe fails we fall back
to an empty SHA rather than failing the whole publish — the upload
itself succeeded by then and surfacing a misleading TransportError
would mask a genuine success.

Error wrapping: every ``huggingface_hub`` exception is repackaged as
:class:`pd_ocr_synth.publish.transport.TransportError`. Tests assert
on the wrapping rather than on the SDK-specific class so the
contract is stable across SDK versions.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from pd_ocr_synth.publish.transport import CommitInfo, TransportError

# Repo type stays "dataset" for every call we make. The synth tool only
# publishes datasets; pinning the value here means individual call-sites
# do not have to repeat it.
_REPO_TYPE = "dataset"


class HfHubTransport:
    """Production transport satisfying :class:`HfTransport`.

    Constructed once per CLI invocation by
    :func:`pd_ocr_synth.publish.sdk_transport.make_default_transport`.
    The constructor builds an internal ``HfApi`` and threads the token
    through so per-call ``token=`` plumbing isn't needed.

    Parameters
    ----------
    token:
        The resolved HF token (per spec 10 § Authentication). Used for
        every authenticated call. ``HfApi`` itself accepts ``None`` and
        falls back to ``HfFolder``-cached creds, but we never want that
        ambiguity here — auth resolution is the CLI runner's job and
        the resolved token is what the adapter must use.
    api:
        Test seam. When ``None`` (the production default) the
        constructor builds ``HfApi(token=token, library_name=...)``.
        Tests inject a ``unittest.mock.Mock(spec=HfApi)`` so they can
        assert on call arguments and program return values without a
        real network. Library name is set so HF's own logging /
        analytics distinguishes our calls — purely cosmetic but cheap.
    """

    # Library name passed to HfApi so HF-side request analytics can
    # distinguish synth uploads from generic SDK use. Mirrors the
    # convention used elsewhere in the HF ecosystem.
    _LIBRARY_NAME = "pd-ocr-synth"

    def __init__(self, token: str, *, api: HfApi | None = None) -> None:
        # We accept an optional pre-built ``HfApi`` for tests; production
        # always lets the constructor build one. The eager construction
        # means a bad token surfaces at adapter-creation time (cheap)
        # rather than mid-upload (expensive — partial work to clean up).
        self._token = token
        self._api: HfApi = (
            api
            if api is not None
            else HfApi(
                token=token,
                library_name=self._LIBRARY_NAME,
            )
        )

    # ------------------------------------------------------------------
    # HfTransport implementation
    # ------------------------------------------------------------------

    def repo_exists(self, repo_id: str) -> bool:
        """Probe whether ``repo_id`` exists as a dataset.

        ``HfApi.repo_exists`` returns ``True``/``False`` without raising
        on a missing repo (it converts the 404 internally). Any *other*
        failure (auth error, network outage, 5xx) re-raises as
        :class:`HfHubHTTPError` and we wrap it in
        :class:`TransportError` so callers do not need to catch the
        SDK's own exception hierarchy.
        """

        try:
            return bool(self._api.repo_exists(repo_id, repo_type=_REPO_TYPE))
        except HfHubHTTPError as exc:  # pragma: no cover — network-shaped
            raise TransportError(f"failed to probe repo {repo_id}: {exc}") from exc

    def create_repo(
        self,
        repo_id: str,
        *,
        private: bool,
        exist_ok: bool = True,
    ) -> None:
        """Create the dataset repo, or no-op when it already exists.

        ``HfApi.create_repo`` itself accepts ``exist_ok`` so this maps
        cleanly to the Protocol contract. A non-404 HTTP error wraps as
        :class:`TransportError`; an existing-repo + ``exist_ok=False``
        case re-raises HF's own ``HfHubHTTPError`` text wrapped the
        same way.
        """

        try:
            self._api.create_repo(
                repo_id,
                repo_type=_REPO_TYPE,
                private=private,
                exist_ok=exist_ok,
            )
        except HfHubHTTPError as exc:
            raise TransportError(f"failed to create repo {repo_id}: {exc}") from exc

    def read_remote_card_data(
        self,
        repo_id: str,
        *,
        revision: str = "main",
    ) -> Mapping[str, Any]:
        """Fetch the dataset card front matter for the idempotency check.

        ``HfApi.dataset_info`` returns a ``DatasetInfo`` whose
        ``card_data`` is a ``DatasetCardData`` (dict-like) or ``None``
        for a brand-new repo with no README. We coerce both to a plain
        ``dict`` so the contract returned to callers is just
        ``Mapping[str, Any]``.

        :class:`RepositoryNotFoundError` is re-raised as
        :class:`TransportError`; the spec wants the idempotency check
        to *probe* with :meth:`repo_exists` and only call this method
        when the repo is known to exist, so a 404 here is a programmer
        error worth surfacing rather than swallowing as an empty dict.
        """

        try:
            info = self._api.dataset_info(repo_id, revision=revision)
        except RepositoryNotFoundError as exc:
            raise TransportError(f"repo {repo_id} not found while reading card data") from exc
        except HfHubHTTPError as exc:  # pragma: no cover — network-shaped
            raise TransportError(f"failed to read card data for {repo_id}: {exc}") from exc

        card = getattr(info, "card_data", None)
        if card is None:
            return {}
        # ``DatasetCardData`` exposes a ``to_dict`` method; fall back to
        # ``dict(card)`` for forward-compat in case a future SDK version
        # changes the shape.
        to_dict = getattr(card, "to_dict", None)
        if callable(to_dict):
            return dict(to_dict())
        return dict(card)

    def upload_folder(
        self,
        repo_id: str,
        *,
        folder_path: Path,
        commit_message: str,
        revision: str = "main",
    ) -> CommitInfo:
        """Upload every file under ``folder_path`` via the chunked API.

        ``HfApi.upload_large_folder`` returns ``None`` because it
        creates multiple commits internally. To populate the
        :class:`CommitInfo` our Protocol promises, we fetch the latest
        commit on ``revision`` via ``list_repo_commits`` immediately
        after the upload returns. The probe runs in a separate
        ``try`` so a transient failure on the post-upload read doesn't
        invalidate a successful upload — we just return an empty SHA
        and let the runner print a "see HF UI for commit" line.

        The ``commit_message`` parameter cannot be honored by
        ``upload_large_folder`` (the chunked API auto-generates
        per-chunk messages). It is accepted to satisfy the Protocol
        and so the orchestrator's commit-message logic still lives at
        one site; the adapter docstring documents the limitation.
        """

        try:
            self._api.upload_large_folder(
                repo_id,
                folder_path=str(folder_path),
                repo_type=_REPO_TYPE,
                revision=revision,
            )
        except HfHubHTTPError as exc:
            raise TransportError(
                f"upload_large_folder failed for {repo_id} ({folder_path}): {exc}"
            ) from exc

        # Best-effort: fetch the latest commit SHA so callers get the
        # link to the uploaded commit. A failure here is non-fatal —
        # the upload itself succeeded.
        commit_sha = ""
        commit_url = ""
        try:
            commits = self._api.list_repo_commits(
                repo_id,
                repo_type=_REPO_TYPE,
                revision=revision,
            )
            if commits:
                commit_sha = commits[0].commit_id
                commit_url = f"https://huggingface.co/datasets/{repo_id}/commit/{commit_sha}"
        except HfHubHTTPError:  # pragma: no cover — best-effort probe
            # Swallow: we already uploaded successfully. The runner
            # falls back to a synthesized URL when we return empty.
            pass

        # ``commit_message`` is intentionally unused for the SDK call
        # (see method docstring). Acknowledge the parameter so static
        # analysis doesn't flag it.
        _ = commit_message
        return CommitInfo(commit_sha=commit_sha, commit_url=commit_url)

    def create_tag(
        self,
        repo_id: str,
        *,
        tag: str,
        revision: str = "main",
        message: str | None = None,
    ) -> None:
        """Tag ``revision`` with ``tag`` (annotated when ``message`` is set).

        ``HfApi.create_tag`` accepts ``tag_message`` as its annotated-tag
        text; we forward our ``message`` parameter to it verbatim. A
        tag-already-exists conflict (with a different SHA) surfaces as
        :class:`HfHubHTTPError` from HF; we wrap it.
        ``exist_ok=True`` makes idempotent re-tagging at the same SHA a
        no-op, matching :class:`HfTransport`'s contract.
        """

        try:
            self._api.create_tag(
                repo_id,
                repo_type=_REPO_TYPE,
                tag=tag,
                revision=revision,
                tag_message=message,
                exist_ok=True,
            )
        except HfHubHTTPError as exc:
            raise TransportError(f"failed to tag {repo_id} as {tag!r}: {exc}") from exc


__all__ = ["HfHubTransport"]
