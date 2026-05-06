"""Unit tests for the ``HfHubTransport`` SDK adapter (M08).

The adapter is the *only* file in ``pd_ocr_synth`` that imports
``huggingface_hub``. Every test in this file mocks ``HfApi`` rather
than calling the real network — the contract under test is that each
:class:`pd_ocr_synth.publish.transport.HfTransport` Protocol method
forwards to the documented ``HfApi`` call with the right keyword
arguments and that SDK-specific exceptions are repackaged as
:class:`pd_ocr_synth.publish.transport.TransportError`.

The mapping under test mirrors ``docs/specs/10-publishing.md`` § Tooling
used:

- ``repo_exists`` → ``HfApi.repo_exists(repo_id, repo_type='dataset')``
- ``create_repo`` → ``HfApi.create_repo(..., exist_ok=...)``
- ``read_remote_card_data`` → ``HfApi.dataset_info(...).card_data``
- ``upload_folder`` → ``HfApi.upload_large_folder(...)`` then
  ``HfApi.list_repo_commits(...)`` for the commit SHA
- ``create_tag`` → ``HfApi.create_tag(..., tag_message=message)``
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import httpx
import pytest
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from pd_ocr_synth.publish import HfTransport
from pd_ocr_synth.publish.hf_hub_transport import HfHubTransport
from pd_ocr_synth.publish.transport import TransportError


def _http_error(message: str, *, status: int = 500) -> HfHubHTTPError:
    """Build an ``HfHubHTTPError`` with a synthetic ``httpx.Response``.

    The SDK requires a real response object on the keyword. Tests don't
    care about the body, only that the exception type round-trips into
    the adapter's wrapping. We build a minimal response and feed it in
    so each error-branch test reads cleanly.
    """

    response = httpx.Response(
        status_code=status,
        request=httpx.Request("GET", "https://huggingface.co/api/test"),
    )
    return HfHubHTTPError(message, response=response)


def _repo_not_found(message: str = "404 not found") -> RepositoryNotFoundError:
    """Same idea as :func:`_http_error` for the 404-shaped subclass."""

    response = httpx.Response(
        status_code=404,
        request=httpx.Request("GET", "https://huggingface.co/api/test"),
    )
    return RepositoryNotFoundError(message, response=response)


def _make_api_mock() -> MagicMock:
    """Build a ``MagicMock(spec=HfApi)`` with the methods we exercise.

    Using ``spec=`` so calls to attributes the adapter does not use
    fail at test time rather than silently passing — locks the
    "tight surface" contract documented in the adapter module.
    """

    return MagicMock(spec=HfApi)


def _make_transport(api: MagicMock | None = None) -> HfHubTransport:
    """Build an adapter wired to ``api`` (default: a fresh mock)."""

    return HfHubTransport(token="hf_fake_test_token", api=api or _make_api_mock())


# ---------------------------------------------------------------------------
# Type contract
# ---------------------------------------------------------------------------


def test_adapter_satisfies_hf_transport_protocol() -> None:
    """The adapter MUST be substitutable for the Protocol so the CLI
    runner / orchestrator can drive it without isinstance checks."""

    transport = _make_transport()
    assert isinstance(transport, HfTransport)


def test_adapter_constructed_lazily_via_make_default_transport() -> None:
    """``make_default_transport`` is the production seam; it returns
    an :class:`HfHubTransport` when the SDK is importable."""

    from pd_ocr_synth.publish.sdk_transport import make_default_transport

    t = make_default_transport("hf_fake_test_token")
    assert isinstance(t, HfHubTransport)


def test_adapter_exported_from_publish_package() -> None:
    """``pd_ocr_synth.publish.HfHubTransport`` is the public access
    point. The package surface must expose it lazily — the SDK is an
    optional extra."""

    from pd_ocr_synth import publish

    cls = publish.HfHubTransport  # triggers PEP 562 __getattr__
    assert cls is HfHubTransport


# ---------------------------------------------------------------------------
# repo_exists
# ---------------------------------------------------------------------------


def test_repo_exists_forwards_to_hf_api_with_dataset_type() -> None:
    """``repo_exists`` always passes ``repo_type='dataset'`` — synth
    only publishes datasets."""

    api = _make_api_mock()
    api.repo_exists.return_value = True
    transport = _make_transport(api)

    assert transport.repo_exists("alice/x") is True
    api.repo_exists.assert_called_once_with("alice/x", repo_type="dataset")


def test_repo_exists_returns_false_for_missing_repo() -> None:
    """HF returns ``False`` for a missing repo (it converts the 404
    internally); the adapter must not invent a True / raise here."""

    api = _make_api_mock()
    api.repo_exists.return_value = False
    transport = _make_transport(api)

    assert transport.repo_exists("alice/missing") is False


def test_repo_exists_wraps_http_errors_as_transport_error() -> None:
    """Any non-404 HTTP error wraps as :class:`TransportError` so the
    CLI runner does not need to catch SDK-specific exceptions."""

    api = _make_api_mock()
    api.repo_exists.side_effect = _http_error("server gave a 500", status=500)
    transport = _make_transport(api)

    with pytest.raises(TransportError) as exc_info:
        transport.repo_exists("alice/x")
    assert "alice/x" in str(exc_info.value)


# ---------------------------------------------------------------------------
# create_repo
# ---------------------------------------------------------------------------


def test_create_repo_forwards_private_and_exist_ok_flags() -> None:
    """The Protocol contract is to forward ``private`` and ``exist_ok``
    verbatim. Test both branches of ``private`` so a future regression
    that hard-codes the value is caught."""

    api = _make_api_mock()
    transport = _make_transport(api)

    transport.create_repo("alice/x", private=True, exist_ok=False)
    api.create_repo.assert_called_once_with(
        "alice/x", repo_type="dataset", private=True, exist_ok=False
    )


def test_create_repo_defaults_exist_ok_true() -> None:
    """The Protocol's default ``exist_ok=True`` matches HF's own
    contract; the adapter must preserve it without forcing callers to
    re-state."""

    api = _make_api_mock()
    transport = _make_transport(api)

    transport.create_repo("alice/x", private=False)
    api.create_repo.assert_called_once_with(
        "alice/x", repo_type="dataset", private=False, exist_ok=True
    )


def test_create_repo_wraps_http_errors() -> None:
    api = _make_api_mock()
    api.create_repo.side_effect = _http_error("403 forbidden", status=403)
    transport = _make_transport(api)

    with pytest.raises(TransportError) as exc_info:
        transport.create_repo("alice/x", private=False)
    assert "alice/x" in str(exc_info.value)


# ---------------------------------------------------------------------------
# read_remote_card_data
# ---------------------------------------------------------------------------


def _fake_dataset_info(card_data_obj: object | None) -> SimpleNamespace:
    """Build a stand-in ``DatasetInfo`` with just the ``card_data`` slot."""

    return SimpleNamespace(card_data=card_data_obj)


def test_read_remote_card_data_returns_dict_for_existing_card() -> None:
    """The adapter must coerce the SDK's ``DatasetCardData`` into a
    plain dict so callers depend only on the Protocol's
    ``Mapping[str, Any]`` contract."""

    api = _make_api_mock()
    fake_card = SimpleNamespace(
        to_dict=lambda: {"pd-ocr-content-sha": "abc123", "license": "cc-by-4.0"}
    )
    api.dataset_info.return_value = _fake_dataset_info(fake_card)
    transport = _make_transport(api)

    result = transport.read_remote_card_data("alice/x")
    assert result == {"pd-ocr-content-sha": "abc123", "license": "cc-by-4.0"}
    api.dataset_info.assert_called_once_with("alice/x", revision="main")


def test_read_remote_card_data_falls_back_to_dict_when_no_to_dict() -> None:
    """A future SDK version that drops ``to_dict`` should still work as
    long as ``DatasetCardData`` remains dict-like."""

    api = _make_api_mock()
    api.dataset_info.return_value = _fake_dataset_info({"pd-ocr-content-sha": "deadbeef"})
    transport = _make_transport(api)

    result = transport.read_remote_card_data("alice/x")
    assert result == {"pd-ocr-content-sha": "deadbeef"}


def test_read_remote_card_data_returns_empty_for_missing_card() -> None:
    """A brand-new repo with no README has ``card_data == None``; the
    adapter must surface that as an empty dict (the idempotency check
    treats it as 'remote has no digest yet → upload')."""

    api = _make_api_mock()
    api.dataset_info.return_value = _fake_dataset_info(None)
    transport = _make_transport(api)

    assert transport.read_remote_card_data("alice/empty") == {}


def test_read_remote_card_data_wraps_repository_not_found() -> None:
    """A 404 for a probed repo is a programmer error (the orchestrator
    should call ``repo_exists`` first); we surface it as a typed
    :class:`TransportError` rather than swallowing as empty."""

    api = _make_api_mock()
    api.dataset_info.side_effect = _repo_not_found()
    transport = _make_transport(api)

    with pytest.raises(TransportError) as exc_info:
        transport.read_remote_card_data("alice/missing")
    assert "alice/missing" in str(exc_info.value)


def test_read_remote_card_data_passes_revision() -> None:
    """The Protocol allows reading a non-main revision; the adapter
    must thread it through."""

    api = _make_api_mock()
    api.dataset_info.return_value = _fake_dataset_info(None)
    transport = _make_transport(api)

    transport.read_remote_card_data("alice/x", revision="v1.0")
    api.dataset_info.assert_called_once_with("alice/x", revision="v1.0")


# ---------------------------------------------------------------------------
# upload_folder
# ---------------------------------------------------------------------------


def _fake_commit(sha: str) -> SimpleNamespace:
    """Build a stand-in ``GitCommitInfo`` with just ``commit_id``."""

    return SimpleNamespace(
        commit_id=sha,
        authors=["alice"],
        created_at=datetime(2026, 5, 6),
        title="upload",
        message="",
        formatted_title=None,
        formatted_message=None,
    )


def test_upload_folder_calls_upload_large_folder_with_dataset_type(
    tmp_path: Path,
) -> None:
    """Spec 10 § Tooling used: ``upload_large_folder`` is the upload
    primitive. The adapter must forward ``repo_type='dataset'`` and
    convert the ``Path`` to ``str`` (the SDK accepts either but we
    pin str for deterministic call-arg assertions)."""

    api = _make_api_mock()
    api.list_repo_commits.return_value = [_fake_commit("a" * 40)]
    transport = _make_transport(api)

    folder = tmp_path / "staging"
    folder.mkdir()
    info = transport.upload_folder(
        "alice/x",
        folder_path=folder,
        commit_message="custom msg",
    )

    api.upload_large_folder.assert_called_once_with(
        "alice/x",
        folder_path=str(folder),
        repo_type="dataset",
        revision="main",
    )
    # commit_message is intentionally NOT forwarded to upload_large_folder
    # (the chunked API does not accept one). Adapter behavior pinned by
    # checking the call kwargs do not contain it.
    call_kwargs = api.upload_large_folder.call_args.kwargs
    assert "commit_message" not in call_kwargs

    assert info.commit_sha == "a" * 40
    assert info.commit_url.endswith("/commit/" + "a" * 40)
    assert "alice/x" in info.commit_url


def test_upload_folder_returns_empty_sha_when_list_commits_fails(
    tmp_path: Path,
) -> None:
    """Upload itself succeeded, so a transient failure on the post-
    upload commit-probe must not invalidate the publish — we return an
    empty SHA and let the runner print a fallback message."""

    api = _make_api_mock()
    api.list_repo_commits.side_effect = _http_error("503", status=503)
    transport = _make_transport(api)

    folder = tmp_path / "staging"
    folder.mkdir()
    info = transport.upload_folder("alice/x", folder_path=folder, commit_message="")

    assert info.commit_sha == ""
    assert info.commit_url == ""


def test_upload_folder_returns_empty_sha_when_no_commits(
    tmp_path: Path,
) -> None:
    """An empty commit list (shouldn't happen in practice — we just
    uploaded — but defensive) yields empty SHA rather than IndexError."""

    api = _make_api_mock()
    api.list_repo_commits.return_value = []
    transport = _make_transport(api)

    folder = tmp_path / "staging"
    folder.mkdir()
    info = transport.upload_folder("alice/x", folder_path=folder, commit_message="")
    assert info.commit_sha == ""


def test_upload_folder_wraps_upload_errors(tmp_path: Path) -> None:
    """A failed ``upload_large_folder`` is the user-visible publish
    failure; wrap it as :class:`TransportError` so the CLI runner's
    exit-7 branch fires."""

    api = _make_api_mock()
    api.upload_large_folder.side_effect = _http_error("connection reset", status=500)
    transport = _make_transport(api)

    folder = tmp_path / "staging"
    folder.mkdir()
    with pytest.raises(TransportError) as exc_info:
        transport.upload_folder("alice/x", folder_path=folder, commit_message="")
    assert "alice/x" in str(exc_info.value)
    # The list_repo_commits probe must NOT run when the upload itself failed.
    api.list_repo_commits.assert_not_called()


def test_upload_folder_passes_revision(tmp_path: Path) -> None:
    """A non-default revision must thread through to both the upload
    and the post-upload commit probe."""

    api = _make_api_mock()
    api.list_repo_commits.return_value = [_fake_commit("b" * 40)]
    transport = _make_transport(api)

    folder = tmp_path / "staging"
    folder.mkdir()
    transport.upload_folder(
        "alice/x",
        folder_path=folder,
        commit_message="",
        revision="dev",
    )

    api.upload_large_folder.assert_called_once_with(
        "alice/x",
        folder_path=str(folder),
        repo_type="dataset",
        revision="dev",
    )
    api.list_repo_commits.assert_called_once_with("alice/x", repo_type="dataset", revision="dev")


# ---------------------------------------------------------------------------
# create_tag
# ---------------------------------------------------------------------------


def test_create_tag_forwards_message_as_tag_message() -> None:
    """``HfApi.create_tag`` calls the annotated-tag text ``tag_message``;
    the Protocol calls it ``message``. Verify the rename happens."""

    api = _make_api_mock()
    transport = _make_transport(api)

    transport.create_tag("alice/x", tag="v2026.05.05", revision="abc", message="release notes")
    api.create_tag.assert_called_once_with(
        "alice/x",
        repo_type="dataset",
        tag="v2026.05.05",
        revision="abc",
        tag_message="release notes",
        exist_ok=True,
    )


def test_create_tag_passes_none_message_through() -> None:
    """``message=None`` is the Protocol default for "lightweight tag";
    the SDK accepts ``tag_message=None`` to mean the same."""

    api = _make_api_mock()
    transport = _make_transport(api)

    transport.create_tag("alice/x", tag="v1")
    call_kwargs = api.create_tag.call_args.kwargs
    assert call_kwargs["tag_message"] is None


def test_create_tag_wraps_http_errors() -> None:
    """A tag conflict (different SHA) surfaces as ``HfHubHTTPError``;
    repackage so the runner's ``TransportError`` branch maps it to
    exit 7."""

    api = _make_api_mock()
    api.create_tag.side_effect = _http_error("tag conflict", status=409)
    transport = _make_transport(api)

    with pytest.raises(TransportError) as exc_info:
        transport.create_tag("alice/x", tag="v1")
    assert "v1" in str(exc_info.value) or "alice/x" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Constructor wiring
# ---------------------------------------------------------------------------


def test_constructor_builds_hf_api_with_token_when_api_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production callers don't pass ``api=...``; the constructor
    builds ``HfApi(token=token, library_name=...)`` itself. Verify the
    token + library_name reach ``HfApi``."""

    captured: dict[str, object] = {}

    def fake_hf_api(*args: object, **kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock(spec=HfApi)

    monkeypatch.setattr("pd_ocr_synth.publish.hf_hub_transport.HfApi", fake_hf_api)

    HfHubTransport(token="hf_constructor_test")

    assert captured["token"] == "hf_constructor_test"
    assert captured["library_name"] == "pd-ocr-synth"


# ---------------------------------------------------------------------------
# Token-redaction wiring
# ---------------------------------------------------------------------------
#
# Audit invariant from ``docs/specs/10-publishing.md`` § Authentication +
# the round-trip test in ``test_cli_publish_upload.py``: the resolved
# HF token must never reach stdout/stderr. The transport adapter's
# error wrappers run the message body through
# :func:`pd_ocr_synth.publish.redaction.redact_token` before raising —
# these tests lock that wiring, on each error branch, against
# regression. If a future refactor of the wrappers drops the
# redaction step, the tests below catch it without needing to wait for
# an end-to-end CLI run.


def _http_error_echoing_token(status: int = 401) -> HfHubHTTPError:
    """Build an ``HfHubHTTPError`` whose message echoes a fake HF
    token, simulating what the SDK can produce on a 401 response that
    surfaces the offending ``Authorization`` header.
    """

    return _http_error(
        "401 Unauthorized: Bearer hf_LEAKEDLEAKEDLEAKEDLEAKED rejected",
        status=status,
    )


def test_repo_exists_redacts_tokens_in_wrapped_error() -> None:
    """A 5xx (or any non-404) probe error that includes a token in its
    body must not surface that token through ``TransportError``."""

    api = _make_api_mock()
    api.repo_exists.side_effect = _http_error_echoing_token(status=500)
    transport = _make_transport(api)

    with pytest.raises(TransportError) as exc_info:
        transport.repo_exists("alice/x")
    msg = str(exc_info.value)
    assert "hf_LEAKEDLEAKEDLEAKEDLEAKED" not in msg
    assert "[redacted-hf-token]" in msg


def test_create_repo_redacts_tokens_in_wrapped_error() -> None:
    api = _make_api_mock()
    api.create_repo.side_effect = _http_error_echoing_token(status=403)
    transport = _make_transport(api)

    with pytest.raises(TransportError) as exc_info:
        transport.create_repo("alice/x", private=False)
    msg = str(exc_info.value)
    assert "hf_LEAKEDLEAKEDLEAKEDLEAKED" not in msg
    assert "[redacted-hf-token]" in msg


def test_upload_folder_redacts_tokens_in_wrapped_error(tmp_path: Path) -> None:
    api = _make_api_mock()
    api.upload_large_folder.side_effect = _http_error_echoing_token(status=500)
    transport = _make_transport(api)

    folder = tmp_path / "staging"
    folder.mkdir()
    with pytest.raises(TransportError) as exc_info:
        transport.upload_folder("alice/x", folder_path=folder, commit_message="")
    msg = str(exc_info.value)
    assert "hf_LEAKEDLEAKEDLEAKEDLEAKED" not in msg
    assert "[redacted-hf-token]" in msg


def test_create_tag_redacts_tokens_in_wrapped_error() -> None:
    api = _make_api_mock()
    api.create_tag.side_effect = _http_error_echoing_token(status=409)
    transport = _make_transport(api)

    with pytest.raises(TransportError) as exc_info:
        transport.create_tag("alice/x", tag="v1")
    msg = str(exc_info.value)
    assert "hf_LEAKEDLEAKEDLEAKEDLEAKED" not in msg
    assert "[redacted-hf-token]" in msg
