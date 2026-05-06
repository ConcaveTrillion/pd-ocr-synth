"""Unit tests for ``pd_ocr_synth.publish.sdk_transport`` (M08).

The SDK adapter has not yet landed (``huggingface_hub`` is
deliberately not in ``pyproject.toml`` until the milestone introducing
it lands; see the repo CLAUDE.md). Until it does, the production
factory raises :class:`SdkUnavailableError`. These tests pin the
typing contract (``SdkUnavailableError`` is a ``TransportError`` so
the CLI runner's existing transport-error branch catches it) and
the message hint so the user knows how to remediate.

Once the adapter lands the factory will succeed and these tests
become "the factory returns an HfTransport instance, not a fake".
That migration is a one-file change.
"""

from __future__ import annotations

import pytest

from pd_ocr_synth.publish.sdk_transport import (
    SdkUnavailableError,
    make_default_transport,
)
from pd_ocr_synth.publish.transport import TransportError


def test_sdk_unavailable_error_is_transport_error() -> None:
    """The CLI runner catches :class:`TransportError` to map publish
    failures to exit 7. :class:`SdkUnavailableError` MUST inherit
    from it so the existing branch handles "SDK not installed"
    without a special case."""

    assert issubclass(SdkUnavailableError, TransportError)


def test_make_default_transport_raises_until_adapter_lands() -> None:
    """Today the factory raises unconditionally — the SDK adapter
    is the next chunk after this CLI wiring. The error message must
    name the remediation (install the SDK, or use --dry-run)."""

    with pytest.raises(SdkUnavailableError) as exc_info:
        make_default_transport("hf_fake_token")

    msg = str(exc_info.value)
    # Must mention the install-able dependency and the dry-run alternative.
    assert "huggingface_hub" in msg
    assert "--dry-run" in msg


def test_make_default_transport_accepts_token_string() -> None:
    """The factory signature MUST be ``(token: str) -> HfTransport``.
    Tests that pass a token still get the expected error path; the
    signature is what the CLI runner depends on, not the success path
    (yet)."""

    # Empty token is still a string — the type contract holds even
    # though semantically the resolver rejects empty tokens.
    with pytest.raises(SdkUnavailableError):
        make_default_transport("")


def test_sdk_transport_names_reexported_from_publish_package() -> None:
    """The CLI runner and any future programmatic caller imports from
    ``pd_ocr_synth.publish``; lock the names at the package surface."""

    from pd_ocr_synth import publish

    assert publish.make_default_transport is make_default_transport
    assert publish.SdkUnavailableError is SdkUnavailableError
