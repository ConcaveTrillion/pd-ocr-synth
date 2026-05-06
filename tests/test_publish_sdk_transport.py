"""Unit tests for ``pd_ocr_synth.publish.sdk_transport`` (M08).

The factory's job is to return a real :class:`HfTransport` when the
SDK-backed adapter (:class:`pd_ocr_synth.publish.HfHubTransport`) is
importable, and to raise :class:`SdkUnavailableError` otherwise. These
tests pin both branches and the typing contract that
:class:`SdkUnavailableError` is a :class:`TransportError` so the CLI
runner's existing transport-error branch maps it to exit 7 without a
special case.
"""

from __future__ import annotations

import sys

import pytest

from pd_ocr_synth.publish.hf_hub_transport import HfHubTransport
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


def test_make_default_transport_returns_hf_hub_transport_when_sdk_present() -> None:
    """With ``huggingface_hub`` installed (the ``[publish]`` extra is
    in the all-dev group used by ``make ci``), the factory returns the
    real adapter rather than raising."""

    transport = make_default_transport("hf_fake_token")
    assert isinstance(transport, HfHubTransport)


def test_make_default_transport_raises_sdk_unavailable_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``huggingface_hub`` cannot be imported the factory must raise
    :class:`SdkUnavailableError` (a :class:`TransportError` subclass)
    with a message that names the optional-extra remediation. We
    simulate the import failure by stashing the adapter module out of
    ``sys.modules`` and pre-poisoning ``sys.modules`` so its re-import
    surfaces as ``ImportError``."""

    # Drop both the adapter module and its top-level dep so the lazy
    # ``from ... import HfHubTransport`` re-imports them.
    saved = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "pd_ocr_synth.publish.hf_hub_transport"
        or name == "huggingface_hub"
        or name.startswith("huggingface_hub.")
    }

    # Poison ``huggingface_hub`` so the re-import fails.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    try:
        with pytest.raises(SdkUnavailableError) as exc_info:
            make_default_transport("hf_fake_token")

        msg = str(exc_info.value)
        assert "pd-ocr-synth[publish]" in msg
        assert "--dry-run" in msg
    finally:
        # Restore: never leave ``huggingface_hub`` poisoned for
        # subsequent tests in the same worker.
        monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
        for name, mod in saved.items():
            sys.modules[name] = mod


def test_make_default_transport_accepts_token_string() -> None:
    """The factory signature is ``(token: str) -> HfTransport``; an
    empty string still produces an adapter (auth-time validation is
    HF's job, not the factory's)."""

    transport = make_default_transport("")
    assert isinstance(transport, HfHubTransport)


def test_sdk_transport_names_reexported_from_publish_package() -> None:
    """The CLI runner and any future programmatic caller imports from
    ``pd_ocr_synth.publish``; lock the names at the package surface."""

    from pd_ocr_synth import publish

    assert publish.make_default_transport is make_default_transport
    assert publish.SdkUnavailableError is SdkUnavailableError
