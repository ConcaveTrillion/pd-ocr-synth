"""Default-transport factory for the real upload path (M08).

This module is the single seam where the CLI runner asks for an
:class:`HfTransport` to use against the live Hugging Face Hub. The
intent is that the *only* file that imports ``huggingface_hub`` is
the concrete adapter — everywhere else (the CLI, the orchestrator,
tests) drives against the :class:`HfTransport` Protocol.

What lives here:

1. :class:`SdkUnavailableError` — a typed, transport-shaped error
   that the CLI runner maps to the documented exit-7 publish
   failure when ``huggingface_hub`` isn't installed (the package is
   shipped as an *optional* extra, ``pd-ocr-synth[publish]``, so a
   user who only renders locally never has to pull the SDK).
2. :func:`make_default_transport` — the production factory. It
   imports :mod:`pd_ocr_synth.publish.hf_hub_transport` lazily, which
   in turn imports ``huggingface_hub``; an ``ImportError`` on that
   path is repackaged as :class:`SdkUnavailableError`. The CLI runner
   wraps this call and prints a clear remediation message ("install
   pd-ocr-synth[publish]"). Tests inject a different factory
   (returning a :class:`FakeTransport`) to exercise the upload path
   hermetically.

Why a dedicated factory rather than a top-level ``import``: a
top-level import would make the publish package itself unimportable
on machines without the SDK, which would in turn break the dry-run
path (which has no SDK requirement) and ``pd-ocr-synth render``
(which never publishes). The factory pattern keeps the import lazy
and the failure mode typed.
"""

from __future__ import annotations

from pd_ocr_synth.publish.transport import HfTransport, TransportError


class SdkUnavailableError(TransportError):
    """Raised when ``huggingface_hub`` is required but not installed.

    Subclasses :class:`TransportError` so the CLI runner's existing
    transport-error branch (which maps to spec 01's exit-7 publish
    failure) catches it without a special case. The message names the
    install hint so the user knows the remediation.
    """


def make_default_transport(token: str) -> HfTransport:
    """Construct the production SDK-backed :class:`HfTransport`.

    Imports :mod:`pd_ocr_synth.publish.hf_hub_transport` lazily so
    machines without ``huggingface_hub`` installed can still import
    :mod:`pd_ocr_synth.publish` (and therefore run ``pd-ocr-synth
    render`` and ``pd-ocr-synth publish --dry-run``). An import
    failure is repackaged as :class:`SdkUnavailableError` so the CLI
    runner's existing :class:`TransportError` branch maps it to
    exit 7 with a remediation hint.

    Parameters
    ----------
    token:
        The resolved HF token (from
        :func:`pd_ocr_synth.publish.auth.resolve_hf_token`). Threaded
        into ``HfApi(token=...)`` by the adapter so per-call ``token=``
        plumbing isn't needed.

    Returns:
    -------
    HfTransport
        The production SDK-backed transport.

    Raises:
    ------
    SdkUnavailableError
        ``huggingface_hub`` cannot be imported (the ``[publish]``
        optional dependency was not installed, or the install is
        broken).
    """

    try:
        from pd_ocr_synth.publish.hf_hub_transport import HfHubTransport
    except ImportError as exc:
        # Repackage so the CLI runner only catches TransportError;
        # users see a clear message naming the optional-extra group.
        raise SdkUnavailableError(
            "the Hugging Face SDK is not installed; "
            "install with `pip install pd-ocr-synth[publish]` (or "
            "`uv sync --extra publish`), or use --dry-run to "
            "preview the upload plan without an SDK"
        ) from exc

    return HfHubTransport(token=token)


__all__ = [
    "SdkUnavailableError",
    "make_default_transport",
]
