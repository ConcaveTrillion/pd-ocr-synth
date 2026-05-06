"""Default-transport factory for the real upload path (M08).

This module is the single seam where the CLI runner asks for an
:class:`HfTransport` to use against the live Hugging Face Hub. The
intent is that the *only* file that imports ``huggingface_hub`` is
the concrete adapter — everywhere else (the CLI, the orchestrator,
tests) drives against the :class:`HfTransport` Protocol.

The concrete ``HfHubTransport`` adapter is **not yet implemented**
(``huggingface_hub`` is deliberately not in ``pyproject.toml`` until
the milestone introducing it lands — see the repo CLAUDE.md). What
*is* implemented here:

1. :class:`SdkUnavailableError` — a typed, transport-shaped error
   that the CLI runner can map to the documented exit-7 publish
   failure when the SDK isn't installed yet.
2. :func:`make_default_transport` — the production factory. It
   tries to construct the SDK-backed transport and raises
   :class:`SdkUnavailableError` if the import fails. The CLI runner
   wraps this call and prints a clear remediation message ("install
   huggingface_hub" / "run via pipx with [hf] extra"). Tests inject
   a different factory (returning a :class:`FakeTransport`) to
   exercise the upload path hermetically.

Why a dedicated factory rather than a top-level ``import``: a top-
level import would make the publish package itself unimportable on
machines without the SDK, which would in turn break the dry-run
path (which has no SDK requirement). The factory pattern keeps the
import lazy and the failure mode typed.
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

    Currently raises :class:`SdkUnavailableError` unconditionally —
    the SDK adapter (the only file that imports ``huggingface_hub``)
    is the next chunk after this CLI wiring. Wiring the factory now
    lets the real-upload code path land and be fully testable via
    fakes; swapping in the live adapter is then a one-file change
    that doesn't need to revisit the CLI.

    Parameters
    ----------
    token:
        The resolved HF token (from
        :func:`pd_ocr_synth.publish.auth.resolve_hf_token`). The
        adapter will pass this to ``HfApi(token=...)``. We accept it
        eagerly here so the factory signature is final from day one
        — every transport implementation needs a token, and threading
        it through the factory keeps the CLI runner from re-resolving
        auth at construction time.

    Returns
    -------
    HfTransport
        The production transport.

    Raises
    ------
    SdkUnavailableError
        Today, always: the SDK adapter has not yet landed. Once it
        does, this branch only fires when ``huggingface_hub`` cannot
        be imported (uninstalled, broken environment, etc).
    """

    # Acknowledge the parameter so static analysis doesn't flag it as
    # unused while the adapter is pending. This will become
    # ``HfHubTransport(token=token)`` in the next chunk.
    _ = token
    raise SdkUnavailableError(
        "the Hugging Face SDK adapter is not yet available; "
        "install `huggingface_hub` and re-run, or use --dry-run to "
        "preview the upload plan without an SDK"
    )


__all__ = [
    "SdkUnavailableError",
    "make_default_transport",
]
