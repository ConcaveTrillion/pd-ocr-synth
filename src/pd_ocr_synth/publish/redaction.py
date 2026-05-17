"""Token redaction for upload-path error messages (M08).

Per ``docs/specs/10-publishing.md`` § Authentication and the audit
invariant exercised by ``test_cli_publish_upload.py``::

    assert secret not in captured.out
    assert secret not in captured.err

…the resolved Hugging Face token must never reach stdout or stderr.
:mod:`pd_ocr_synth.publish.auth` already handles the resolver side
(it returns a :class:`ResolvedToken` whose ``source`` is logged but
whose ``token`` value is not). The remaining gap is the *upload*
path: :class:`pd_ocr_synth.publish.hf_hub_transport.HfHubTransport`
repackages every ``huggingface_hub`` exception with ``f"...: {exc}"``,
and an ``HfHubHTTPError`` constructed from a 401/403 response can —
on some SDK versions and some HF deployments — echo the offending
``Authorization: Bearer hf_*`` header back into its ``str()``
rendering. That string then flows through the orchestrator and out
to stderr.

This module provides one tiny helper: :func:`redact_token` walks an
arbitrary string, replaces anything that looks like an HF token with
a fixed sentinel, and returns the cleaned string. Callers wrap it
around the message body before raising :class:`TransportError`.

## Why a regex rather than "hide the resolved token only"

The resolved token is the obvious target — strip its literal value
from every error message. But the SDK can also surface tokens that
*aren't* the one we resolved (think: a stale token in
``$HUGGING_FACE_HUB_TOKEN`` that the SDK noticed before our
overrider, or a multi-account user whose ``hf auth login`` cache has
a different token from ``$HF_TOKEN``). A pattern-based scrub catches
both: the resolver is no longer the single source of truth for
"what counts as a secret".

The regex is intentionally narrow:

* ``hf_[A-Za-z0-9]{20,}`` — current HF access-token scheme
  (``hf_`` prefix + ~36 base62 chars; we accept 20+ to allow for
  future shortening without re-tuning).
* ``api_org_[A-Za-z0-9]{20,}`` — legacy organisation-token scheme
  still surfaced by some SDK error paths.

Falsely redacting a non-token string that happens to start with
``hf_`` is acceptable: the resulting "[redacted-hf-token]" reads
unambiguously, and any operator who needs the raw error has the
original exception's ``__cause__`` chain via
``raise X(...) from exc``.

## Why not silently rewrite ``str(exc)`` at the SDK layer

Because the SDK's exception types are public API: monkey-patching
their ``__str__`` would be fragile across versions. The transport
adapter is the right seam — it already owns the SDK→Protocol
translation, so adding redaction at the same boundary keeps every
"text leaving our process" decision in one file.
"""

from __future__ import annotations

import re

# Sentinel emitted in place of any matched token. Single-form so log
# scrapers and tests can grep for one literal; the trailing
# ``-hf-token`` makes the redaction self-describing without echoing
# any of the original characters.
REDACTED_SENTINEL = "[redacted-hf-token]"


# Compiled once at import time. The patterns are anchored on the
# distinctive prefixes (``hf_``, ``api_org_``) so we don't sweep up
# arbitrary 20+ character base62 strings.
_TOKEN_PATTERN = re.compile(
    r"\b(?:hf_[A-Za-z0-9]{20,}|api_org_[A-Za-z0-9]{20,})\b",
)


def redact_token(text: str) -> str:
    """Return ``text`` with any HF-shaped tokens replaced by a sentinel.

    Intended for wrapping around the body of upload-path error
    messages immediately before they reach the user. Idempotent:
    calling :func:`redact_token` on already-redacted text is a no-op
    (the sentinel itself doesn't match the pattern).

    Parameters
    ----------
    text:
        Free-form message body. Typically ``str(exc)`` for an
        ``HfHubHTTPError`` we just caught, or the wrapper string we
        are about to ``raise TransportError(...)`` with.

    Returns:
    -------
    str
        Same string with every ``hf_<...>`` / ``api_org_<...>`` run
        of 20+ alphanumeric characters replaced by
        :data:`REDACTED_SENTINEL`.

    Examples:
    --------
    >>> redact_token("401 Unauthorized: bad token hf_AAAAAAAAAAAAAAAAAAAAAA")
    '401 Unauthorized: bad token [redacted-hf-token]'
    >>> redact_token("ok, no secrets here")
    'ok, no secrets here'
    """

    return _TOKEN_PATTERN.sub(REDACTED_SENTINEL, text)


__all__ = ["REDACTED_SENTINEL", "redact_token"]
