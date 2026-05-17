"""Hugging Face token resolution for ``pd-ocr-synth publish``.

Per ``docs/specs/10-publishing.md`` § Authentication and the matching
deliverable in ``docs/roadmap/08-publishing-hf.md``:

> Order: ``--token`` flag → ``HF_TOKEN`` env →
> ``~/.cache/huggingface/token``. Clear error message naming the
> resolution chain on failure.

This module implements *only* the resolution + error-formatting half
of auth. It deliberately does **not** import ``huggingface_hub``: the
helper is called early in the publish flow (often before any network
work) and we want token discovery to work in environments where the
SDK isn't installed yet — and to stay testable without it. The token
string is then handed to ``HfApi(token=...)`` by the upload-time code
that lands later in M08.

The exit-code mapping (publish-time auth failure → exit 7) is covered
by the CLI surface, which lands in a later chunk; the helper here only
raises a typed exception. Keeping the exception layer narrow means a
future ``--dry-run`` path can recover from "no token" without ever
touching the SDK.

## Why a fresh helper rather than ``huggingface_hub.HfFolder``

``huggingface_hub`` ships its own resolver
(``HfFolder.get_token`` / ``get_token``) that is *almost* what we want
but differs in two ways the spec calls out:

1. We want the ``--token`` CLI flag to take priority over the env
   var, which the SDK doesn't model — its resolver only sees the
   env + cached file.
2. The error message must name *all three* resolution steps so the
   user knows what to set. The SDK raises a generic "no token" with
   no actionable suggestions.

Rather than wrap the SDK and patch around those gaps, we own the
resolution chain explicitly. The SDK still runs at upload time with
the resolved token string.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

# Public env-var name. Spec 10 § Authentication uses ``HF_TOKEN``; the
# SDK also recognizes ``HUGGING_FACE_HUB_TOKEN`` as an alias. We honor
# only the documented name here so the resolution chain we report to
# users matches the spec verbatim — no surprise fallback channels.
HF_TOKEN_ENV_VAR = "HF_TOKEN"

# Default location of the on-disk cached token written by
# ``hf auth login`` / ``huggingface-cli login``. Honors the standard
# ``HF_HOME`` override so containers / CI runners that relocate the
# cache root work without environment hacks; falls back to the
# documented default (``~/.cache/huggingface/token``) otherwise.
_HF_HOME_ENV_VAR = "HF_HOME"
_DEFAULT_HF_HOME = Path("~/.cache/huggingface")
_TOKEN_FILENAME = "token"


class AuthError(Exception):
    """Raised when no Hugging Face token can be resolved.

    Distinct exception type so the future CLI's exit-code mapping can
    catch this specifically (publish auth failures → exit 7 per
    ``docs/specs/01-cli.md``) without swallowing unrelated errors.
    The message is pre-formatted to name every step of the resolution
    chain so the user knows exactly which knobs they can turn.
    """


@dataclass(frozen=True)
class ResolvedToken:
    """Result of a successful token resolution.

    The ``source`` field is informational — used by ``--dry-run`` and
    verbose logging to report *how* the token was found without
    leaking the value. It is **never** included in upload payloads or
    error messages.

    Attributes
    ----------
    token:
        The resolved token string. Treat as a secret — do not log or
        echo. Pass directly to ``HfApi(token=...)``.
    source:
        Human-readable label of the resolution step that produced
        the token. One of ``"flag"``, ``"env"``, or ``"cache"``.
    """

    token: str
    source: str


def resolve_hf_token(
    *,
    flag_token: str | None = None,
    env: dict[str, str] | None = None,
    home: Path | None = None,
) -> ResolvedToken:
    """Resolve a Hugging Face token following the documented chain.

    Resolution order, first hit wins:

    1. ``flag_token`` — typically the parsed ``--token`` CLI flag.
       Empty / whitespace-only strings are ignored (treated the same
       as "not passed") so a shell that expanded an empty variable
       doesn't silently produce an unauthenticated upload.
    2. ``$HF_TOKEN`` — recommended for CI and shell usage; same
       empty-string handling as the flag.
    3. ``<HF_HOME>/token`` (default ``~/.cache/huggingface/token``) —
       the file written by ``hf auth login``. Read fully, stripped of
       surrounding whitespace.

    Parameters
    ----------
    flag_token:
        Value of the ``--token`` CLI flag, or ``None`` if not
        supplied. Whitespace-only values are treated as missing.
    env:
        Environment mapping, defaulting to ``os.environ``. Injectable
        for tests so they don't leak the real shell environment in.
    home:
        Override for the user home directory used to locate the
        cached-token file. Injectable for tests; production code
        should leave this ``None`` so the standard
        ``HF_HOME`` / ``$HOME`` resolution applies.

    Returns
    -------
    ResolvedToken
        The token plus a label for the source.

    Raises
    ------
    AuthError
        If none of the three sources yields a non-empty token. The
        message names the full resolution chain so the user has a
        clear remediation path.
    """

    env_map = os.environ if env is None else env

    flag_value = _normalize(flag_token)
    if flag_value is not None:
        return ResolvedToken(token=flag_value, source="flag")

    env_value = _normalize(env_map.get(HF_TOKEN_ENV_VAR))
    if env_value is not None:
        return ResolvedToken(token=env_value, source="env")

    cache_path = _resolve_token_file(env_map, home)
    cache_value = _read_token_file(cache_path)
    if cache_value is not None:
        return ResolvedToken(token=cache_value, source="cache")

    raise AuthError(_no_token_message(cache_path))


def format_resolution_chain(token_file: Path | None = None) -> str:
    """Return the human-readable resolution-chain explainer.

    Used by :class:`AuthError`'s message and exposed publicly so the
    ``--dry-run`` summary and any future verbose-log path can echo the
    same wording without re-deriving it. Keeping a single source of
    truth means the spec-mandated "name the resolution chain" wording
    only ever needs to be edited in one place.

    Parameters
    ----------
    token_file:
        The cached-token path actually probed during resolution.
        ``None`` means use the documented default
        (``~/.cache/huggingface/token``) for the explainer — useful
        when no resolution has been attempted yet.
    """

    if token_file is None:
        token_file = _DEFAULT_HF_HOME.expanduser() / _TOKEN_FILENAME
    return (
        "No Hugging Face token found. Resolution order:\n"
        "  1. --token <token>\n"
        f"  2. ${HF_TOKEN_ENV_VAR} environment variable\n"
        f"  3. {token_file} (run `hf auth login` to populate)"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalize(value: str | None) -> str | None:
    """Treat ``None`` and whitespace-only strings as "not provided".

    Centralized because the same rule applies to both the CLI flag
    and the env var: a shell that expanded an empty variable into the
    flag (``--token "$HF_TOKEN"`` with ``HF_TOKEN`` unset) should fall
    through rather than silently authenticate with an empty string —
    HF would then reject the upload with a confusing 401.
    """

    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _resolve_token_file(env: Mapping[str, str], home: Path | None) -> Path:
    """Compute the path to the cached-token file.

    Honors ``$HF_HOME`` because the SDK does — keeping our resolver
    in sync with the SDK's behavior matters for users who already
    have a non-default HF cache root configured. ``home`` overrides
    the user-home base directory; production callers leave it
    ``None`` so ``Path.expanduser()`` applies (which itself honors
    ``$HOME``).
    """

    hf_home = env.get(_HF_HOME_ENV_VAR)
    if hf_home:
        base = Path(hf_home)
    elif home is not None:
        base = home / ".cache" / "huggingface"
    else:
        base = _DEFAULT_HF_HOME.expanduser()
    return base / _TOKEN_FILENAME


def _read_token_file(path: Path) -> str | None:
    """Read and normalize the cached-token file.

    Returns ``None`` for any of: missing file, unreadable file, file
    that exists but contains only whitespace. We deliberately do not
    distinguish those cases — the resolution chain is a soft
    fallback, and a corrupted cache file is no different to the user
    from a missing one. (A surfaced ``PermissionError`` here would
    just confuse a user whose real fix is "set ``HF_TOKEN``".)
    """

    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
        return None
    return _normalize(raw)


def _no_token_message(token_file: Path) -> str:
    """Build the ``AuthError`` message body."""

    return format_resolution_chain(token_file)
