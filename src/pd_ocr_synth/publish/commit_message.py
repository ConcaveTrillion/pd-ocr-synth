"""Commit-message formatting for ``pd-ocr-synth publish`` (M08).

Per ``docs/specs/10-publishing.md`` § Versioning:

> ``--message "..."`` overrides the auto-generated commit message
> (default: ``pd-ocr-synth render @<recipe-sha>``).

This module centralizes the default message format so the CLI runner
and any future programmatic caller (e.g. a Python notebook) can build
the same line. Splitting it out as a tiny pure helper means the
formatter is unit-testable on its own and a future change to the
default format only edits one place.

The formatter is deliberately lenient about a missing recipe SHA: a
staging dir whose README front matter happens to be missing
``pd-ocr-recipe-sha`` (e.g. a hand-edited card) still gets a
defensible default message rather than an exception. The preflight
step is what enforces "the staging dir is publish-ready"; this
helper is one rung lower and just renders the string.
"""

from __future__ import annotations

# Centralize the default prefix so a future style tweak edits one
# place. The literal ``pd-ocr-synth render`` matches the spec's
# example verbatim and is what consumers (the trainer's HF source
# path) will grep for when they want to know "did synth produce
# this commit?".
_DEFAULT_PREFIX = "pd-ocr-synth render"


def default_commit_message(recipe_sha: str | None) -> str:
    """Render the spec's default commit message.

    Spec 10 § Versioning prescribes ``pd-ocr-synth render
    @<recipe-sha>``. The caller supplies the SHA (typically pulled
    from the staging README's ``pd-ocr-recipe-sha`` front-matter key
    via the preflight report); we format it.

    Parameters
    ----------
    recipe_sha:
        The hex digest of the recipe snapshot. ``None`` (or empty)
        falls back to the bare prefix without an ``@<sha>`` suffix
        — defensible default for a staging dir that for whatever
        reason lacks the recipe SHA. The preflight is what owns
        "is this staging dir publishable"; the formatter does not
        re-litigate that decision.

    Returns
    -------
    str
        The formatted commit message. Single-line; no trailing newline.
        HF's commit-message field is fine with longer multi-line
        bodies, but the spec's example is a single line and matching
        it keeps the grep-target simple.

    Examples
    --------
    >>> default_commit_message("a" * 64)
    'pd-ocr-synth render @aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    >>> default_commit_message(None)
    'pd-ocr-synth render'
    >>> default_commit_message("")
    'pd-ocr-synth render'
    """

    if not recipe_sha:
        return _DEFAULT_PREFIX
    return f"{_DEFAULT_PREFIX} @{recipe_sha}"


def resolve_commit_message(
    *,
    override: str | None,
    recipe_sha: str | None,
) -> str:
    """Pick between an explicit ``--message`` override and the default.

    Spec 10 § Versioning describes the ``--message`` override as
    "overrides the auto-generated commit message" — hence: if an
    override is supplied (and is non-empty), use it verbatim;
    otherwise build the default. A whitespace-only override is
    treated as missing so a shell that expanded an empty
    ``--message "$VAR"`` doesn't accidentally produce an empty
    commit message (which HF would reject).

    Parameters
    ----------
    override:
        The value of the CLI's ``--message`` flag, or ``None``.
        Whitespace-only values are treated as not provided, which
        falls through to the default.
    recipe_sha:
        Forwarded to :func:`default_commit_message` if the override
        is missing.

    Returns
    -------
    str
        The commit message the orchestrator should use.
    """

    if override is not None:
        stripped = override.strip()
        if stripped:
            return stripped
    return default_commit_message(recipe_sha)


__all__ = [
    "default_commit_message",
    "resolve_commit_message",
]
