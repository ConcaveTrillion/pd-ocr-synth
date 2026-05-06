"""Unit tests for ``pd_ocr_synth.publish.commit_message`` (M08).

Per ``docs/specs/10-publishing.md`` § Versioning, the default commit
message is ``pd-ocr-synth render @<recipe-sha>`` and ``--message
<MSG>`` overrides it. These tests pin the exact format and the
override semantics so a regression in either direction is caught
locally without re-running the heavier CLI integration tests.
"""

from __future__ import annotations

from pd_ocr_synth.publish.commit_message import (
    default_commit_message,
    resolve_commit_message,
)

# ---------------------------------------------------------------------------
# default_commit_message
# ---------------------------------------------------------------------------


def test_default_commit_message_uses_spec_format() -> None:
    """Spec 10 § Versioning literal: ``pd-ocr-synth render @<recipe-sha>``."""

    sha = "a" * 64
    assert default_commit_message(sha) == f"pd-ocr-synth render @{sha}"


def test_default_commit_message_drops_at_when_sha_missing() -> None:
    """A staging dir without a recipe SHA still gets a defensible
    default message — preflight is what enforces "publishable", not
    the formatter."""

    assert default_commit_message(None) == "pd-ocr-synth render"
    assert default_commit_message("") == "pd-ocr-synth render"


# ---------------------------------------------------------------------------
# resolve_commit_message
# ---------------------------------------------------------------------------


def test_resolve_commit_message_prefers_override() -> None:
    """``--message "..."`` wins over the auto-generated default."""

    out = resolve_commit_message(
        override="custom human message",
        recipe_sha="a" * 64,
    )
    assert out == "custom human message"


def test_resolve_commit_message_falls_back_to_default_when_override_none() -> None:
    out = resolve_commit_message(override=None, recipe_sha="b" * 64)
    assert out == f"pd-ocr-synth render @{'b' * 64}"


def test_resolve_commit_message_treats_blank_override_as_missing() -> None:
    """A shell that expanded an empty ``--message "$VAR"`` should
    fall through to the default — HF rejects empty commit messages."""

    out = resolve_commit_message(override="   ", recipe_sha="c" * 64)
    assert out == f"pd-ocr-synth render @{'c' * 64}"


def test_resolve_commit_message_strips_whitespace_from_override() -> None:
    """Surrounding whitespace in a non-empty override is trimmed —
    matches HF's commit-message hygiene."""

    out = resolve_commit_message(override="  hello world  ", recipe_sha=None)
    assert out == "hello world"


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_commit_message_helpers_reexported_from_publish_package() -> None:
    """The CLI runner imports from ``pd_ocr_synth.publish``; make sure
    the new names land there."""

    from pd_ocr_synth import publish

    assert publish.default_commit_message is default_commit_message
    assert publish.resolve_commit_message is resolve_commit_message
