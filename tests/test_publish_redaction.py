"""Unit tests for ``pd_ocr_synth.publish.redaction`` (M08).

The helper is a pure-string regex utility: every test exercises one
input/output mapping, no fixtures needed. The audit invariant the
module protects (token strings must never reach stdout/stderr) is
already exercised end-to-end in
``test_cli_publish_upload.py::test_real_upload_does_not_leak_token_into_stdout_or_stderr``;
these tests lock the helper's behaviour at finer granularity so the
end-to-end test stays diagnostic on regression rather than failing
opaquely on a string mismatch.
"""

from __future__ import annotations

from pd_ocr_synth.publish.redaction import REDACTED_SENTINEL, redact_token

# ---------------------------------------------------------------------------
# Happy paths: tokens are replaced
# ---------------------------------------------------------------------------


def test_redact_token_replaces_user_access_token() -> None:
    """``hf_<base62>`` is the modern HF user access-token format.
    Anything that matches must be replaced wholesale, not partially."""

    text = "401 Unauthorized: bad token hf_AAAAAAAAAAAAAAAAAAAAAA"
    result = redact_token(text)
    assert "hf_AAAAAAAAAAAAAAAAAAAAAA" not in result
    assert REDACTED_SENTINEL in result
    # The non-token prefix is preserved verbatim — the user still
    # gets the actionable error context.
    assert result.startswith("401 Unauthorized: bad token ")


def test_redact_token_replaces_legacy_org_token() -> None:
    """``api_org_<base62>`` is the older HF org-token format. Some
    SDK error paths still surface tokens shaped this way; we cover
    both prefixes so a refresh of the SDK doesn't surprise us."""

    text = "GET / -> 403: api_org_BBBBBBBBBBBBBBBBBBBBBBBB rejected"
    result = redact_token(text)
    assert "api_org_BBBBBBBBBBBBBBBBBBBBBBBB" not in result
    assert REDACTED_SENTINEL in result


def test_redact_token_replaces_multiple_tokens_in_one_string() -> None:
    """Multi-line stack traces can echo the same (or different)
    tokens at multiple call sites. Each must be redacted."""

    text = (
        "Authorization: Bearer hf_AAAAAAAAAAAAAAAAAAAAAA\n"
        "fallback header echo: hf_BBBBBBBBBBBBBBBBBBBBBB"
    )
    result = redact_token(text)
    assert "hf_AAAAAAAAAAAAAAAAAAAAAA" not in result
    assert "hf_BBBBBBBBBBBBBBBBBBBBBB" not in result
    assert result.count(REDACTED_SENTINEL) == 2


def test_redact_token_handles_long_realistic_token() -> None:
    """Realistic HF tokens are ~40 base62 chars after the prefix.
    Make sure the pattern's ``{20,}`` lower bound doesn't truncate
    them or leave a tail."""

    realistic = "hf_" + "x" * 37
    text = f"upload failed: token={realistic} expired"
    result = redact_token(text)
    assert realistic not in result
    assert REDACTED_SENTINEL in result
    assert "expired" in result


# ---------------------------------------------------------------------------
# No-op paths: non-token strings are preserved exactly
# ---------------------------------------------------------------------------


def test_redact_token_passes_through_clean_text() -> None:
    text = "404 Not Found: alice/dataset"
    assert redact_token(text) == text


def test_redact_token_does_not_match_short_hf_prefixed_strings() -> None:
    """A bare ``hf_short`` (e.g. an internal flag name) is not a
    token. The 20-char lower bound prevents over-eager redaction."""

    text = "see hf_dev for the dev fixture"
    assert redact_token(text) == text


def test_redact_token_is_idempotent() -> None:
    """Calling :func:`redact_token` twice on the same input must
    produce the same result. Any future change that makes the
    sentinel itself look like a token would break this — caught
    here at unit-test scope."""

    once = redact_token("err hf_AAAAAAAAAAAAAAAAAAAAAA bad")
    twice = redact_token(once)
    assert once == twice
    assert REDACTED_SENTINEL in twice


def test_redact_token_handles_empty_string() -> None:
    assert redact_token("") == ""


# ---------------------------------------------------------------------------
# Word boundaries: tokens embedded in identifiers should not match
# ---------------------------------------------------------------------------


def test_redact_token_respects_word_boundaries() -> None:
    """A 20+ char alphanumeric run that happens to contain ``hf_`` in
    the middle of a longer identifier must NOT be redacted — that
    would scrub legitimate path / id strings."""

    # "myhf_FAKETOKENHERELOOKSLIKEONE" embeds the prefix but the
    # boundary requirement means it should not match: ``\b`` requires
    # a word boundary immediately before ``hf_``.
    text = "see myhf_FAKETOKENHERELOOKSLIKEONE for context"
    assert redact_token(text) == text
