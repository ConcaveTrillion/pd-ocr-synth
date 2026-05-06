"""Unit tests for the HF token resolver (M08).

Covers ``pd_ocr_synth.publish.auth``. The resolver is the publish
flow's first network-touching gate: it picks a token from the
documented chain (``--token`` flag → ``HF_TOKEN`` env →
``~/.cache/huggingface/token``) and produces a typed error naming
that chain when nothing resolves.

Tests treat the resolution order as a contract per
``docs/specs/10-publishing.md`` § Authentication and the matching
deliverable in ``docs/roadmap/08-publishing-hf.md``: ordering matters,
empty-string inputs must fall through (not silently succeed), and the
error message must name every step.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pd_ocr_synth.publish import (
    HF_TOKEN_ENV_VAR,
    AuthError,
    ResolvedToken,
    format_resolution_chain,
    resolve_hf_token,
)

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_token_file(home: Path, value: str) -> Path:
    """Populate ``<home>/.cache/huggingface/token`` and return the path.

    Mirrors the layout written by ``hf auth login`` so the resolver
    exercises its real lookup path. Returning the path lets the test
    assert against the same string the resolver would have probed.
    """

    cache = home / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / "token"
    path.write_text(value, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Resolution-order tests
# ---------------------------------------------------------------------------


def test_flag_takes_priority_over_env_and_cache(tmp_path: Path) -> None:
    """``--token`` wins even when env + cache are also populated."""

    _write_token_file(tmp_path, "hf_cache_token")
    resolved = resolve_hf_token(
        flag_token="hf_flag_token",
        env={HF_TOKEN_ENV_VAR: "hf_env_token"},
        home=tmp_path,
    )
    assert resolved == ResolvedToken(token="hf_flag_token", source="flag")


def test_env_used_when_flag_absent(tmp_path: Path) -> None:
    """Env wins over cache when the flag is not supplied."""

    _write_token_file(tmp_path, "hf_cache_token")
    resolved = resolve_hf_token(
        flag_token=None,
        env={HF_TOKEN_ENV_VAR: "hf_env_token"},
        home=tmp_path,
    )
    assert resolved == ResolvedToken(token="hf_env_token", source="env")


def test_cache_used_when_flag_and_env_absent(tmp_path: Path) -> None:
    """Cached token is the last fallback in the chain."""

    _write_token_file(tmp_path, "hf_cache_token")
    resolved = resolve_hf_token(flag_token=None, env={}, home=tmp_path)
    assert resolved == ResolvedToken(token="hf_cache_token", source="cache")


def test_cache_token_whitespace_is_stripped(tmp_path: Path) -> None:
    """Trailing newlines from ``echo $token > file`` must not leak through."""

    _write_token_file(tmp_path, "hf_cache_token\n")
    resolved = resolve_hf_token(flag_token=None, env={}, home=tmp_path)
    assert resolved.token == "hf_cache_token"


# ---------------------------------------------------------------------------
# Empty-input fall-through
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("flag_token", ["", "   ", "\t\n"])
def test_empty_flag_falls_through(tmp_path: Path, flag_token: str) -> None:
    """Whitespace-only ``--token`` values must fall through to env.

    Guards against a shell-expansion footgun: ``--token "$HF_TOKEN"``
    with ``HF_TOKEN`` unset would otherwise hand HF an empty string
    and trigger a confusing 401 instead of using the env / cache.
    """

    resolved = resolve_hf_token(
        flag_token=flag_token,
        env={HF_TOKEN_ENV_VAR: "hf_env_token"},
        home=tmp_path,
    )
    assert resolved == ResolvedToken(token="hf_env_token", source="env")


@pytest.mark.parametrize("env_value", ["", "   ", "\n"])
def test_empty_env_falls_through(tmp_path: Path, env_value: str) -> None:
    """Whitespace-only ``$HF_TOKEN`` must fall through to cache."""

    _write_token_file(tmp_path, "hf_cache_token")
    resolved = resolve_hf_token(
        flag_token=None,
        env={HF_TOKEN_ENV_VAR: env_value},
        home=tmp_path,
    )
    assert resolved == ResolvedToken(token="hf_cache_token", source="cache")


def test_empty_cache_file_falls_through_to_error(tmp_path: Path) -> None:
    """Whitespace-only cache file is treated as missing → AuthError."""

    _write_token_file(tmp_path, "   \n")
    with pytest.raises(AuthError):
        resolve_hf_token(flag_token=None, env={}, home=tmp_path)


# ---------------------------------------------------------------------------
# Error path
# ---------------------------------------------------------------------------


def test_no_token_anywhere_raises_auth_error(tmp_path: Path) -> None:
    """Missing across all three sources → ``AuthError``."""

    with pytest.raises(AuthError):
        resolve_hf_token(flag_token=None, env={}, home=tmp_path)


def test_auth_error_names_the_full_chain(tmp_path: Path) -> None:
    """Per spec: error message must name every resolution step.

    Locks the spec contract that the user can read the error and
    know exactly which knobs to turn — flag, env, or cache file.
    """

    with pytest.raises(AuthError) as excinfo:
        resolve_hf_token(flag_token=None, env={}, home=tmp_path)
    message = str(excinfo.value)
    assert "--token" in message
    assert HF_TOKEN_ENV_VAR in message
    expected_path = tmp_path / ".cache" / "huggingface" / "token"
    assert str(expected_path) in message


def test_auth_error_path_honors_hf_home(tmp_path: Path) -> None:
    """``$HF_HOME`` redirects the cache-file path the error reports."""

    custom_home = tmp_path / "custom-hf-home"
    with pytest.raises(AuthError) as excinfo:
        resolve_hf_token(
            flag_token=None,
            env={"HF_HOME": str(custom_home)},
            home=tmp_path,
        )
    assert str(custom_home / "token") in str(excinfo.value)


# ---------------------------------------------------------------------------
# HF_HOME override
# ---------------------------------------------------------------------------


def test_hf_home_redirects_cache_lookup(tmp_path: Path) -> None:
    """``$HF_HOME`` moves the cached-token file the resolver probes.

    Mirrors the SDK's behavior so users with a relocated HF cache
    don't get an inconsistent token resolution between ``hf`` and
    ``pd-ocr-synth``.
    """

    custom_home = tmp_path / "custom-hf-home"
    custom_home.mkdir()
    (custom_home / "token").write_text("hf_custom_home_token\n", encoding="utf-8")

    resolved = resolve_hf_token(
        flag_token=None,
        env={"HF_HOME": str(custom_home)},
        home=tmp_path,
    )
    assert resolved == ResolvedToken(token="hf_custom_home_token", source="cache")


def test_default_token_path_used_when_no_overrides(tmp_path: Path) -> None:
    """With no env + no ``home`` override, the documented default path applies.

    We can't mutate the real ``$HOME`` safely from tests, but we can
    assert the *explainer* points at the documented default — which
    is what the spec promises the user.
    """

    chain = format_resolution_chain()
    assert ".cache/huggingface/token" in chain


# ---------------------------------------------------------------------------
# Default-env behavior
# ---------------------------------------------------------------------------


def test_default_env_reads_os_environ(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling without ``env=`` reads ``os.environ``.

    The injectable ``env`` parameter exists for tests; production CLI
    code calls ``resolve_hf_token(flag_token=...)`` and trusts the
    real environment. Lock that path explicitly.
    """

    monkeypatch.setenv(HF_TOKEN_ENV_VAR, "hf_real_env_token")
    resolved = resolve_hf_token(flag_token=None, home=tmp_path)
    assert resolved == ResolvedToken(token="hf_real_env_token", source="env")


# ---------------------------------------------------------------------------
# Public explainer
# ---------------------------------------------------------------------------


def test_format_resolution_chain_lists_all_three_sources(tmp_path: Path) -> None:
    """The explainer lists flag, env var, and cache file in order."""

    explainer = format_resolution_chain(tmp_path / "tok")
    flag_idx = explainer.index("--token")
    env_idx = explainer.index(HF_TOKEN_ENV_VAR)
    cache_idx = explainer.index(str(tmp_path / "tok"))
    assert flag_idx < env_idx < cache_idx


def test_format_resolution_chain_default_path() -> None:
    """No path arg → documented default path appears."""

    explainer = format_resolution_chain()
    assert ".cache/huggingface/token" in explainer
