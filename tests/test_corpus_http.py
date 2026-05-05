"""Tests for the HTTP infra in ``pd_ocr_synth.corpus.http``."""

from __future__ import annotations

from collections.abc import Callable

import httpx
import pytest

from pd_ocr_synth.corpus.http import (
    DEFAULT_USER_AGENT,
    HostRateLimiter,
    build_client,
    get_with_retries,
)


def _client_with(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    return build_client(transport=httpx.MockTransport(handler))


def test_user_agent_is_set() -> None:
    captured: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request.headers["user-agent"])
        return httpx.Response(200, text="ok")

    with _client_with(handler) as client:
        client.get("https://example.com/")
    assert captured == [DEFAULT_USER_AGENT]


def test_get_with_retries_succeeds_first_try() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        return httpx.Response(200, text="hello")

    with _client_with(handler) as client:
        response = get_with_retries(client, "https://example.com/x", retries=2)
    assert response.status_code == 200
    assert response.text == "hello"
    assert len(calls) == 1


def test_get_with_retries_recovers_from_transient_5xx() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        if len(calls) <= 2:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, text="finally")

    sleeps: list[float] = []
    with _client_with(handler) as client:
        response = get_with_retries(
            client,
            "https://example.com/x",
            retries=3,
            sleep=sleeps.append,
        )
    assert response.status_code == 200
    assert response.text == "finally"
    assert len(calls) == 3
    # Two retries → two backoff sleeps. Plus per-host rate limiter
    # sleeps are zero-ish on first request.
    assert sum(1 for s in sleeps if s > 0) >= 2


def test_get_with_retries_gives_up_after_attempts() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="busy")

    with _client_with(handler) as client, pytest.raises(httpx.HTTPStatusError):
        get_with_retries(client, "https://example.com/", retries=1, sleep=lambda _: None)


def test_get_with_retries_does_not_retry_4xx_non_transient() -> None:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(404, text="missing")

    with _client_with(handler) as client, pytest.raises(httpx.HTTPStatusError):
        get_with_retries(client, "https://example.com/", retries=3, sleep=lambda _: None)
    assert len(calls) == 1


def test_rate_limiter_sleeps_on_repeat_host() -> None:
    sleeps: list[float] = []
    rl = HostRateLimiter(min_interval_s=0.5)
    rl.wait("https://example.com/a", sleep=sleeps.append)
    rl.wait("https://example.com/b", sleep=sleeps.append)
    # The second call should request a positive sleep duration.
    assert any(s > 0 for s in sleeps)


def test_rate_limiter_does_not_block_different_hosts() -> None:
    sleeps: list[float] = []
    rl = HostRateLimiter(min_interval_s=0.5)
    rl.wait("https://a.example.com/", sleep=sleeps.append)
    rl.wait("https://b.example.com/", sleep=sleeps.append)
    assert all(s == 0 for s in sleeps if s > 0) or not any(s > 0 for s in sleeps)
    # Stronger: there should be no positive sleeps at all.
    assert not any(s > 0 for s in sleeps)
