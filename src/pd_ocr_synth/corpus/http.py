"""HTTP client factory + per-host rate limiter.

The corpus layer uses one ``httpx.Client`` per fetch run. The factory
configures a polite default user agent, exponential backoff retries on
transient failures, and a sliding per-host minimum interval between
requests so we don't hammer small archives like CELT.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from pd_ocr_synth import __version__

DEFAULT_USER_AGENT = (
    f"pd-ocr-synth/{__version__} (+https://github.com/ConcaveTrillion/pd-ocr-synth)"
)
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_RETRIES = 3
DEFAULT_PER_HOST_INTERVAL_S = 1.0
TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504})


@dataclass(slots=True)
class HostRateLimiter:
    """Block until at least ``min_interval_s`` has passed for the host.

    Thread-safe: the corpus layer is sync today, but a future parallel
    fetch implementation gets safety for free.
    """

    min_interval_s: float = DEFAULT_PER_HOST_INTERVAL_S
    _last_request_at: dict[str, float] | None = None
    _lock: threading.Lock | None = None

    def __post_init__(self) -> None:
        self._last_request_at = {}
        self._lock = threading.Lock()

    def wait(self, url: str, *, sleep: Callable[[float], None] = time.sleep) -> None:
        if self.min_interval_s <= 0:
            return
        host = urlparse(url).netloc or "_default"
        assert self._last_request_at is not None
        assert self._lock is not None
        with self._lock:
            now = time.monotonic()
            previous = self._last_request_at.get(host)
            if previous is not None:
                wait_for = self.min_interval_s - (now - previous)
                if wait_for > 0:
                    sleep(wait_for)
                    now = time.monotonic()
            self._last_request_at[host] = now


def build_client(
    *,
    user_agent: str = DEFAULT_USER_AGENT,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Create a configured ``httpx.Client``.

    ``transport`` is exposed mainly for tests, which wire an
    ``httpx.MockTransport`` to keep fetch tests offline. In production
    the default httpx transport is fine.
    """

    return httpx.Client(
        headers={"User-Agent": user_agent},
        timeout=timeout_s,
        follow_redirects=True,
        transport=transport,
    )


def get_with_retries(
    client: httpx.Client,
    url: str,
    *,
    retries: int = DEFAULT_RETRIES,
    rate_limiter: HostRateLimiter | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """GET ``url`` with exponential backoff on transient failures.

    Retries cover ``httpx.RequestError`` (connect / read errors) and
    HTTP statuses in ``TRANSIENT_STATUSES`` (408, 425, 429, 5xx).
    Non-transient errors raise via ``raise_for_status`` immediately.
    """

    if rate_limiter is None:
        rate_limiter = HostRateLimiter()

    last_exc: Exception | None = None
    last_response: httpx.Response | None = None
    for attempt in range(retries + 1):
        rate_limiter.wait(url, sleep=sleep)
        try:
            response = client.get(url)
        except httpx.RequestError as exc:
            last_exc = exc
            last_response = None
        else:
            if response.status_code not in TRANSIENT_STATUSES:
                response.raise_for_status()
                return response
            last_exc = None
            last_response = response

        if attempt < retries:
            sleep(_backoff_seconds(attempt))
            continue
        if last_response is not None:
            last_response.raise_for_status()
            # Defensive: raise_for_status only raises for 4xx/5xx; if
            # somehow a 408 sneaks through without an HTTPStatusError,
            # bail explicitly.
            raise httpx.HTTPStatusError(
                f"transient status {last_response.status_code} after {retries} retries",
                request=last_response.request,
                response=last_response,
            )
        if last_exc is not None:
            raise last_exc
    raise RuntimeError("get_with_retries fell through without an outcome")  # pragma: no cover


def _backoff_seconds(attempt: int) -> float:
    """Exponential backoff: 0.5, 1.0, 2.0, 4.0 ... seconds."""
    return 0.5 * (2**attempt)
