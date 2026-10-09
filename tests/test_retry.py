import httpx
import pytest

from flense.config import RetryConfig
from flense.proxy import _backoff_delay, _send_upstream

# Zero-delay policy so tests don't sleep.
FAST = RetryConfig(max_retries=2, backoff_base=0.0, backoff_max=0.0)


class FakeResp:
    def __init__(self, status, headers=None):
        self.status_code = status
        self.headers = headers or {}
        self.closed = False

    async def aclose(self):
        self.closed = True


class FakeClient:
    """Returns/raises queued results from send(); records call count."""

    def __init__(self, results):
        self._results = list(results)
        self.sent = 0

    def build_request(self, **kwargs):
        return ("req", kwargs)

    async def send(self, req, stream=False):
        item = self._results[self.sent]
        self.sent += 1
        if isinstance(item, Exception):
            raise item
        return item


async def _run(client, cfg=FAST):
    return await _send_upstream(
        client,
        method="POST",
        url="https://api.example.com/v1",
        headers={},
        body=b"{}",
        retry_cfg=cfg,
        provider="anthropic",
    )


# --- _send_upstream -------------------------------------------------------

async def test_success_first_try():
    c = FakeClient([FakeResp(200)])
    resp, retries = await _run(c)
    assert resp.status_code == 200
    assert retries == 0
    assert c.sent == 1


async def test_retryable_status_then_success():
    c = FakeClient([FakeResp(503), FakeResp(200)])
    resp, retries = await _run(c)
    assert resp.status_code == 200
    assert retries == 1
    assert c.sent == 2


async def test_transport_error_then_success():
    c = FakeClient([httpx.ConnectError("boom"), FakeResp(200)])
    resp, retries = await _run(c)
    assert resp.status_code == 200
    assert retries == 1
    assert c.sent == 2


async def test_exhausts_transport_errors_raises():
    c = FakeClient([httpx.ConnectError("a"), httpx.ReadError("b"), httpx.ConnectError("c")])
    with pytest.raises(httpx.RequestError):
        await _run(c)
    assert c.sent == 3  # max_retries=2 -> 3 attempts


async def test_exhausts_retryable_status_passes_through():
    c = FakeClient([FakeResp(503), FakeResp(503), FakeResp(503)])
    resp, retries = await _run(c)
    assert resp.status_code == 503  # final upstream error is surfaced
    assert retries == 2
    assert c.sent == 3


async def test_non_retryable_status_not_retried():
    c = FakeClient([FakeResp(400), FakeResp(200)])
    resp, retries = await _run(c)
    assert resp.status_code == 400
    assert retries == 0
    assert c.sent == 1


async def test_max_retries_zero_disables():
    c = FakeClient([FakeResp(503), FakeResp(200)])
    resp, retries = await _run(c, cfg=RetryConfig(max_retries=0, backoff_base=0.0, backoff_max=0.0))
    assert resp.status_code == 503
    assert retries == 0
    assert c.sent == 1


# --- _backoff_delay -------------------------------------------------------

def test_backoff_honors_integer_retry_after():
    cfg = RetryConfig(backoff_max=5.0)
    assert _backoff_delay(0, cfg, "3") == 3.0
    assert _backoff_delay(0, cfg, "999") == 5.0  # capped at backoff_max


def test_backoff_ignores_non_numeric_retry_after():
    cfg = RetryConfig(backoff_base=1.0, backoff_max=8.0)
    d = _backoff_delay(0, cfg, "Wed, 21 Oct 2026 07:28:00 GMT")
    assert 0.0 <= d <= 1.0


def test_backoff_full_jitter_within_cap():
    cfg = RetryConfig(backoff_base=1.0, backoff_max=8.0)
    for attempt in range(4):
        cap = min(8.0, 1.0 * (2**attempt))
        for _ in range(20):
            assert 0.0 <= _backoff_delay(attempt, cfg, None) <= cap


def test_backoff_zero_config_is_zero():
    cfg = RetryConfig(backoff_base=0.0, backoff_max=0.0)
    assert _backoff_delay(3, cfg, None) == 0.0
