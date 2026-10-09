"""End-to-end proxy integration tests for the code-writer bypass.

Uses httpx.AsyncClient with ASGITransport to drive the full FastAPI app
without starting a real server. The internal httpx_client (used by the
code_writer to call the cheap model) is replaced with a mock after lifespan
startup so no real API calls are made.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from httpx import ASGITransport

from flense.app import create_app
from flense.config import CodeWriterConfig, FlenseConfig, RetryConfig

# Zero-backoff retry policy so retry tests don't actually sleep.
_FAST_RETRY = RetryConfig(max_retries=2, backoff_base=0.0, backoff_max=0.0)


def _mock_anthropic_client(generated_code: str) -> AsyncMock:
    """Return a mock httpx_client that simulates a successful Anthropic response."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"content": [{"text": generated_code}]}

    client = AsyncMock()
    client.post.return_value = mock_resp
    return client


@pytest.fixture
def ref_file(tmp_path) -> Path:
    f = tmp_path / "reference.py"
    f.write_text("def existing(x):\n    return x * 2\n")
    return f


@pytest.fixture
def flense_app(tmp_path):
    cfg = FlenseConfig()
    cfg.retry = _FAST_RETRY
    cfg.code_writer = CodeWriterConfig(
        enabled=True,
        model="claude-haiku-4-5",
        fallback="gpt-4o-mini",
        output_dir=str(tmp_path / "generated"),
        allowed_ref_dir=str(tmp_path),
    )
    return create_app(cfg)


@pytest.fixture
def disabled_app(tmp_path):
    cfg = FlenseConfig()
    cfg.retry = _FAST_RETRY
    cfg.code_writer = CodeWriterConfig(
        enabled=False,
        output_dir=str(tmp_path / "generated"),
        allowed_ref_dir=str(tmp_path),
    )
    return create_app(cfg)


class TestCodeWriterProxyIntegration:
    async def test_successful_generation(self, flense_app, ref_file):
        generated = "def new_func(x):\n    return x + 1\n"
        mock_client = _mock_anthropic_client(generated)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=flense_app), base_url="http://test"
        ) as client:
            flense_app.state.httpx_client = mock_client

            resp = await client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-test",
                    "x-flense-strategy": "code-writer",
                    "x-flense-ref-file": str(ref_file),
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": "Write a function that adds 1"}],
                },
            )

        assert resp.status_code == 200
        body = resp.json()

        # Anthropic-shaped ACK
        assert body["type"] == "message"
        assert body["role"] == "assistant"
        assert "Generated and written to" in body["content"][0]["text"]

        # Telemetry headers
        assert resp.headers["x-flense-strategy"] == "code-writer"
        assert resp.headers["x-flense-tokens-saved"] != "0"
        output_path = resp.headers["x-flense-output-file"]
        assert output_path

        # File on disk
        assert Path(output_path).read_text() == generated

    async def test_missing_ref_file_header_returns_400(self, flense_app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=flense_app), base_url="http://test"
        ) as client:
            flense_app.state.httpx_client = AsyncMock()

            resp = await client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-test",
                    "x-flense-strategy": "code-writer",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": "Do something"}],
                },
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "flense_code_writer_error"
        assert "X-Flense-Ref-File" in body["error"]["message"]

    async def test_ref_file_not_found_returns_400(self, flense_app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=flense_app), base_url="http://test"
        ) as client:
            flense_app.state.httpx_client = AsyncMock()

            resp = await client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-test",
                    "x-flense-strategy": "code-writer",
                    "x-flense-ref-file": "/nonexistent/file.py",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": "Do something"}],
                },
            )

        assert resp.status_code == 400
        body = resp.json()
        assert body["error"]["type"] == "flense_code_writer_error"

    async def test_explicit_output_file(self, flense_app, ref_file, tmp_path):
        generated = "# explicit output"
        mock_client = _mock_anthropic_client(generated)
        out = str(tmp_path / "generated" / "explicit_output.py")

        async with httpx.AsyncClient(
            transport=ASGITransport(app=flense_app), base_url="http://test"
        ) as client:
            flense_app.state.httpx_client = mock_client

            resp = await client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-test",
                    "x-flense-strategy": "code-writer",
                    "x-flense-ref-file": str(ref_file),
                    "x-flense-output-file": out,
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": "Generate something"}],
                },
            )

        assert resp.status_code == 200
        assert Path(resp.headers["x-flense-output-file"]) == Path(out).resolve()
        assert Path(out).read_text() == generated

    async def test_disabled_returns_403(self, disabled_app, ref_file):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=disabled_app), base_url="http://test"
        ) as client:
            disabled_app.state.httpx_client = AsyncMock()

            resp = await client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-test",
                    "x-flense-strategy": "code-writer",
                    "x-flense-ref-file": str(ref_file),
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": "Write something"}],
                },
            )

        assert resp.status_code == 403
        assert resp.json()["error"]["type"] == "flense_code_writer_disabled"

    async def test_ref_file_outside_allowed_dir_returns_400(self, flense_app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app=flense_app), base_url="http://test"
        ) as client:
            flense_app.state.httpx_client = AsyncMock()

            resp = await client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-test",
                    "x-flense-strategy": "code-writer",
                    "x-flense-ref-file": "/etc/passwd",
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": "Leak secrets"}],
                },
            )

        assert resp.status_code == 400
        assert resp.json()["error"]["type"] == "flense_code_writer_error"

    async def test_session_stats_record_code_writer(self, flense_app, ref_file):
        generated = "x = 1"
        mock_client = _mock_anthropic_client(generated)

        async with httpx.AsyncClient(
            transport=ASGITransport(app=flense_app), base_url="http://test"
        ) as client:
            flense_app.state.httpx_client = mock_client

            await client.post(
                "/anthropic/v1/messages",
                headers={
                    "x-api-key": "sk-ant-test",
                    "x-flense-strategy": "code-writer",
                    "x-flense-ref-file": str(ref_file),
                    "anthropic-version": "2023-06-01",
                },
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 512,
                    "messages": [{"role": "user", "content": "Write something"}],
                },
            )

        snap = flense_app.state.session_stats.snapshot()
        assert snap["total_requests"] == 1
        assert snap["by_strategy"].get("code-writer", 0) == 1


class _FakeStreamResp:
    """Minimal stand-in for a streaming httpx.Response."""

    def __init__(self, status_code, chunks=(b"ok",), headers=None):
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks
        self.closed = False

    async def aiter_raw(self):
        for c in self._chunks:
            yield c

    async def aclose(self):
        self.closed = True


class TestProxyUpstreamErrors:
    async def test_upstream_connect_error_retries_then_502(self, flense_app):
        # send() always raises a connection error -> exhausts retries -> 502.
        bad_client = MagicMock()
        bad_client.build_request = MagicMock(return_value=MagicMock())
        bad_client.send = AsyncMock(side_effect=httpx.ConnectError("boom"))

        async with httpx.AsyncClient(
            transport=ASGITransport(app=flense_app), base_url="http://test"
        ) as client:
            flense_app.state.httpx_client = bad_client

            resp = await client.post(
                "/anthropic/v1/messages",
                headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 502
        body = resp.json()
        assert body["type"] == "error"
        assert body["error"]["type"] == "api_error"
        # max_retries=2 -> 3 total attempts
        assert bad_client.send.await_count == 3

    async def test_upstream_retries_then_succeeds(self, flense_app):
        # 503, then a successful stream -> client sees 200 and a retries header.
        client_mock = MagicMock()
        client_mock.build_request = MagicMock(return_value=MagicMock())
        client_mock.send = AsyncMock(
            side_effect=[
                _FakeStreamResp(503, headers={"retry-after": "0"}),
                _FakeStreamResp(200, chunks=(b"hello",)),
            ]
        )

        async with httpx.AsyncClient(
            transport=ASGITransport(app=flense_app), base_url="http://test"
        ) as client:
            flense_app.state.httpx_client = client_mock

            resp = await client.post(
                "/anthropic/v1/messages",
                headers={"x-api-key": "sk-ant-test", "anthropic-version": "2023-06-01"},
                json={
                    "model": "claude-haiku-4-5",
                    "max_tokens": 64,
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )

        assert resp.status_code == 200
        assert resp.content == b"hello"
        assert resp.headers["x-flense-upstream-retries"] == "1"
        assert client_mock.send.await_count == 2


class TestProxyAuth:
    def _app(self):
        cfg = FlenseConfig()
        cfg.server.auth_token = "s3cret"
        return create_app(cfg)

    async def test_missing_token_returns_401(self):
        app = self._app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            app.state.httpx_client = AsyncMock()
            resp = await client.post(
                "/anthropic/v1/messages",
                headers={"x-api-key": "sk-ant-test"},
                json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 401
        assert resp.json()["error"]["type"] == "flense_auth_error"

    async def test_wrong_token_returns_401(self):
        app = self._app()
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            app.state.httpx_client = AsyncMock()
            resp = await client.post(
                "/anthropic/v1/messages",
                headers={"x-api-key": "sk-ant-test", "x-flense-auth": "wrong"},
                json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]},
            )
        assert resp.status_code == 401

    async def test_correct_token_passes_auth(self):
        # A correct token gets past auth; we prove it by tripping the body-size
        # cap (413) instead of being rejected at auth (401).
        cfg = FlenseConfig()
        cfg.server.auth_token = "s3cret"
        cfg.server.max_body_bytes = 10
        app = create_app(cfg)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            app.state.httpx_client = AsyncMock()
            resp = await client.post(
                "/anthropic/v1/messages",
                headers={"x-api-key": "sk-ant-test", "x-flense-auth": "s3cret"},
                json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "a long body"}]},
            )
        assert resp.status_code == 413
        assert resp.json()["error"]["type"] == "flense_payload_too_large"


class TestProxyBodySizeCap:
    async def test_oversized_body_returns_413(self):
        cfg = FlenseConfig()
        cfg.server.max_body_bytes = 10
        app = create_app(cfg)
        async with httpx.AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as client:
            app.state.httpx_client = AsyncMock()
            resp = await client.post(
                "/anthropic/v1/messages",
                headers={"x-api-key": "sk-ant-test"},
                json={"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "x" * 1000}]},
            )
        assert resp.status_code == 413
        assert resp.json()["error"]["type"] == "flense_payload_too_large"
