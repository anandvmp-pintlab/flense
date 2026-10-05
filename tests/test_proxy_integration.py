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
from flense.config import CodeWriterConfig, FlenseConfig


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
    cfg.code_writer = CodeWriterConfig(
        model="claude-haiku-4-5",
        fallback="gpt-4o-mini",
        output_dir=str(tmp_path / "generated"),
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

    async def test_missing_ref_file_header_returns_500(self, flense_app):
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

        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["type"] == "flense_code_writer_error"
        assert "X-Flense-Ref-File" in body["error"]["message"]

    async def test_ref_file_not_found_returns_500(self, flense_app):
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

        assert resp.status_code == 500
        body = resp.json()
        assert body["error"]["type"] == "flense_code_writer_error"

    async def test_explicit_output_file(self, flense_app, ref_file, tmp_path):
        generated = "# explicit output"
        mock_client = _mock_anthropic_client(generated)
        out = str(tmp_path / "explicit_output.py")

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
        assert resp.headers["x-flense-output-file"] == out
        assert Path(out).read_text() == generated

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
