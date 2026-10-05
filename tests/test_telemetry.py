import json

from flense.compression.types import CompressionResult, Strategy
from flense.telemetry import (
    RequestLog,
    SessionStats,
    build_request_log,
    build_telemetry_headers,
    emit_headless_log,
)


def _make_result(
    was_compressed: bool = True,
    tokens_before: int = 10000,
    tokens_after: int = 3000,
    strategy: Strategy = Strategy.AST,
    compression_time_ms: float = 12.5,
) -> CompressionResult:
    return CompressionResult(
        original_body=b"{}",
        compressed_body=b"{}",
        strategy_applied=strategy,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        blocks_compressed=2,
        blocks_total=3,
        compression_time_ms=compression_time_ms,
        was_compressed=was_compressed,
    )


class TestBuildRequestLog:
    def test_compressed_request(self):
        result = _make_result()
        entry = build_request_log(result, "anthropic", "claude-sonnet-4-20250514")
        assert entry.tokens_saved == 7000
        assert entry.est_cost_saved is not None
        assert entry.est_cost_saved > 0
        assert entry.provider == "anthropic"
        assert entry.model == "claude-sonnet-4-20250514"
        assert entry.strategy == Strategy.AST

    def test_passthrough_request(self):
        result = _make_result(
            was_compressed=False,
            tokens_before=100,
            tokens_after=100,
            strategy=Strategy.PASSTHROUGH,
        )
        entry = build_request_log(result, "openai", "gpt-4o")
        assert entry.tokens_saved == 0
        assert entry.est_cost_saved is None  # no savings when 0 tokens saved

    def test_unknown_model(self):
        result = _make_result()
        entry = build_request_log(result, "anthropic", "some-unknown-model")
        assert entry.tokens_saved == 7000
        assert entry.est_cost_saved is None  # can't price unknown model

    def test_no_model(self):
        result = _make_result()
        entry = build_request_log(result, "anthropic", None)
        assert entry.est_cost_saved is None

    def test_pricing_overrides(self):
        result = _make_result(tokens_before=1_000_000, tokens_after=0)
        entry = build_request_log(
            result, "anthropic", "custom-model",
            pricing_overrides={"custom-model": 10.0},
        )
        # 1M tokens * $10/M = $10.00
        assert entry.est_cost_saved is not None
        assert abs(entry.est_cost_saved - 10.0) < 1e-6


class TestBuildTelemetryHeaders:
    def test_headers_with_cost(self):
        result = _make_result()
        entry = build_request_log(result, "anthropic", "claude-sonnet-4-20250514")
        headers = build_telemetry_headers(entry)
        assert headers["X-Flense-Strategy"] == "ast"
        assert headers["X-Flense-Tokens-Saved"] == "7000"
        assert "X-Flense-Compression-Time" in headers
        assert "X-Flense-Est-Savings" in headers
        assert float(headers["X-Flense-Est-Savings"]) > 0

    def test_headers_without_cost(self):
        result = _make_result()
        entry = build_request_log(result, "anthropic", None)
        headers = build_telemetry_headers(entry)
        assert "X-Flense-Est-Savings" not in headers
        assert headers["X-Flense-Strategy"] == "ast"

    def test_passthrough_headers(self):
        result = _make_result(
            was_compressed=False,
            tokens_before=100,
            tokens_after=100,
            strategy=Strategy.PASSTHROUGH,
        )
        entry = build_request_log(result, "openai", "gpt-4o")
        headers = build_telemetry_headers(entry)
        assert headers["X-Flense-Strategy"] == "passthrough"
        assert headers["X-Flense-Tokens-Saved"] == "0"


class TestSessionStats:
    def test_record_and_snapshot(self):
        stats = SessionStats()
        result = _make_result()
        entry = build_request_log(result, "anthropic", "claude-sonnet-4-20250514")
        stats.record(entry)

        snap = stats.snapshot()
        assert snap["total_requests"] == 1
        assert snap["total_tokens_saved"] == 7000
        assert snap["total_cost_saved_usd"] > 0
        assert "anthropic" in snap["by_provider"]
        assert snap["by_provider"]["anthropic"]["requests"] == 1
        assert "ast" in snap["by_strategy"]
        assert snap["by_strategy"]["ast"] == 1

    def test_multiple_providers(self):
        stats = SessionStats()

        r1 = _make_result(tokens_before=10000, tokens_after=3000)
        stats.record(build_request_log(r1, "anthropic", "claude-sonnet-4"))

        r2 = _make_result(tokens_before=5000, tokens_after=2000)
        stats.record(build_request_log(r2, "openai", "gpt-4o"))

        snap = stats.snapshot()
        assert snap["total_requests"] == 2
        assert snap["total_tokens_saved"] == 10000
        assert "anthropic" in snap["by_provider"]
        assert "openai" in snap["by_provider"]

    def test_listener_called(self):
        stats = SessionStats()
        received = []
        stats.add_listener(lambda e: received.append(e))

        result = _make_result()
        entry = build_request_log(result, "anthropic", "claude-sonnet-4")
        stats.record(entry)

        assert len(received) == 1
        assert received[0] is entry

    def test_request_log_stored(self):
        stats = SessionStats()
        result = _make_result()
        entry = build_request_log(result, "anthropic", "claude-sonnet-4")
        stats.record(entry)
        assert len(stats.request_log) == 1


class TestHeadlessLog:
    def test_emit_json_line(self, capsys):
        result = _make_result()
        entry = build_request_log(result, "anthropic", "claude-sonnet-4-20250514")
        emit_headless_log(entry)

        captured = capsys.readouterr()
        line = json.loads(captured.out.strip())
        assert line["provider"] == "anthropic"
        assert line["model"] == "claude-sonnet-4-20250514"
        assert line["strategy"] == "ast"
        assert line["tokens_saved"] == 7000
        assert "est_cost_saved_usd" in line

    def test_emit_no_cost(self, capsys):
        result = _make_result()
        entry = build_request_log(result, "anthropic", None)
        emit_headless_log(entry)

        captured = capsys.readouterr()
        line = json.loads(captured.out.strip())
        assert "est_cost_saved_usd" not in line
