"""Telemetry: per-request logging, session aggregates, and response headers."""

from __future__ import annotations

import json
import logging
import sys
import threading
import time
from dataclasses import dataclass, field

from .compression.types import CompressionResult, Strategy
from .pricing import estimate_cost_saved

logger = logging.getLogger(__name__)


@dataclass
class RequestLog:
    """Snapshot of a single proxied request."""

    timestamp: float
    provider: str
    model: str | None
    strategy: Strategy
    tokens_before: int
    tokens_after: int
    tokens_saved: int
    est_cost_saved: float | None
    compression_time_ms: float
    status: str  # "ok" or "error"


@dataclass
class SessionStats:
    """Running aggregates for the current flense session.

    Thread-safe: all mutations go through a lock so the TUI can read
    from the main thread while the ASGI event loop writes from another.
    """

    total_requests: int = 0
    total_tokens_saved: int = 0
    total_cost_saved: float = 0.0
    by_provider: dict[str, _ProviderBucket] = field(default_factory=dict)
    by_strategy: dict[str, int] = field(default_factory=dict)
    request_log: list[RequestLog] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _listeners: list = field(default_factory=list, repr=False)

    def record(self, entry: RequestLog) -> None:
        with self._lock:
            self.total_requests += 1
            self.total_tokens_saved += entry.tokens_saved
            if entry.est_cost_saved is not None:
                self.total_cost_saved += entry.est_cost_saved

            # Per-provider
            bucket = self.by_provider.setdefault(
                entry.provider, _ProviderBucket()
            )
            bucket.requests += 1
            bucket.tokens_saved += entry.tokens_saved
            if entry.est_cost_saved is not None:
                bucket.cost_saved += entry.est_cost_saved

            # Per-strategy
            key = entry.strategy.value
            self.by_strategy[key] = self.by_strategy.get(key, 0) + 1

            self.request_log.append(entry)

        # Notify listeners outside the lock
        for callback in self._listeners:
            try:
                callback(entry)
            except Exception:
                logger.debug("Listener callback failed", exc_info=True)

    def add_listener(self, callback) -> None:
        self._listeners.append(callback)

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of session stats."""
        with self._lock:
            return {
                "total_requests": self.total_requests,
                "total_tokens_saved": self.total_tokens_saved,
                "total_cost_saved_usd": round(self.total_cost_saved, 6),
                "by_provider": {
                    name: {
                        "requests": b.requests,
                        "tokens_saved": b.tokens_saved,
                        "cost_saved_usd": round(b.cost_saved, 6),
                    }
                    for name, b in self.by_provider.items()
                },
                "by_strategy": dict(self.by_strategy),
            }


@dataclass
class _ProviderBucket:
    requests: int = 0
    tokens_saved: int = 0
    cost_saved: float = 0.0


def build_request_log(
    compression_result: CompressionResult,
    provider: str,
    model: str | None,
    pricing_overrides: dict[str, float] | None = None,
    status: str = "ok",
) -> RequestLog:
    """Build a RequestLog from a CompressionResult."""
    tokens_saved = max(
        0,
        compression_result.tokens_before - compression_result.tokens_after,
    )
    est_cost = None
    if model and tokens_saved > 0:
        est_cost = estimate_cost_saved(
            tokens_saved, model, provider, pricing_overrides
        )

    return RequestLog(
        timestamp=time.time(),
        provider=provider,
        model=model,
        strategy=compression_result.strategy_applied,
        tokens_before=compression_result.tokens_before,
        tokens_after=compression_result.tokens_after,
        tokens_saved=tokens_saved,
        est_cost_saved=est_cost,
        compression_time_ms=compression_result.compression_time_ms,
        status=status,
    )


def build_telemetry_headers(entry: RequestLog) -> dict[str, str]:
    """Build the X-Flense-* response headers from a RequestLog."""
    headers: dict[str, str] = {
        "X-Flense-Strategy": entry.strategy.value,
        "X-Flense-Tokens-Saved": str(entry.tokens_saved),
        "X-Flense-Compression-Time": f"{entry.compression_time_ms:.1f}",
    }
    if entry.est_cost_saved is not None:
        headers["X-Flense-Est-Savings"] = f"{entry.est_cost_saved:.6f}"
    return headers


def emit_headless_log(entry: RequestLog) -> None:
    """Write a structured JSON log line to stdout (headless mode)."""
    record = {
        "ts": entry.timestamp,
        "provider": entry.provider,
        "model": entry.model,
        "strategy": entry.strategy.value,
        "tokens_before": entry.tokens_before,
        "tokens_after": entry.tokens_after,
        "tokens_saved": entry.tokens_saved,
        "compression_time_ms": round(entry.compression_time_ms, 1),
        "status": entry.status,
    }
    if entry.est_cost_saved is not None:
        record["est_cost_saved_usd"] = round(entry.est_cost_saved, 6)
    sys.stdout.write(json.dumps(record) + "\n")
    sys.stdout.flush()
