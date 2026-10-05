from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import CompressionConfig

# Hop-by-hop headers that must not be forwarded through a proxy.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "content-length",
        "transfer-encoding",
        "connection",
        "keep-alive",
        "upgrade",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
    }
)


class ProviderAdapter(ABC):
    """Base class for provider-specific adapters.

    Each adapter knows how to rewrite URLs and filter headers for its
    upstream API. In later phases, adapters will also own request parsing,
    token counting, and compression dispatch.
    """

    def __init__(
        self,
        name: str,
        upstream: str,
        compression_config: CompressionConfig | None = None,
    ) -> None:
        self.name = name
        self.upstream = upstream.rstrip("/")
        self.compression_config = compression_config

    @abstractmethod
    def rewrite_url(self, path: str) -> str:
        """Convert an inbound proxy path to the full upstream URL."""
        ...

    def filter_request_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Strip hop-by-hop headers, pass everything else through."""
        return {
            k: v
            for k, v in headers.items()
            if k.lower() not in _HOP_BY_HOP
        }

    def filter_response_headers(self, headers: dict[str, str]) -> dict[str, str]:
        """Strip hop-by-hop headers from the upstream response."""
        return {
            k: v
            for k, v in headers.items()
            if k.lower() not in _HOP_BY_HOP
        }

    @abstractmethod
    def make_ack_response(self, message: str, model: str | None) -> dict:
        """Build a minimal synthetic response for the code-writer bypass ACK."""
        ...
