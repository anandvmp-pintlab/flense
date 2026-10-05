from __future__ import annotations

import uuid

from .base import ProviderAdapter


class AnthropicAdapter(ProviderAdapter):
    def rewrite_url(self, path: str) -> str:
        stripped = path.removeprefix("/anthropic")
        return f"{self.upstream}{stripped}"

    def make_ack_response(self, message: str, model: str | None) -> dict:
        return {
            "id": f"msg_flense_{uuid.uuid4().hex[:12]}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": message}],
            "model": model or "unknown",
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }

    def make_error_response(self, message: str) -> dict:
        return {
            "type": "error",
            "error": {"type": "api_error", "message": message},
        }
