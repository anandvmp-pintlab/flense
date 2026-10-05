from __future__ import annotations

import time
import uuid

from .base import ProviderAdapter


class OpenAIAdapter(ProviderAdapter):
    def rewrite_url(self, path: str) -> str:
        stripped = path.removeprefix("/openai")
        return f"{self.upstream}{stripped}"

    def make_ack_response(self, message: str, model: str | None) -> dict:
        return {
            "id": f"chatcmpl-flense-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model or "unknown",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }
