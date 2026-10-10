from __future__ import annotations

from .openai import OpenAIAdapter


class MistralAdapter(OpenAIAdapter):
    """Mistral adapter.

    Mistral's API is OpenAI-compatible (same `/v1/chat/completions` request and
    response shape, `Authorization: Bearer` auth), so this reuses the OpenAI
    adapter's ACK/error shaping and only changes the URL prefix and upstream.
    """

    def rewrite_url(self, path: str) -> str:
        stripped = path.removeprefix("/mistral")
        return f"{self.upstream}{stripped}"
