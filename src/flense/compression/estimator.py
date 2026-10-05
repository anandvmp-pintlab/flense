from __future__ import annotations

import tiktoken

_encodings: dict[str, tiktoken.Encoding] = {}


def _get_encoding(provider: str) -> tiktoken.Encoding:
    """Return the appropriate tiktoken encoding for a provider."""
    if provider not in _encodings:
        # cl100k_base is a reasonable approximation for both Anthropic and OpenAI.
        _encodings[provider] = tiktoken.get_encoding("cl100k_base")
    return _encodings[provider]


def estimate_tokens(text: str, provider: str = "anthropic") -> int:
    """Estimate the token count for a string."""
    enc = _get_encoding(provider)
    return len(enc.encode(text, disallowed_special=()))


def estimate_messages_tokens(messages: list[dict], provider: str = "anthropic") -> int:
    """Estimate total tokens across a messages array.

    Handles string content and list-of-blocks content (Anthropic and OpenAI
    formats), including tool result blocks. Adds a per-message overhead of
    ~4 tokens for role/separators.
    """
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        content = msg.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content, provider)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        total += estimate_tokens(text, provider)
                    # Anthropic tool_result blocks
                    if block.get("type") == "tool_result":
                        inner = block.get("content", "")
                        if isinstance(inner, str):
                            total += estimate_tokens(inner, provider)
                        elif isinstance(inner, list):
                            for sub in inner:
                                if isinstance(sub, dict) and sub.get("text"):
                                    total += estimate_tokens(sub["text"], provider)
    return total
