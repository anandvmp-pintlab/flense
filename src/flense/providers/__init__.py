from __future__ import annotations

from flense.config import FlenseConfig

from .anthropic import AnthropicAdapter
from .base import ProviderAdapter
from .mistral import MistralAdapter
from .openai import OpenAIAdapter

_ADAPTER_CLASSES: dict[str, type[ProviderAdapter]] = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
    "mistral": MistralAdapter,
}


def build_adapters(config: FlenseConfig) -> dict[str, ProviderAdapter]:
    """Build adapter instances from config.

    Returns a fresh dict so each app owns its own adapters (no shared global
    state across apps in the same process).
    """
    adapters: dict[str, ProviderAdapter] = {}
    for name, prov_config in config.providers.items():
        cls = _ADAPTER_CLASSES.get(name)
        if cls is not None:
            adapters[name] = cls(
                name=name,
                upstream=prov_config.upstream,
                compression_config=prov_config.compression,
            )
    return adapters
