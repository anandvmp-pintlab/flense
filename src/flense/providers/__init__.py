from __future__ import annotations

from flense.config import FlenseConfig

from .anthropic import AnthropicAdapter
from .base import ProviderAdapter
from .openai import OpenAIAdapter

_ADAPTER_CLASSES: dict[str, type[ProviderAdapter]] = {
    "anthropic": AnthropicAdapter,
    "openai": OpenAIAdapter,
}

_registry: dict[str, ProviderAdapter] = {}


def register_providers(config: FlenseConfig) -> None:
    """Build adapter instances from config and populate the registry."""
    _registry.clear()
    for name, prov_config in config.providers.items():
        cls = _ADAPTER_CLASSES.get(name)
        if cls is not None:
            _registry[name] = cls(
                name=name,
                upstream=prov_config.upstream,
                compression_config=prov_config.compression,
            )


def get_adapter(provider_name: str) -> ProviderAdapter | None:
    return _registry.get(provider_name)
