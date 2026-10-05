"""Baked-in model pricing and cost estimation.

Prices are USD per million tokens. Only input pricing is needed for
savings calculations since compression affects input tokens only.
Config overrides are applied on top of these defaults.
"""

from __future__ import annotations

# {model_prefix: input_price_per_million_tokens}
# Prefixes are matched greedily so "claude-sonnet-4" matches before "claude".
_ANTHROPIC_PRICING: dict[str, float] = {
    "claude-opus-4": 15.00,
    "claude-sonnet-4": 3.00,
    "claude-haiku-4": 1.00,
    "claude-3-5-sonnet": 3.00,
    "claude-3-5-haiku": 0.80,
    "claude-3-opus": 15.00,
    "claude-3-sonnet": 3.00,
    "claude-3-haiku": 0.25,
}

_OPENAI_PRICING: dict[str, float] = {
    "gpt-4o-mini": 0.15,
    "gpt-4o": 2.50,
    "gpt-4-turbo": 10.00,
    "gpt-4": 30.00,
    "gpt-3.5-turbo": 0.50,
    "o3-mini": 1.10,
    "o3": 10.00,
    "o1-mini": 3.00,
    "o1": 15.00,
}

_PROVIDER_PRICING: dict[str, dict[str, float]] = {
    "anthropic": _ANTHROPIC_PRICING,
    "openai": _OPENAI_PRICING,
}


def lookup_input_price(
    model: str,
    provider: str,
    overrides: dict[str, float] | None = None,
) -> float | None:
    """Return the input price per million tokens for a model, or None if unknown.

    Checks config overrides first, then baked-in defaults. Model names are
    matched by prefix so "claude-sonnet-4-20250514" hits "claude-sonnet-4".
    """
    if overrides:
        price = _match_prefix(model, overrides)
        if price is not None:
            return price

    builtin = _PROVIDER_PRICING.get(provider, {})
    return _match_prefix(model, builtin)


def estimate_cost_saved(
    tokens_saved: int,
    model: str,
    provider: str,
    overrides: dict[str, float] | None = None,
) -> float | None:
    """Estimate USD saved from compressing input tokens.

    Returns None if the model's pricing is unknown.
    """
    price = lookup_input_price(model, provider, overrides)
    if price is None:
        return None
    return tokens_saved * price / 1_000_000


def _match_prefix(model: str, prices: dict[str, float]) -> float | None:
    """Find the longest prefix match for a model name."""
    best: str | None = None
    for prefix in prices:
        if model.startswith(prefix):
            if best is None or len(prefix) > len(best):
                best = prefix
    return prices[best] if best is not None else None
