"""Tests for telemetry-related config additions (headless, pricing)."""

import tempfile
from pathlib import Path

from flense.config import FlenseConfig, load_config


def test_default_headless_false():
    config = FlenseConfig()
    assert config.server.headless is False


def test_default_pricing_empty():
    config = FlenseConfig()
    for prov in config.providers.values():
        assert prov.pricing == {}


def test_parse_headless():
    toml_content = b"""
[server]
headless = true
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        config = load_config(Path(f.name))

    assert config.server.headless is True


def test_parse_pricing_dict():
    toml_content = b"""
[providers.openai.pricing]
"gpt-4o" = { input = 2.50, output = 10.00 }
"gpt-4o-mini" = { input = 0.15, output = 0.60 }
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        config = load_config(Path(f.name))

    openai = config.providers["openai"]
    assert openai.pricing["gpt-4o"] == 2.50
    assert openai.pricing["gpt-4o-mini"] == 0.15


def test_parse_pricing_scalar():
    toml_content = b"""
[providers.anthropic.pricing]
"claude-sonnet-4" = 3.00
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=False) as f:
        f.write(toml_content)
        f.flush()
        config = load_config(Path(f.name))

    anthropic = config.providers["anthropic"]
    assert anthropic.pricing["claude-sonnet-4"] == 3.00
