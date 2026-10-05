import tempfile
from pathlib import Path

from flense.config import FlenseConfig, load_config


def test_defaults():
    config = FlenseConfig()
    assert config.compression.threshold == 5000
    assert config.compression.strategy == "auto"
    assert config.server.port == 2912
    assert "anthropic" in config.providers
    assert "openai" in config.providers


def test_parse_compression_section():
    toml_content = b"""
[compression]
threshold = 3000
strategy = "ast"
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=True) as f:
        f.write(toml_content)
        f.flush()
        config = load_config(Path(f.name))

    assert config.compression.threshold == 3000
    assert config.compression.strategy == "ast"


def test_parse_provider_compression():
    toml_content = b"""
[providers.anthropic]
upstream = "https://api.anthropic.com"

[providers.anthropic.compression]
strategy = "ctags"
threshold = 2000
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=True) as f:
        f.write(toml_content)
        f.flush()
        config = load_config(Path(f.name))

    prov = config.providers["anthropic"]
    assert prov.compression is not None
    assert prov.compression.strategy == "ctags"
    assert prov.compression.threshold == 2000


def test_no_compression_section_uses_defaults():
    toml_content = b"""
[server]
port = 9000
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=True) as f:
        f.write(toml_content)
        f.flush()
        config = load_config(Path(f.name))

    assert config.compression.threshold == 5000
    assert config.compression.strategy == "auto"


def test_provider_without_compression_override():
    toml_content = b"""
[providers.anthropic]
upstream = "https://custom.endpoint.com"
"""
    with tempfile.NamedTemporaryFile(suffix=".toml", delete=True) as f:
        f.write(toml_content)
        f.flush()
        config = load_config(Path(f.name))

    assert config.providers["anthropic"].compression is None
