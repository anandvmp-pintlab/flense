from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 2912
    headless: bool = False


@dataclass
class CompressionConfig:
    threshold: int = 5000
    strategy: str = "auto"


@dataclass
class CodeWriterConfig:
    # Opt-in: the code-writer bypass reads and writes files on the host, so it
    # is disabled unless explicitly enabled in config.
    enabled: bool = False
    model: str = "claude-haiku-4-5"
    fallback: str = "gpt-4o-mini"
    output_dir: str = "./generated"
    # Reference files must resolve within this directory (defends against
    # arbitrary file reads via the X-Flense-Ref-File header).
    allowed_ref_dir: str = "."
    # Maximum size of a reference file that will be read and sent upstream.
    max_ref_bytes: int = 1_000_000


@dataclass
class ProviderConfig:
    upstream: str = ""
    compression: CompressionConfig | None = None
    pricing: dict[str, float] = field(default_factory=dict)


@dataclass
class FlenseConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    providers: dict[str, ProviderConfig] = field(default_factory=dict)
    code_writer: CodeWriterConfig = field(default_factory=CodeWriterConfig)

    def __post_init__(self) -> None:
        defaults = {
            "anthropic": ProviderConfig(upstream="https://api.anthropic.com"),
            "openai": ProviderConfig(upstream="https://api.openai.com"),
        }
        for name, default in defaults.items():
            if name not in self.providers:
                self.providers[name] = default


_SEARCH_PATHS = [
    Path("flense.toml"),
    Path.home() / ".flense" / "flense.toml",
]


def load_config(path: Path | None = None) -> FlenseConfig:
    """Load config from a TOML file. Returns defaults if no file found."""
    if path is not None:
        return _parse_file(path)

    for candidate in _SEARCH_PATHS:
        if candidate.is_file():
            return _parse_file(candidate)

    return FlenseConfig()


def _parse_file(path: Path) -> FlenseConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    server_raw = raw.get("server", {})
    server = ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=server_raw.get("port", 2912),
        headless=server_raw.get("headless", False),
    )

    comp_raw = raw.get("compression", {})
    compression = CompressionConfig(
        threshold=comp_raw.get("threshold", 5000),
        strategy=comp_raw.get("strategy", "auto"),
    )

    providers: dict[str, ProviderConfig] = {}
    for name, prov_raw in raw.get("providers", {}).items():
        prov_comp_raw = prov_raw.get("compression")
        prov_compression = None
        if prov_comp_raw is not None:
            prov_compression = CompressionConfig(
                threshold=prov_comp_raw.get("threshold", 5000),
                strategy=prov_comp_raw.get("strategy", "auto"),
            )
        # Pricing overrides: {"model": {"input": X, "output": Y}} -> {"model": X}
        pricing_raw = prov_raw.get("pricing", {})
        pricing: dict[str, float] = {}
        for model_name, price_data in pricing_raw.items():
            if isinstance(price_data, dict):
                pricing[model_name] = float(price_data.get("input", 0))
            elif isinstance(price_data, (int, float)):
                pricing[model_name] = float(price_data)

        providers[name] = ProviderConfig(
            upstream=prov_raw.get("upstream", ""),
            compression=prov_compression,
            pricing=pricing,
        )

    cw_raw = raw.get("code_writer", {})
    code_writer = CodeWriterConfig(
        enabled=cw_raw.get("enabled", False),
        model=cw_raw.get("model", "claude-haiku-4-5"),
        fallback=cw_raw.get("fallback", "gpt-4o-mini"),
        output_dir=cw_raw.get("output_dir", "./generated"),
        allowed_ref_dir=cw_raw.get("allowed_ref_dir", "."),
        max_ref_bytes=cw_raw.get("max_ref_bytes", 1_000_000),
    )

    return FlenseConfig(server=server, compression=compression, providers=providers, code_writer=code_writer)
