"""Entry point for the daemon process. Invoked by daemon.py via subprocess."""
from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="flense proxy server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2912)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    # CLI flags override config values.
    config.server.host = args.host
    config.server.port = args.port
    config.server.headless = True  # daemon has no terminal

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
