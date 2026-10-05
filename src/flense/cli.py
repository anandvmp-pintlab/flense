from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="flense",
    help="Lightweight reverse proxy that compresses AI API payloads.",
    add_completion=False,
)


@app.command()
def start(
    port: Optional[int] = typer.Option(None, help="Override server port"),
    foreground: bool = typer.Option(
        False, "--fg", help="Run in foreground (no daemon)"
    ),
    headless: Optional[bool] = typer.Option(
        None, help="Headless mode (JSON logs, no TUI)"
    ),
    config: Optional[Path] = typer.Option(
        None, help="Path to flense.toml"
    ),
) -> None:
    """Start the flense proxy."""
    from .config import load_config

    cfg = load_config(config)
    effective_port = port if port is not None else cfg.server.port

    # CLI flag overrides config file
    use_headless = headless if headless is not None else cfg.server.headless

    if foreground:
        if use_headless:
            _run_headless(cfg, effective_port)
        else:
            _run_tui(cfg, effective_port)
    else:
        from .daemon import is_running, start_daemon

        if is_running():
            from .daemon import read_pid

            typer.echo(f"flense is already running (PID {read_pid()})")
            raise typer.Exit(1)

        pid = start_daemon(cfg.server.host, effective_port, config)
        typer.echo(f"flense started on {cfg.server.host}:{effective_port} (PID {pid})")


def _run_headless(cfg, port: int) -> None:
    """Run the proxy in foreground with structured JSON logs (no TUI)."""
    import uvicorn

    from .app import create_app

    cfg.server.headless = True
    instance = create_app(cfg)
    uvicorn.run(instance, host=cfg.server.host, port=port)


def _run_tui(cfg, port: int) -> None:
    """Run the proxy in foreground with the Textual TUI dashboard."""
    import threading

    import uvicorn

    from .app import create_app
    from .tui import FlenseDashboard

    instance = create_app(cfg)
    stats = instance.state.session_stats

    # Run uvicorn in a background thread so Textual owns the main thread.
    server = uvicorn.Server(
        uvicorn.Config(
            instance,
            host=cfg.server.host,
            port=port,
            log_level="warning",
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    dashboard = FlenseDashboard(
        session_stats=stats,
        server_host=cfg.server.host,
        server_port=port,
    )
    dashboard.run()

    # TUI exited, shut down the server.
    server.should_exit = True
    thread.join(timeout=5)


@app.command()
def stop() -> None:
    """Stop the flense proxy."""
    from .daemon import stop_daemon

    try:
        stop_daemon()
    except RuntimeError as exc:
        typer.echo(str(exc))
        raise typer.Exit(1)

    typer.echo("flense stopped.")


@app.command()
def status() -> None:
    """Check whether flense is running."""
    from .daemon import is_running, read_pid

    if is_running():
        typer.echo(f"flense is running (PID {read_pid()})")
    else:
        typer.echo("flense is not running.")
