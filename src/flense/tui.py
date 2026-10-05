"""Two-panel Textual TUI dashboard for flense.

Panel 1: Live request feed (per request)
Panel 2: Session aggregates (running totals)
"""

from __future__ import annotations

import threading
from datetime import datetime

from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.widgets import DataTable, Footer, Header, Static

from .telemetry import RequestLog, SessionStats

# Maximum rows kept in the live feed table before pruning old entries.
_MAX_FEED_ROWS = 200


class StatsPanel(Static):
    """Right panel showing session aggregate stats."""

    def __init__(self, stats: SessionStats, **kwargs) -> None:
        super().__init__(**kwargs)
        self._stats = stats

    def render_stats(self) -> str:
        snap = self._stats.snapshot()
        lines = [
            "[bold]Session totals[/bold]",
            "",
            f"  Requests:     {snap['total_requests']}",
            f"  Tokens saved: {snap['total_tokens_saved']:,}",
            f"  Cost saved:   ${snap['total_cost_saved_usd']:.4f}",
            "",
            "[bold]By provider[/bold]",
        ]
        for name, bucket in snap["by_provider"].items():
            lines.append(
                f"  {name}: {bucket['requests']} req, "
                f"{bucket['tokens_saved']:,} tok, "
                f"${bucket['cost_saved_usd']:.4f}"
            )
        if not snap["by_provider"]:
            lines.append("  (none yet)")

        lines.append("")
        lines.append("[bold]By strategy[/bold]")
        for strat, count in snap["by_strategy"].items():
            lines.append(f"  {strat}: {count}")
        if not snap["by_strategy"]:
            lines.append("  (none yet)")

        return "\n".join(lines)

    def on_mount(self) -> None:
        self.update(self.render_stats())
        self.set_interval(1.0, self._refresh)

    def _refresh(self) -> None:
        self.update(self.render_stats())


class FlenseDashboard(App):
    """Main TUI application."""

    TITLE = "flense"
    CSS = """
    Horizontal {
        height: 1fr;
    }
    #feed {
        width: 3fr;
    }
    #stats {
        width: 1fr;
        padding: 1 2;
        border-left: solid $accent;
    }
    DataTable {
        height: 1fr;
    }
    """
    BINDINGS = [("q", "quit", "Quit")]

    def __init__(
        self,
        session_stats: SessionStats,
        server_host: str = "127.0.0.1",
        server_port: int = 8000,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._stats = session_stats
        self._server_host = server_host
        self._server_port = server_port
        self._pending: list[RequestLog] = []
        self._lock = threading.Lock()

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield DataTable(id="feed")
            yield StatsPanel(self._stats, id="stats")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#feed", DataTable)
        table.add_columns(
            "Time", "Provider", "Model", "Before", "After",
            "Saved", "Strategy", "Time(ms)", "Status",
        )
        self.sub_title = f"listening on {self._server_host}:{self._server_port}"
        self._stats.add_listener(self._on_request)
        self.set_interval(0.25, self._drain_pending)

    def _on_request(self, entry: RequestLog) -> None:
        """Called from the ASGI thread; just queue for the TUI thread."""
        with self._lock:
            self._pending.append(entry)

    def _drain_pending(self) -> None:
        """Called on the TUI event loop; moves queued entries into the table."""
        with self._lock:
            batch = list(self._pending)
            self._pending.clear()

        if not batch:
            return

        table = self.query_one("#feed", DataTable)
        for entry in batch:
            ts = datetime.fromtimestamp(entry.timestamp).strftime("%H:%M:%S")
            model = entry.model or "-"
            if len(model) > 28:
                model = model[:25] + "..."
            table.add_row(
                ts,
                entry.provider,
                model,
                f"{entry.tokens_before:,}",
                f"{entry.tokens_after:,}",
                f"{entry.tokens_saved:,}",
                entry.strategy.value,
                f"{entry.compression_time_ms:.1f}",
                entry.status,
            )

        # Prune old rows
        while table.row_count > _MAX_FEED_ROWS:
            first_key = next(iter(table.rows))
            table.remove_row(first_key)

        table.scroll_end(animate=False)
