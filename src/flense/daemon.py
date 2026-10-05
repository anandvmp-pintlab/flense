from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

FLENSE_DIR = Path.home() / ".flense"
PID_FILE = FLENSE_DIR / "flense.pid"
LOG_FILE = FLENSE_DIR / "flense.log"


def start_daemon(
    host: str,
    port: int,
    config_path: Path | None = None,
) -> int:
    """Spawn flense as a background process. Returns the PID."""
    FLENSE_DIR.mkdir(parents=True, exist_ok=True)

    if is_running():
        raise RuntimeError(f"flense is already running (PID {read_pid()})")

    cmd = [
        sys.executable,
        "-m",
        "flense.server",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if config_path is not None:
        cmd.extend(["--config", str(config_path)])

    log_fh = open(LOG_FILE, "a")  # noqa: SIM115
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=log_fh,
        start_new_session=True,
    )
    PID_FILE.write_text(str(proc.pid))
    return proc.pid


def stop_daemon() -> None:
    """Stop a running flense daemon via SIGTERM."""
    pid = read_pid()
    if pid is None:
        raise RuntimeError("flense is not running (no PID file)")

    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass  # already dead

    PID_FILE.unlink(missing_ok=True)


def is_running() -> bool:
    """Check whether the daemon process is alive."""
    pid = read_pid()
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        PID_FILE.unlink(missing_ok=True)
        return False


def read_pid() -> int | None:
    """Read the PID from the PID file, or None if absent/invalid."""
    if not PID_FILE.exists():
        return None
    try:
        return int(PID_FILE.read_text().strip())
    except (ValueError, OSError):
        return None
