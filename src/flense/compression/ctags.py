from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile

from .types import CodeBlock, CompressedBlock

logger = logging.getLogger(__name__)

_LANG_TO_EXT: dict[str, str] = {
    "python": ".py", "javascript": ".js", "typescript": ".ts",
    "java": ".java", "c": ".c", "cpp": ".cpp", "go": ".go",
    "rust": ".rs", "ruby": ".rb", "php": ".php", "swift": ".swift",
    "kotlin": ".kt", "scala": ".scala", "bash": ".sh",
    "lua": ".lua", "r": ".r", "dart": ".dart", "elixir": ".ex",
    "haskell": ".hs", "ocaml": ".ml", "zig": ".zig", "c_sharp": ".cs",
}


def ctags_available() -> bool:
    """Check if universal-ctags is installed and accessible."""
    return shutil.which("ctags") is not None


def compress_with_ctags(block: CodeBlock) -> CompressedBlock | None:
    """Compress a code block using Universal Ctags.

    Returns a CompressedBlock on success, None if ctags is unavailable
    or fails to produce output.
    """
    if not ctags_available():
        return None

    ext = _LANG_TO_EXT.get(block.language or "", ".txt")

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=ext, delete=True,
        ) as tmp:
            tmp.write(block.content)
            tmp.flush()

            result = subprocess.run(
                [
                    "ctags",
                    "--output-format=json",
                    "--fields=+n+S+K",
                    "-f", "-",
                    tmp.name,
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )

        if result.returncode != 0:
            logger.debug("ctags returned %d: %s", result.returncode, result.stderr)
            return None

        output_lines = _parse_ctags_output(result.stdout)
        if not output_lines:
            return None

        compressed = "\n".join(output_lines)
        return CompressedBlock(
            original=block,
            compressed=compressed,
            original_token_estimate=0,
            compressed_token_estimate=0,
            method="ctags",
        )

    except (subprocess.TimeoutExpired, OSError) as exc:
        logger.warning("ctags execution failed: %s", exc)
        return None


def _parse_ctags_output(raw: str) -> list[str]:
    """Parse JSON-format ctags output into [Line N] symbol lines."""
    results: list[tuple[int, str]] = []
    for line in raw.strip().splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue

        name = entry.get("name", "")
        line_num = entry.get("line", 0)
        kind = entry.get("kind", "")
        signature = entry.get("signature", "")

        if name and line_num:
            label = f"{kind} {name}{signature}" if signature else f"{kind} {name}"
            results.append((line_num, label.strip()))

    results.sort(key=lambda x: x[0])
    return [f"[Line {ln}] {label}" for ln, label in results]
