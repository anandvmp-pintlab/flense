from __future__ import annotations

import re

from .types import CodeBlock, CompressedBlock

_DEFINITION_PATTERNS: list[re.Pattern] = [
    # Python
    re.compile(r"^\s*(def\s+\w+\s*\(.*?\).*?:)", re.MULTILINE),
    re.compile(r"^\s*(class\s+\w+.*?:)", re.MULTILINE),
    re.compile(r"^\s*((?:from\s+\S+\s+)?import\s+.+)$", re.MULTILINE),
    # JS/TS
    re.compile(r"^\s*((?:export\s+)?(?:default\s+)?function\s+\w+\s*\(.*?\).*?)\s*\{?\s*$", re.MULTILINE),
    re.compile(r"^\s*((?:export\s+)?class\s+\w+.*?)\s*\{?\s*$", re.MULTILINE),
    re.compile(r"^\s*((?:export\s+)?(?:const|let|var)\s+\w+\s*=\s*(?:async\s+)?\(.*?\)\s*=>)", re.MULTILINE),
    # Go
    re.compile(r"^\s*(func\s+(?:\(\w+\s+\*?\w+\)\s+)?\w+\(.*?\).*?)\s*\{?\s*$", re.MULTILINE),
    re.compile(r"^\s*(type\s+\w+\s+(?:struct|interface))\s*\{?\s*$", re.MULTILINE),
    # Rust
    re.compile(r"^\s*((?:pub(?:\(crate\))?\s+)?fn\s+\w+.*?)\s*\{?\s*$", re.MULTILINE),
    re.compile(r"^\s*((?:pub\s+)?(?:struct|enum|trait|impl)\s+\w+.*?)\s*\{?\s*$", re.MULTILINE),
    # Java / C#
    re.compile(r"^\s*((?:public|private|protected)\s+(?:static\s+)?(?:class|interface)\s+\w+.*?)\s*\{?\s*$", re.MULTILINE),
    re.compile(r"^\s*((?:public|private|protected)\s+(?:static\s+)?(?:[\w<>\[\]]+\s+)+\w+\s*\(.*?\).*?)\s*\{?\s*$", re.MULTILINE),
]


def compress_with_regex(block: CodeBlock) -> CompressedBlock | None:
    """Compress a code block using regex-based heuristic extraction.

    Last resort. Extracts lines that look like definitions/signatures.
    """
    extracted: list[tuple[int, str]] = []
    matched_lines: set[int] = set()

    for pattern in _DEFINITION_PATTERNS:
        for match in pattern.finditer(block.content):
            start_offset = match.start()
            line_num = block.content[:start_offset].count("\n") + 1
            if line_num not in matched_lines:
                matched_lines.add(line_num)
                sig = match.group(1).strip().rstrip("{").rstrip(":").strip()
                extracted.append((line_num, sig))

    if not extracted:
        return None

    extracted.sort(key=lambda x: x[0])
    output_lines = [f"[Line {ln}] {sig}" for ln, sig in extracted]
    compressed = "\n".join(output_lines)

    return CompressedBlock(
        original=block,
        compressed=compressed,
        original_token_estimate=0,
        compressed_token_estimate=0,
        method="regex",
    )
