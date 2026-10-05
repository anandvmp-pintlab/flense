from __future__ import annotations

import re

# Canonical language names match tree-sitter grammar package suffixes.
_TAG_MAP: dict[str, str] = {
    "python": "python", "py": "python",
    "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript",
    "tsx": "tsx",
    "jsx": "javascript",
    "java": "java",
    "c": "c", "h": "c",
    "cpp": "cpp", "c++": "cpp", "cc": "cpp", "cxx": "cpp", "hpp": "cpp",
    "csharp": "c_sharp", "cs": "c_sharp", "c#": "c_sharp",
    "go": "go", "golang": "go",
    "rust": "rust", "rs": "rust",
    "ruby": "ruby", "rb": "ruby",
    "php": "php",
    "swift": "swift",
    "kotlin": "kotlin", "kt": "kotlin",
    "scala": "scala",
    "bash": "bash", "sh": "bash", "shell": "bash", "zsh": "bash",
    "sql": "sql",
    "html": "html", "htm": "html",
    "css": "css",
    "json": "json",
    "yaml": "yaml", "yml": "yaml",
    "toml": "toml",
    "xml": "xml",
    "markdown": "markdown", "md": "markdown",
    "lua": "lua",
    "r": "r",
    "dart": "dart",
    "elixir": "elixir", "ex": "elixir",
    "haskell": "haskell", "hs": "haskell",
    "ocaml": "ocaml", "ml": "ocaml",
    "zig": "zig",
}

_EXT_MAP: dict[str, str] = {
    ".py": "python", ".pyi": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "typescript", ".tsx": "tsx", ".jsx": "javascript",
    ".java": "java",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp",
    ".cs": "c_sharp",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".kt": "kotlin", ".kts": "kotlin",
    ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".sql": "sql",
    ".html": "html", ".htm": "html",
    ".css": "css",
    ".json": "json",
    ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml",
    ".xml": "xml",
    ".md": "markdown",
    ".lua": "lua",
    ".r": "r",
    ".dart": "dart",
    ".ex": "elixir", ".exs": "elixir",
    ".hs": "haskell",
    ".ml": "ocaml",
    ".zig": "zig",
}

_HEURISTICS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^\s*def\s+\w+.*:\s*$", re.MULTILINE), "python"),
    (re.compile(r"^\s*from\s+\w+\s+import\s+", re.MULTILINE), "python"),
    (re.compile(r"^\s*func\s+\w+\(", re.MULTILINE), "go"),
    (re.compile(r"^\s*package\s+\w+\s*$", re.MULTILINE), "go"),
    (re.compile(r"^\s*fn\s+\w+\(", re.MULTILINE), "rust"),
    (re.compile(r"^\s*pub\s+(?:fn|struct|enum|mod)\s+", re.MULTILINE), "rust"),
    (re.compile(r"^\s*function\s+\w+\s*\(", re.MULTILINE), "javascript"),
    (re.compile(r"^\s*interface\s+\w+\s*\{", re.MULTILINE), "typescript"),
    (re.compile(r":\s*(?:string|number|boolean)\b", re.MULTILINE), "typescript"),
    (re.compile(r"^\s*public\s+class\s+\w+", re.MULTILINE), "java"),
]


def detect_language(
    content: str,
    fence_tag: str | None = None,
    file_path: str | None = None,
) -> str | None:
    """Detect the programming language of a code block.

    Priority: fence_tag > file extension > heuristic patterns.
    Returns a canonical language name or None.
    """
    if fence_tag:
        tag = fence_tag.strip().lower()
        if tag in _TAG_MAP:
            return _TAG_MAP[tag]

    if file_path:
        for ext, lang in _EXT_MAP.items():
            if file_path.lower().endswith(ext):
                return lang

    for pattern, lang in _HEURISTICS:
        if pattern.search(content):
            return lang

    return None
