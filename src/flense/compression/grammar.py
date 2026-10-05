from __future__ import annotations

import importlib
import logging

import tree_sitter

logger = logging.getLogger(__name__)

_GRAMMAR_PACKAGES: dict[str, str] = {
    "python": "tree-sitter-python",
    "javascript": "tree-sitter-javascript",
    "typescript": "tree-sitter-typescript",
    "tsx": "tree-sitter-typescript",
    "java": "tree-sitter-java",
    "c": "tree-sitter-c",
    "cpp": "tree-sitter-cpp",
    "c_sharp": "tree-sitter-c-sharp",
    "go": "tree-sitter-go",
    "rust": "tree-sitter-rust",
    "ruby": "tree-sitter-ruby",
    "php": "tree-sitter-php",
    "swift": "tree-sitter-swift",
    "kotlin": "tree-sitter-kotlin",
    "bash": "tree-sitter-bash",
    "html": "tree-sitter-html",
    "css": "tree-sitter-css",
    "json": "tree-sitter-json",
    "yaml": "tree-sitter-yaml",
    "toml": "tree-sitter-toml",
    "sql": "tree-sitter-sql",
    "scala": "tree-sitter-scala",
    "lua": "tree-sitter-lua",
    "elixir": "tree-sitter-elixir",
    "haskell": "tree-sitter-haskell",
    "ocaml": "tree-sitter-ocaml",
    "zig": "tree-sitter-zig",
    "markdown": "tree-sitter-markdown",
}

# Grammar packages use underscores in import paths.
_IMPORT_PATHS: dict[str, str] = {
    k: v.replace("-", "_") for k, v in _GRAMMAR_PACKAGES.items()
}

_languages: dict[str, tree_sitter.Language] = {}


def get_language(lang: str) -> tree_sitter.Language | None:
    """Get a tree-sitter Language, installing the grammar if needed.

    Returns None if the language is unsupported or installation fails.
    """
    if lang in _languages:
        return _languages[lang]

    if lang not in _GRAMMAR_PACKAGES:
        return None

    module_name = _IMPORT_PATHS[lang]
    try:
        mod = importlib.import_module(module_name)
    except ImportError:
        # Grammars are optional dependencies and are never installed at request
        # time (that would block the event loop and let request content drive
        # package installation). If missing, fall back to ctags/regex upstream.
        logger.info(
            "Tree-sitter grammar for %r not installed (%s). Install the grammar "
            "extras with 'pip install flense[grammars]' to enable AST compression "
            "for this language; falling back to ctags/regex.",
            lang,
            _GRAMMAR_PACKAGES[lang],
        )
        return None

    try:
        if lang == "tsx":
            language = tree_sitter.Language(mod.language_tsx())
        elif lang == "typescript":
            language = tree_sitter.Language(mod.language_typescript())
        else:
            language = tree_sitter.Language(mod.language())
    except (AttributeError, TypeError) as exc:
        logger.warning("Failed to load language %s: %s", lang, exc)
        return None

    _languages[lang] = language
    return language


def is_language_available(lang: str) -> bool:
    return lang in _GRAMMAR_PACKAGES
