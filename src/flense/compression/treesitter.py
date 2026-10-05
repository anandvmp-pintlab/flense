from __future__ import annotations

import logging

import tree_sitter

from .grammar import get_language
from .types import CodeBlock, CompressedBlock

logger = logging.getLogger(__name__)

# Node types to extract per language. These represent structural/signature nodes.
_EXTRACTABLE_NODES: dict[str, set[str]] = {
    "python": {
        "class_definition", "function_definition", "decorated_definition",
        "import_statement", "import_from_statement",
    },
    "javascript": {
        "class_declaration", "function_declaration", "method_definition",
        "import_statement", "export_statement", "lexical_declaration",
    },
    "typescript": {
        "class_declaration", "function_declaration", "method_definition",
        "import_statement", "export_statement", "interface_declaration",
        "type_alias_declaration", "lexical_declaration",
    },
    "tsx": {
        "class_declaration", "function_declaration", "method_definition",
        "import_statement", "export_statement", "interface_declaration",
        "type_alias_declaration", "lexical_declaration",
    },
    "go": {
        "function_declaration", "method_declaration", "type_declaration",
        "import_declaration",
    },
    "rust": {
        "function_item", "struct_item", "enum_item", "impl_item",
        "trait_item", "use_declaration", "mod_item",
    },
    "java": {
        "class_declaration", "method_declaration", "interface_declaration",
        "import_declaration", "constructor_declaration",
    },
    "c": {
        "function_definition", "struct_specifier", "enum_specifier",
        "declaration",
    },
    "cpp": {
        "function_definition", "class_specifier", "struct_specifier",
        "enum_specifier", "declaration", "namespace_definition",
    },
}

_DEFAULT_EXTRACTABLE = {
    "class_definition", "class_declaration",
    "function_definition", "function_declaration",
    "method_definition", "method_declaration",
    "import_statement",
}


def compress_with_treesitter(block: CodeBlock) -> CompressedBlock | None:
    """Compress a code block using tree-sitter AST extraction.

    Returns a CompressedBlock on success, or None if tree-sitter cannot
    handle this language or fails to parse.
    """
    if not block.language:
        return None

    language = get_language(block.language)
    if language is None:
        return None

    parser = tree_sitter.Parser(language)
    source = block.content.encode("utf-8")

    try:
        tree = parser.parse(source)
    except Exception as exc:
        logger.warning("Tree-sitter parse failed for %s: %s", block.language, exc)
        return None

    lines = block.content.split("\n")
    extractable = _EXTRACTABLE_NODES.get(block.language, _DEFAULT_EXTRACTABLE)
    extracted: list[str] = []

    _walk_and_extract(tree.root_node, lines, extractable, extracted, block.language)

    if not extracted:
        return None

    compressed = "\n".join(extracted)
    return CompressedBlock(
        original=block,
        compressed=compressed,
        original_token_estimate=0,
        compressed_token_estimate=0,
        method="treesitter",
    )


def _walk_and_extract(
    node: tree_sitter.Node,
    lines: list[str],
    extractable: set[str],
    output: list[str],
    language: str,
) -> None:
    """Recursively walk the AST and extract structural information."""
    if node.type in extractable:
        signature = _extract_signature(node, lines, language)
        if signature:
            line_num = node.start_point[0] + 1  # 1-indexed
            output.append(f"[Line {line_num}] {signature}")
        # Recurse into children for nested definitions
        for child in node.children:
            _walk_and_extract(child, lines, extractable, output, language)
    else:
        for child in node.children:
            _walk_and_extract(child, lines, extractable, output, language)


def _extract_signature(
    node: tree_sitter.Node,
    lines: list[str],
    language: str,
) -> str | None:
    """Extract the signature line(s) from a definition node."""
    if language == "python":
        return _extract_python_signature(node, lines)
    if language in ("go", "rust", "java", "c", "cpp"):
        return _extract_first_line(node, lines)
    if language in ("javascript", "typescript", "tsx"):
        return _extract_js_signature(node, lines)
    return _extract_first_line(node, lines)


def _extract_python_signature(
    node: tree_sitter.Node,
    lines: list[str],
) -> str | None:
    if node.type == "class_definition":
        start = node.start_point[0]
        if start < len(lines):
            return lines[start].strip().rstrip(":")
        return None

    if node.type == "decorated_definition":
        decorators: list[str] = []
        func_node = node
        for child in node.children:
            if child.type == "decorator":
                dec_line = child.start_point[0]
                if dec_line < len(lines):
                    decorators.append(lines[dec_line].strip())
            elif child.type == "function_definition":
                func_node = child

        sig = _python_func_sig(func_node, lines)
        if sig and decorators:
            return "\n".join(decorators) + "\n" + sig
        return sig

    if node.type == "function_definition":
        return _python_func_sig(node, lines)

    if node.type in ("import_statement", "import_from_statement"):
        start = node.start_point[0]
        if start < len(lines):
            return lines[start].strip()
        return None

    return None


def _python_func_sig(node: tree_sitter.Node, lines: list[str]) -> str | None:
    """Extract a Python function signature from def to the colon."""
    start = node.start_point[0]
    if start >= len(lines):
        return None
    # Signatures can span multiple lines
    for i in range(start, min(start + 10, len(lines))):
        if ":" in lines[i] and not lines[i].strip().startswith("#"):
            sig_lines = [lines[j].strip() for j in range(start, i + 1)]
            return " ".join(sig_lines).rstrip(":")
    return lines[start].strip().rstrip(":")


def _extract_js_signature(
    node: tree_sitter.Node,
    lines: list[str],
) -> str | None:
    """Extract a JS/TS declaration signature."""
    start = node.start_point[0]
    if start >= len(lines):
        return None

    line = lines[start].strip()
    # For declarations with a body, strip the opening brace
    if line.endswith("{"):
        line = line[:-1].strip()
    return line or None


def _extract_first_line(node: tree_sitter.Node, lines: list[str]) -> str | None:
    """Generic extraction: take the first line of the node."""
    start = node.start_point[0]
    if start >= len(lines):
        return None
    line = lines[start].strip()
    if line.endswith("{"):
        line = line[:-1].strip()
    return line or None
