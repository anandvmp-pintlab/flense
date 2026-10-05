from __future__ import annotations

import copy
import json
import logging
import re
import time

from .classifier import detect_signals, resolve_strategy
from .ctags import compress_with_ctags
from .estimator import estimate_messages_tokens, estimate_tokens
from .language import detect_language
from .regex import compress_with_regex
from .treesitter import compress_with_treesitter
from .types import CodeBlock, CompressedBlock, CompressionResult, Strategy

logger = logging.getLogger(__name__)

_CODE_BLOCK_RE = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
_FILE_HINT_RE = re.compile(r"(?:File|Path|file|path):\s*(\S+)")


def compress_payload(
    body: bytes,
    provider: str,
    threshold: int = 5000,
    header_strategy: str | None = None,
    provider_strategy: str | None = None,
    global_strategy: str = "auto",
) -> CompressionResult:
    """Main entry point for the compression pipeline.

    Takes the raw request body (bytes) and returns a CompressionResult
    containing the (possibly compressed) body and metadata.
    """
    start_time = time.monotonic()

    try:
        payload = json.loads(body)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return _passthrough(body, start_time)

    if not isinstance(payload, dict):
        return _passthrough(body, start_time)

    messages = payload.get("messages", [])
    if not messages:
        return _passthrough(body, start_time)

    system = payload.get("system", "")
    if isinstance(system, list):
        system = " ".join(
            b.get("text", "") for b in system if isinstance(b, dict)
        )

    # Token estimation
    tokens_before = estimate_messages_tokens(messages, provider)
    if system:
        tokens_before += estimate_tokens(system, provider)

    if tokens_before < threshold:
        return _passthrough(body, start_time, tokens_before=tokens_before)

    # Classification
    signals = detect_signals(messages, system if system else None)
    strategy = resolve_strategy(
        header_strategy, provider_strategy, global_strategy, signals,
    )

    if strategy == Strategy.PASSTHROUGH:
        return _passthrough(body, start_time, tokens_before=tokens_before)

    # Extract code blocks
    code_blocks = _extract_code_blocks(messages)
    if not code_blocks:
        return _passthrough(body, start_time, tokens_before=tokens_before)

    # Compress each block through the fallback chain
    compressed_blocks: list[CompressedBlock] = []
    for block in code_blocks:
        result = _compress_block(block, strategy)
        if result is not None:
            compressed_blocks.append(result)

    if not compressed_blocks:
        return _passthrough(body, start_time, tokens_before=tokens_before)

    # Reconstruct the payload with compressed blocks
    compressed_payload = _reconstruct_payload(payload, compressed_blocks)
    compressed_body = json.dumps(compressed_payload).encode("utf-8")

    # Estimate tokens after compression
    compressed_messages = compressed_payload.get("messages", [])
    tokens_after = estimate_messages_tokens(compressed_messages, provider)
    if system:
        tokens_after += estimate_tokens(system, provider)

    elapsed_ms = (time.monotonic() - start_time) * 1000

    return CompressionResult(
        original_body=body,
        compressed_body=compressed_body,
        strategy_applied=strategy,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        blocks_compressed=len(compressed_blocks),
        blocks_total=len(code_blocks),
        compression_time_ms=elapsed_ms,
        was_compressed=True,
    )


def _compress_block(block: CodeBlock, strategy: Strategy) -> CompressedBlock | None:
    """Apply the fallback chain: tree-sitter -> ctags -> regex."""
    if strategy in (Strategy.AST, Strategy.AUTO):
        result = compress_with_treesitter(block)
        if result is not None:
            return result

    if strategy in (Strategy.CTAGS, Strategy.AST, Strategy.AUTO):
        result = compress_with_ctags(block)
        if result is not None:
            return result

    # Regex as last resort for any non-passthrough strategy
    result = compress_with_regex(block)
    if result is not None:
        return result

    return None


def _extract_code_blocks(messages: list[dict]) -> list[CodeBlock]:
    """Extract fenced code blocks from all messages."""
    blocks: list[CodeBlock] = []
    for msg in messages:
        text_parts = _get_text_parts(msg)
        for text in text_parts:
            for match in _CODE_BLOCK_RE.finditer(text):
                fence_tag = match.group(1) or None
                content = match.group(2)

                # Look for a file path hint in the ~200 chars before the fence
                pre_text = text[max(0, match.start() - 200):match.start()]
                path_match = _FILE_HINT_RE.search(pre_text)
                source_path = path_match.group(1) if path_match else None

                lang = detect_language(
                    content, fence_tag=fence_tag, file_path=source_path,
                )
                blocks.append(CodeBlock(
                    content=content,
                    language=lang,
                    fence_tag=fence_tag,
                    source_path=source_path,
                ))
    return blocks


def _get_text_parts(msg: dict) -> list[str]:
    """Extract all text content from a message."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return [content]
    parts: list[str] = []
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get("text"):
                    parts.append(block["text"])
                inner = block.get("content", "")
                if isinstance(inner, str) and inner:
                    parts.append(inner)
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and sub.get("text"):
                            parts.append(sub["text"])
    return parts


def _reconstruct_payload(
    payload: dict,
    compressed_blocks: list[CompressedBlock],
) -> dict:
    """Replace original code blocks in the payload with compressed versions."""
    replacements: dict[str, str] = {}
    for cb in compressed_blocks:
        tag = cb.original.fence_tag or ""
        original_fenced = f"```{tag}\n{cb.original.content}```"
        compressed_fenced = (
            f"```{tag}\n"
            f"# [flense: compressed with {cb.method}]\n"
            f"{cb.compressed}\n```"
        )
        replacements[original_fenced] = compressed_fenced

    new_payload = copy.deepcopy(payload)
    for msg in new_payload.get("messages", []):
        _apply_replacements(msg, replacements)

    return new_payload


def _apply_replacements(msg: dict, replacements: dict[str, str]) -> None:
    """Apply text replacements to a message's content in place."""
    content = msg.get("content")
    if isinstance(content, str):
        for original, compressed in replacements.items():
            content = content.replace(original, compressed)
        msg["content"] = content
    elif isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if "text" in block and isinstance(block["text"], str):
                    for original, compressed in replacements.items():
                        block["text"] = block["text"].replace(original, compressed)
                inner = block.get("content")
                if isinstance(inner, str):
                    for original, compressed in replacements.items():
                        inner = inner.replace(original, compressed)
                    block["content"] = inner
                elif isinstance(inner, list):
                    for sub in inner:
                        if isinstance(sub, dict) and "text" in sub:
                            for original, compressed in replacements.items():
                                sub["text"] = sub["text"].replace(original, compressed)


def _passthrough(
    body: bytes,
    start_time: float,
    tokens_before: int = 0,
) -> CompressionResult:
    return CompressionResult(
        original_body=body,
        compressed_body=body,
        strategy_applied=Strategy.PASSTHROUGH,
        tokens_before=tokens_before,
        tokens_after=tokens_before,
        blocks_compressed=0,
        blocks_total=0,
        compression_time_ms=(time.monotonic() - start_time) * 1000,
        was_compressed=False,
    )
