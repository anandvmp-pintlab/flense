from __future__ import annotations

import re

from .types import Signal, SignalStrength, SignalType, Strategy

_CODE_FENCE_RE = re.compile(r"```\w*\n", re.MULTILINE)

# Possessive quantifier (?:...)++ prevents catastrophic backtracking on
# pathological input (long runs of word characters) while matching the same
# file-path shapes. Requires Python 3.11+.
_FILE_PATH_RE = re.compile(
    r"(?:^|[\s\"'`])([/\\]?(?:[\w.\-]+[/\\])++[\w.\-]+\.\w{1,10})(?:[\s\"'`:\n]|$)",
    re.MULTILINE,
)

_SYSTEM_FILE_REF_RE = re.compile(
    r"(?:file|path|repo|repository|codebase|source)\s*[:=]",
    re.IGNORECASE,
)

_HIGH_MESSAGE_THRESHOLD = 10

# Upper bound on the amount of text scanned for signals. Signal detection does
# not need the entire payload, and bounding the input keeps regex work linear
# regardless of request size.
_MAX_SCAN_CHARS = 262_144


def detect_signals(
    messages: list[dict],
    system: str | None = None,
) -> list[Signal]:
    """Scan the messages array and optional system prompt for compression signals."""
    signals: list[Signal] = []
    full_text = _extract_full_text(messages)
    if len(full_text) > _MAX_SCAN_CHARS:
        full_text = full_text[:_MAX_SCAN_CHARS]

    fence_matches = _CODE_FENCE_RE.findall(full_text)
    if fence_matches:
        signals.append(Signal(
            type=SignalType.CODE_FENCE,
            strength=SignalStrength.STRONG,
            detail=f"{len(fence_matches)} code fence(s)",
        ))

    path_matches = _FILE_PATH_RE.findall(full_text)
    if path_matches:
        signals.append(Signal(
            type=SignalType.FILE_PATH,
            strength=SignalStrength.STRONG,
            detail=f"{len(path_matches)} path(s)",
        ))

    if _has_tool_results(messages):
        signals.append(Signal(
            type=SignalType.TOOL_RESULT,
            strength=SignalStrength.STRONG,
        ))

    if system and _SYSTEM_FILE_REF_RE.search(system):
        signals.append(Signal(
            type=SignalType.SYSTEM_FILE_REF,
            strength=SignalStrength.MODERATE,
        ))

    if len(messages) >= _HIGH_MESSAGE_THRESHOLD:
        signals.append(Signal(
            type=SignalType.HIGH_MESSAGE_COUNT,
            strength=SignalStrength.MODERATE,
            detail=f"{len(messages)} messages",
        ))

    return signals


def resolve_strategy(
    header_strategy: str | None,
    provider_strategy: str | None,
    global_strategy: str,
    signals: list[Signal],
) -> Strategy:
    """Apply the strategy precedence chain.

    Precedence: X-Flense-Strategy header > per-provider config > global config > auto.
    Under auto: require 2+ signals to compress, else passthrough.
    """
    if header_strategy is not None:
        return _parse_strategy(header_strategy)

    if provider_strategy is not None:
        return _parse_strategy(provider_strategy)

    resolved = _parse_strategy(global_strategy)
    if resolved != Strategy.AUTO:
        return resolved

    # Auto mode: need 2+ signals
    if len(signals) >= 2:
        return Strategy.AST
    return Strategy.PASSTHROUGH


def _parse_strategy(value: str) -> Strategy:
    try:
        return Strategy(value.lower())
    except ValueError:
        return Strategy.PASSTHROUGH


def _extract_full_text(messages: list[dict]) -> str:
    """Flatten all message content into a single string for signal scanning."""
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text", "")
                    if text:
                        parts.append(text)
                    inner = block.get("content", "")
                    if isinstance(inner, str) and inner:
                        parts.append(inner)
                    elif isinstance(inner, list):
                        for sub in inner:
                            if isinstance(sub, dict) and sub.get("text"):
                                parts.append(sub["text"])
    return "\n".join(parts)


def _has_tool_results(messages: list[dict]) -> bool:
    """Check whether any message contains tool_result blocks."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
        # OpenAI format
        if msg.get("role") == "tool":
            return True
    return False
