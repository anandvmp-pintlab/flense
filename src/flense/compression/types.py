from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Strategy(Enum):
    AUTO = "auto"
    AST = "ast"
    CTAGS = "ctags"
    PASSTHROUGH = "passthrough"
    CODE_WRITER = "code-writer"


class SignalType(Enum):
    CODE_FENCE = "code_fence"
    FILE_PATH = "file_path"
    TOOL_RESULT = "tool_result"
    SYSTEM_FILE_REF = "system_file_ref"
    HIGH_MESSAGE_COUNT = "high_message_count"


class SignalStrength(Enum):
    STRONG = "strong"
    MODERATE = "moderate"


@dataclass
class Signal:
    type: SignalType
    strength: SignalStrength
    detail: str = ""


@dataclass
class CodeBlock:
    """A block of code extracted from a message payload."""
    content: str
    language: str | None = None
    fence_tag: str | None = None
    source_path: str | None = None


@dataclass
class CompressedBlock:
    """Result of compressing a single CodeBlock."""
    original: CodeBlock
    compressed: str
    original_token_estimate: int
    compressed_token_estimate: int
    method: str  # "treesitter", "ctags", "regex"


@dataclass
class CompressionResult:
    """Overall result of the compression pipeline for a full request."""
    original_body: bytes
    compressed_body: bytes
    strategy_applied: Strategy
    tokens_before: int
    tokens_after: int
    blocks_compressed: int
    blocks_total: int
    compression_time_ms: float
    was_compressed: bool
