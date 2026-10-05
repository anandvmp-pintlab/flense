import pytest

from flense.compression.treesitter import compress_with_treesitter
from flense.compression.types import CodeBlock

# Tree-sitter grammars are installed on demand, so these tests may trigger
# a pip install on first run. Skip if the grammar cannot be loaded.

_PYTHON_CODE = '''\
import os
from pathlib import Path


class DataProcessor:
    """Processes data files."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self._cache = {}

    def process(self, path: str) -> list:
        """Process a single file."""
        data = self._read(path)
        return self._transform(data)

    def _read(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()

    def _transform(self, data: bytes) -> list:
        return list(data)


def standalone_func(x: int, y: int) -> int:
    return x + y
'''


@pytest.fixture
def python_block():
    return CodeBlock(content=_PYTHON_CODE, language="python")


def test_compress_python(python_block):
    result = compress_with_treesitter(python_block)
    if result is None:
        pytest.skip("tree-sitter-python grammar not available")
    assert result.method == "treesitter"
    assert "class DataProcessor" in result.compressed
    assert "def __init__" in result.compressed
    assert "def process" in result.compressed
    assert "def _read" in result.compressed
    assert "def _transform" in result.compressed
    assert "def standalone_func" in result.compressed
    assert "import os" in result.compressed


def test_line_numbers_present(python_block):
    result = compress_with_treesitter(python_block)
    if result is None:
        pytest.skip("tree-sitter-python grammar not available")
    assert "[Line " in result.compressed
    # class DataProcessor is on line 5
    assert "[Line 5]" in result.compressed


def test_bodies_stripped(python_block):
    result = compress_with_treesitter(python_block)
    if result is None:
        pytest.skip("tree-sitter-python grammar not available")
    # Function bodies should not appear
    assert "self._cache" not in result.compressed
    assert "return x + y" not in result.compressed


def test_returns_none_for_unknown_language():
    block = CodeBlock(content="some content", language="brainfuck")
    result = compress_with_treesitter(block)
    assert result is None


def test_returns_none_for_no_language():
    block = CodeBlock(content="some content", language=None)
    result = compress_with_treesitter(block)
    assert result is None
