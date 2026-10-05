from flense.compression.regex import compress_with_regex
from flense.compression.types import CodeBlock


def test_python_definitions():
    code = (
        "import os\n"
        "\n"
        "class MyClass:\n"
        "    def __init__(self):\n"
        "        self.x = 1\n"
        "\n"
        "    def method(self):\n"
        "        return self.x\n"
        "\n"
        "def standalone():\n"
        "    pass\n"
    )
    block = CodeBlock(content=code, language="python")
    result = compress_with_regex(block)
    assert result is not None
    assert "import os" in result.compressed
    assert "class MyClass" in result.compressed
    assert "def __init__" in result.compressed
    assert "def method" in result.compressed
    assert "def standalone" in result.compressed
    assert result.method == "regex"


def test_go_definitions():
    code = (
        "func main() {\n"
        "    fmt.Println(\"hello\")\n"
        "}\n"
        "\n"
        "type Server struct {\n"
        "    Port int\n"
        "}\n"
    )
    block = CodeBlock(content=code, language="go")
    result = compress_with_regex(block)
    assert result is not None
    assert "func main()" in result.compressed
    assert "type Server struct" in result.compressed


def test_plain_text_returns_none():
    block = CodeBlock(content="Just some plain text without any code.", language=None)
    result = compress_with_regex(block)
    assert result is None


def test_line_numbers_present():
    code = "def hello():\n    pass\n\ndef world():\n    pass\n"
    block = CodeBlock(content=code, language="python")
    result = compress_with_regex(block)
    assert result is not None
    assert "[Line 1]" in result.compressed
    # Both functions should have line numbers
    lines = result.compressed.strip().split("\n")
    assert len(lines) == 2
    assert all(line.startswith("[Line ") for line in lines)
