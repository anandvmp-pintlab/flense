from flense.compression.language import detect_language


def test_fence_tag_python():
    assert detect_language("x = 1", fence_tag="python") == "python"
    assert detect_language("x = 1", fence_tag="py") == "python"


def test_fence_tag_javascript():
    assert detect_language("var x = 1", fence_tag="js") == "javascript"
    assert detect_language("var x = 1", fence_tag="javascript") == "javascript"


def test_fence_tag_typescript():
    assert detect_language("const x: number = 1", fence_tag="ts") == "typescript"
    assert detect_language("const x: number = 1", fence_tag="tsx") == "tsx"


def test_fence_tag_go():
    assert detect_language("x := 1", fence_tag="go") == "go"
    assert detect_language("x := 1", fence_tag="golang") == "go"


def test_fence_tag_rust():
    assert detect_language("let x = 1;", fence_tag="rust") == "rust"
    assert detect_language("let x = 1;", fence_tag="rs") == "rust"


def test_file_extension():
    assert detect_language("x = 1", file_path="main.py") == "python"
    assert detect_language("x = 1", file_path="app.js") == "javascript"
    assert detect_language("x = 1", file_path="lib.rs") == "rust"
    assert detect_language("x = 1", file_path="main.go") == "go"


def test_heuristic_python():
    code = "def hello():\n    print('hi')\n"
    assert detect_language(code) == "python"


def test_heuristic_go():
    code = "func main() {\n    fmt.Println(\"hello\")\n}\n"
    assert detect_language(code) == "go"


def test_heuristic_rust():
    code = "fn main() {\n    println!(\"hello\");\n}\n"
    assert detect_language(code) == "rust"


def test_undetectable():
    assert detect_language("just some plain text") is None


def test_fence_tag_takes_priority_over_heuristic():
    # Content looks like Python but fence says Go
    code = "def hello():\n    pass\n"
    assert detect_language(code, fence_tag="go") == "go"
