import json

from flense.compression import compress_payload
from flense.compression.types import Strategy


def _make_body(messages, system=None):
    payload = {"messages": messages, "model": "claude-sonnet-4-20250514"}
    if system:
        payload["system"] = system
    return json.dumps(payload).encode()


def test_passthrough_small_payload():
    body = _make_body([{"role": "user", "content": "Hello!"}])
    result = compress_payload(body, provider="anthropic", threshold=5000)
    assert not result.was_compressed
    assert result.strategy_applied == Strategy.PASSTHROUGH
    assert result.compressed_body == body


def test_passthrough_non_json():
    body = b"not json at all"
    result = compress_payload(body, provider="anthropic")
    assert not result.was_compressed
    assert result.compressed_body == body


def test_passthrough_no_messages():
    body = json.dumps({"model": "test"}).encode()
    result = compress_payload(body, provider="anthropic")
    assert not result.was_compressed


def test_passthrough_few_signals():
    # Large payload but only one signal (code fence, no file paths)
    big_code = "x = 1\n" * 1000
    content = f"Here is code:\n```python\n{big_code}```\n"
    body = _make_body([{"role": "user", "content": content}])
    result = compress_payload(body, provider="anthropic", threshold=100)
    # Only one signal (code fence) so auto mode should passthrough
    assert result.strategy_applied == Strategy.PASSTHROUGH


def test_header_forces_passthrough():
    big_code = "x = 1\n" * 1000
    content = f"Look at src/app.py:\n```python\n{big_code}```\n"
    body = _make_body([{"role": "user", "content": content}])
    result = compress_payload(
        body, provider="anthropic", threshold=100,
        header_strategy="passthrough",
    )
    assert result.strategy_applied == Strategy.PASSTHROUGH


def test_compression_with_forced_strategy():
    code = (
        "import os\n\n"
        "class Processor:\n"
        "    def process(self, data):\n"
        "        result = []\n"
        "        for item in data:\n"
        "            result.append(item * 2)\n"
        "        return result\n\n"
        "    def validate(self, item):\n"
        "        return item is not None\n\n"
        "def helper(x):\n"
        "    return x + 1\n"
    )
    # Pad with enough repetitions to exceed threshold
    padded_code = code * 5
    content = f"File: src/processor.py\n```python\n{padded_code}```\n"
    body = _make_body([{"role": "user", "content": content}])

    # Force AST strategy via header to bypass 2-signal requirement
    result = compress_payload(
        body, provider="anthropic", threshold=100,
        header_strategy="ast",
    )
    # Should attempt compression (may fall through to regex if tree-sitter unavailable)
    if result.was_compressed:
        assert result.tokens_after < result.tokens_before
        assert result.blocks_compressed > 0
        assert result.compression_time_ms >= 0
        assert "flense: compressed" in result.compressed_body.decode()


def test_two_signals_triggers_auto():
    code = "def foo():\n    pass\n" * 50
    # Two signals: code fence + file path
    content = f"Look at src/utils/helpers.py:\n```python\n{code}```\n"
    body = _make_body([{"role": "user", "content": content}])
    result = compress_payload(body, provider="anthropic", threshold=100)
    # Auto mode should detect 2 signals and attempt compression
    assert result.strategy_applied != Strategy.PASSTHROUGH


def test_config_values():
    """Verify config values are respected."""
    body = _make_body([{"role": "user", "content": "Hello!"}])
    # Very high threshold means everything passes through
    result = compress_payload(body, provider="anthropic", threshold=999999)
    assert not result.was_compressed


def test_no_shrink_sends_original_body():
    """If 'compression' would not reduce tokens, the original body is sent."""
    # Many bare imports: extraction prefixes each line with "[Line N] " plus a
    # marker, so the "compressed" form is larger than the original.
    code = "import os\n" * 300
    content = f"File: src/imports.py\n```python\n{code}```\n"
    body = _make_body([{"role": "user", "content": content}])

    result = compress_payload(
        body, provider="anthropic", threshold=100, header_strategy="ast",
    )

    # Strategy was attempted (AST), but it did not help, so no compression and
    # the body is returned unchanged.
    if result.strategy_applied != Strategy.PASSTHROUGH:
        assert result.was_compressed is False
        assert result.compressed_body == body
        assert result.tokens_after >= result.tokens_before - 1
