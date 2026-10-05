from flense.compression.classifier import detect_signals, resolve_strategy
from flense.compression.types import SignalType, Strategy


def test_detect_code_fences():
    messages = [{"role": "user", "content": "Here is code:\n```python\ndef foo():\n    pass\n```\n"}]
    signals = detect_signals(messages)
    types = {s.type for s in signals}
    assert SignalType.CODE_FENCE in types


def test_detect_file_paths():
    messages = [{"role": "user", "content": "Look at src/main/app.py for the implementation."}]
    signals = detect_signals(messages)
    types = {s.type for s in signals}
    assert SignalType.FILE_PATH in types


def test_detect_tool_results_anthropic():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "abc", "content": "result"},
            ],
        },
    ]
    signals = detect_signals(messages)
    types = {s.type for s in signals}
    assert SignalType.TOOL_RESULT in types


def test_detect_tool_results_openai():
    messages = [{"role": "tool", "content": "result"}]
    signals = detect_signals(messages)
    types = {s.type for s in signals}
    assert SignalType.TOOL_RESULT in types


def test_detect_system_file_ref():
    messages = [{"role": "user", "content": "hello"}]
    signals = detect_signals(messages, system="repository: /home/user/project")
    types = {s.type for s in signals}
    assert SignalType.SYSTEM_FILE_REF in types


def test_detect_high_message_count():
    messages = [{"role": "user", "content": f"msg {i}"} for i in range(12)]
    signals = detect_signals(messages)
    types = {s.type for s in signals}
    assert SignalType.HIGH_MESSAGE_COUNT in types


def test_no_signals_for_simple_message():
    messages = [{"role": "user", "content": "What is the weather today?"}]
    signals = detect_signals(messages)
    assert len(signals) == 0


# --- resolve_strategy tests ---


def test_header_strategy_wins():
    result = resolve_strategy(
        header_strategy="passthrough",
        provider_strategy="ast",
        global_strategy="auto",
        signals=[],
    )
    assert result == Strategy.PASSTHROUGH


def test_provider_strategy_second():
    result = resolve_strategy(
        header_strategy=None,
        provider_strategy="ast",
        global_strategy="auto",
        signals=[],
    )
    assert result == Strategy.AST


def test_global_strategy_third():
    result = resolve_strategy(
        header_strategy=None,
        provider_strategy=None,
        global_strategy="ctags",
        signals=[],
    )
    assert result == Strategy.CTAGS


def test_auto_needs_two_signals():
    from flense.compression.types import Signal, SignalStrength

    one_signal = [Signal(type=SignalType.CODE_FENCE, strength=SignalStrength.STRONG)]
    result = resolve_strategy(None, None, "auto", one_signal)
    assert result == Strategy.PASSTHROUGH

    two_signals = [
        Signal(type=SignalType.CODE_FENCE, strength=SignalStrength.STRONG),
        Signal(type=SignalType.FILE_PATH, strength=SignalStrength.STRONG),
    ]
    result = resolve_strategy(None, None, "auto", two_signals)
    assert result == Strategy.AST


def test_unknown_strategy_falls_back_to_passthrough():
    result = resolve_strategy("banana", None, "auto", [])
    assert result == Strategy.PASSTHROUGH
