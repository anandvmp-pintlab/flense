from flense.compression.estimator import (
    estimate_messages_tokens,
    estimate_tokens,
)


def test_estimate_tokens_basic():
    count = estimate_tokens("Hello, world!")
    assert count > 0
    assert count < 10


def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0


def test_estimate_tokens_code():
    code = "def hello():\n    print('hello world')\n"
    count = estimate_tokens(code)
    assert count > 5


def test_estimate_messages_string_content():
    messages = [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thanks!"},
    ]
    count = estimate_messages_tokens(messages)
    # 2 messages * 4 overhead + actual tokens
    assert count > 8


def test_estimate_messages_list_content():
    """Anthropic-style list-of-blocks content."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Here is some code to review."},
            ],
        },
    ]
    count = estimate_messages_tokens(messages)
    assert count > 4


def test_estimate_messages_tool_result():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "123",
                    "content": "The file contains 100 lines of Python.",
                },
            ],
        },
    ]
    count = estimate_messages_tokens(messages)
    assert count > 4


def test_estimate_messages_empty():
    assert estimate_messages_tokens([]) == 0
