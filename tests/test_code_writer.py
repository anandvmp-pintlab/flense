import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from flense.code_writer import (
    _build_prompt,
    _extract_spec,
    _get_api_key,
    _infer_provider,
    _resolve_output_path,
    handle_code_writer,
)
from flense.compression.types import Strategy
from flense.config import CodeWriterConfig, FlenseConfig
from flense.providers.anthropic import AnthropicAdapter
from flense.providers.openai import OpenAIAdapter


# ---------------------------------------------------------------------------
# _infer_provider
# ---------------------------------------------------------------------------

class TestInferProvider:
    def test_claude_model(self):
        assert _infer_provider("claude-haiku-4-5") == "anthropic"
        assert _infer_provider("claude-sonnet-4-20250514") == "anthropic"

    def test_openai_model(self):
        assert _infer_provider("gpt-4o-mini") == "openai"
        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("o1-mini") == "openai"


# ---------------------------------------------------------------------------
# _get_api_key
# ---------------------------------------------------------------------------

class TestGetApiKey:
    def test_anthropic_from_header(self):
        headers = {"x-api-key": "sk-ant-test"}
        assert _get_api_key("anthropic", headers) == "sk-ant-test"

    def test_openai_from_bearer_header(self):
        headers = {"authorization": "Bearer sk-openai-test"}
        assert _get_api_key("openai", headers) == "sk-openai-test"

    def test_openai_missing_header_uses_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env-key")
        assert _get_api_key("openai", {}) == "sk-env-key"

    def test_anthropic_missing_header_uses_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-env-ant")
        assert _get_api_key("anthropic", {}) == "sk-env-ant"

    def test_unknown_provider_returns_none(self):
        assert _get_api_key("gemini", {"x-api-key": "key"}) is None

    def test_no_key_available_returns_none(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _get_api_key("anthropic", {}) is None


# ---------------------------------------------------------------------------
# _extract_spec
# ---------------------------------------------------------------------------

class TestExtractSpec:
    def _body(self, messages):
        return json.dumps({"messages": messages, "model": "claude-haiku-4-5"}).encode()

    def test_string_content(self):
        body = self._body([{"role": "user", "content": "Write me a parser"}])
        assert _extract_spec(body) == "Write me a parser"

    def test_list_content(self):
        body = self._body([
            {"role": "user", "content": [
                {"type": "text", "text": "Generate tests"},
                {"type": "text", "text": " for this module"},
            ]}
        ])
        assert _extract_spec(body) == "Generate tests\n for this module"

    def test_last_user_message_used(self):
        body = self._body([
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "OK"},
            {"role": "user", "content": "Second message"},
        ])
        assert _extract_spec(body) == "Second message"

    def test_no_user_message_returns_empty(self):
        body = self._body([{"role": "assistant", "content": "Hello"}])
        assert _extract_spec(body) == ""

    def test_non_json_returns_empty(self):
        assert _extract_spec(b"not json") == ""


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_spec_and_ref(self):
        prompt = _build_prompt("Generate a test suite", "src/app.py", "def foo(): pass")
        assert "Generate a test suite" in prompt
        assert "src/app.py" in prompt
        assert "def foo(): pass" in prompt

    def test_instructs_raw_code_only(self):
        prompt = _build_prompt("Do something", "ref.py", "# code")
        assert "raw code" in prompt.lower()
        assert "no markdown" in prompt.lower()


# ---------------------------------------------------------------------------
# _resolve_output_path
# ---------------------------------------------------------------------------

class TestResolveOutputPath:
    def test_explicit_output_file_relative(self):
        path = _resolve_output_path("out/result.py", None, "./generated")
        assert path == (Path("./generated").resolve() / "out" / "result.py")

    def test_explicit_output_file_absolute_within_dir(self, tmp_path):
        out_dir = tmp_path / "generated"
        absolute = str(out_dir / "result.py")
        path = _resolve_output_path(absolute, None, str(out_dir))
        assert path == Path(absolute).resolve()

    def test_explicit_output_file_absolute_outside_dir_rejected(self, tmp_path):
        out_dir = tmp_path / "generated"
        absolute = str(tmp_path / "escape.py")  # sibling of out_dir, outside it
        with pytest.raises(ValueError, match="outside"):
            _resolve_output_path(absolute, None, str(out_dir))

    def test_traversal_output_file_rejected(self):
        with pytest.raises(ValueError, match="outside"):
            _resolve_output_path("../../etc/passwd", None, "./generated")

    def test_derived_from_ref_file(self):
        path = _resolve_output_path(None, "src/parser.py", "./generated")
        assert path.parent == Path("./generated").resolve()
        assert path.name.startswith("parser_generated_")
        assert path.suffix == ".py"

    def test_derived_uses_basename_only(self):
        # Even an absolute/traversing ref path only contributes its stem.
        path = _resolve_output_path(None, "/etc/passwd", "./generated")
        assert path.parent == Path("./generated").resolve()
        assert path.name.startswith("passwd_generated_")

    def test_no_ref_file(self):
        path = _resolve_output_path(None, None, "./generated")
        assert path.parent == Path("./generated").resolve()
        assert path.name.startswith("generated_")

    def test_ref_file_with_no_extension(self):
        path = _resolve_output_path(None, "Makefile", "./generated")
        assert path.suffix == ".txt"


# ---------------------------------------------------------------------------
# make_ack_response
# ---------------------------------------------------------------------------

class TestAnthropicAckResponse:
    def setup_method(self):
        self.adapter = AnthropicAdapter("anthropic", "https://api.anthropic.com")

    def test_shape(self):
        ack = self.adapter.make_ack_response("Written to ./out.py", "claude-haiku-4-5")
        assert ack["type"] == "message"
        assert ack["role"] == "assistant"
        assert ack["content"][0]["type"] == "text"
        assert "Written to ./out.py" in ack["content"][0]["text"]
        assert ack["model"] == "claude-haiku-4-5"
        assert ack["stop_reason"] == "end_turn"
        assert ack["id"].startswith("msg_flense_")

    def test_none_model(self):
        ack = self.adapter.make_ack_response("msg", None)
        assert ack["model"] == "unknown"


class TestOpenAIAckResponse:
    def setup_method(self):
        self.adapter = OpenAIAdapter("openai", "https://api.openai.com")

    def test_shape(self):
        ack = self.adapter.make_ack_response("Written to ./out.py", "gpt-4o-mini")
        assert ack["object"] == "chat.completion"
        assert ack["choices"][0]["message"]["role"] == "assistant"
        assert "Written to ./out.py" in ack["choices"][0]["message"]["content"]
        assert ack["model"] == "gpt-4o-mini"
        assert ack["choices"][0]["finish_reason"] == "stop"
        assert ack["id"].startswith("chatcmpl-flense-")

    def test_none_model(self):
        ack = self.adapter.make_ack_response("msg", None)
        assert ack["model"] == "unknown"


# ---------------------------------------------------------------------------
# handle_code_writer integration tests
# ---------------------------------------------------------------------------

def _make_body(spec: str) -> bytes:
    return json.dumps({
        "model": "claude-haiku-4-5",
        "messages": [{"role": "user", "content": spec}],
    }).encode()


def _make_anthropic_mock_client(generated_code: str) -> MagicMock:
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json = MagicMock(return_value={
        "content": [{"text": generated_code}]
    })
    client = MagicMock()
    client.post = AsyncMock(return_value=mock_resp)
    return client


class TestHandleCodeWriter:
    @pytest.fixture
    def config(self, tmp_path):
        cfg = FlenseConfig()
        cfg.code_writer = CodeWriterConfig(
            enabled=True,
            model="claude-haiku-4-5",
            fallback="gpt-4o-mini",
            output_dir=str(tmp_path / "generated"),
            allowed_ref_dir=str(tmp_path),
        )
        return cfg

    @pytest.fixture
    def ref_file(self, tmp_path):
        ref = tmp_path / "reference.py"
        ref.write_text("def existing_func(x):\n    return x * 2\n")
        return ref

    async def test_missing_ref_file_header_raises(self, config):
        with pytest.raises(ValueError, match="X-Flense-Ref-File"):
            await handle_code_writer(
                body=_make_body("Write tests"),
                headers={"x-api-key": "sk-test"},
                ref_file_header=None,
                output_file_header=None,
                cw_config=config.code_writer,
                config=config,
                httpx_client=MagicMock(),
            )

    async def test_ref_file_not_found_raises(self, config, tmp_path):
        # Within the allowed dir but does not exist.
        missing = str(tmp_path / "missing.py")
        with pytest.raises(FileNotFoundError, match="not found"):
            await handle_code_writer(
                body=_make_body("Write tests"),
                headers={"x-api-key": "sk-test"},
                ref_file_header=missing,
                output_file_header=None,
                cw_config=config.code_writer,
                config=config,
                httpx_client=MagicMock(),
            )

    async def test_ref_file_outside_allowed_dir_rejected(self, config):
        # Attempt to read a file outside allowed_ref_dir (path traversal / abs).
        with pytest.raises(ValueError, match="outside"):
            await handle_code_writer(
                body=_make_body("Write tests"),
                headers={"x-api-key": "sk-test"},
                ref_file_header="/etc/passwd",
                output_file_header=None,
                cw_config=config.code_writer,
                config=config,
                httpx_client=MagicMock(),
            )

    async def test_ref_file_too_large_rejected(self, config, tmp_path):
        big = tmp_path / "big.py"
        big.write_text("x = 1\n" * 100)
        config.code_writer.max_ref_bytes = 10  # force the limit
        with pytest.raises(ValueError, match="too large"):
            await handle_code_writer(
                body=_make_body("Write tests"),
                headers={"x-api-key": "sk-test"},
                ref_file_header=str(big),
                output_file_header=None,
                cw_config=config.code_writer,
                config=config,
                httpx_client=MagicMock(),
            )

    async def test_empty_spec_raises(self, config, ref_file):
        body = json.dumps({"model": "claude-haiku-4-5", "messages": []}).encode()
        with pytest.raises(ValueError, match="spec"):
            await handle_code_writer(
                body=body,
                headers={"x-api-key": "sk-test"},
                ref_file_header=str(ref_file),
                output_file_header=None,
                cw_config=config.code_writer,
                config=config,
                httpx_client=MagicMock(),
            )

    async def test_successful_generation(self, config, ref_file, tmp_path):
        generated = "def new_func(x):\n    return x + 1\n"
        mock_client = _make_anthropic_mock_client(generated)

        code, output_path = await handle_code_writer(
            body=_make_body("Write a function that adds 1"),
            headers={"x-api-key": "sk-ant-test"},
            ref_file_header=str(ref_file),
            output_file_header=None,
            cw_config=config.code_writer,
            config=config,
            httpx_client=mock_client,
        )

        assert code == generated
        assert Path(output_path).exists()
        assert Path(output_path).read_text() == generated

    async def test_explicit_output_file(self, config, ref_file, tmp_path):
        generated = "# output"
        mock_client = _make_anthropic_mock_client(generated)
        # Output must live within the configured output_dir.
        out = str(tmp_path / "generated" / "myfile.py")

        _, output_path = await handle_code_writer(
            body=_make_body("Generate something"),
            headers={"x-api-key": "sk-ant-test"},
            ref_file_header=str(ref_file),
            output_file_header=out,
            cw_config=config.code_writer,
            config=config,
            httpx_client=mock_client,
        )

        assert Path(output_path) == Path(out).resolve()
        assert Path(output_path).read_text() == generated

    async def test_output_file_outside_dir_rejected(self, config, ref_file, tmp_path):
        mock_client = _make_anthropic_mock_client("# output")
        out = str(tmp_path / "escape.py")  # outside output_dir
        with pytest.raises(ValueError, match="outside"):
            await handle_code_writer(
                body=_make_body("Generate something"),
                headers={"x-api-key": "sk-ant-test"},
                ref_file_header=str(ref_file),
                output_file_header=out,
                cw_config=config.code_writer,
                config=config,
                httpx_client=mock_client,
            )

    async def test_all_models_fail_raises(self, config, ref_file):
        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=Exception("API error"))

        with pytest.raises(RuntimeError, match="All code-writer models failed"):
            await handle_code_writer(
                body=_make_body("Write something"),
                headers={"x-api-key": "sk-ant-test"},
                ref_file_header=str(ref_file),
                output_file_header=None,
                cw_config=config.code_writer,
                config=config,
                httpx_client=mock_client,
            )

    async def test_falls_back_to_second_model(self, config, ref_file):
        generated = "# fallback output"

        fail_resp = MagicMock()
        fail_resp.raise_for_status = MagicMock(side_effect=Exception("primary failed"))

        ok_resp = MagicMock()
        ok_resp.raise_for_status = MagicMock()
        ok_resp.json = MagicMock(return_value={"choices": [{"message": {"content": generated}}]})

        mock_client = MagicMock()
        mock_client.post = AsyncMock(side_effect=[fail_resp, ok_resp])

        # primary = claude-haiku-4-5 (anthropic), fallback = gpt-4o-mini (openai)
        headers = {
            "x-api-key": "sk-ant-test",
            "authorization": "Bearer sk-openai-test",
        }

        code, _ = await handle_code_writer(
            body=_make_body("Write something"),
            headers=headers,
            ref_file_header=str(ref_file),
            output_file_header=None,
            cw_config=config.code_writer,
            config=config,
            httpx_client=mock_client,
        )

        assert code == generated


# ---------------------------------------------------------------------------
# CodeWriterConfig parsing
# ---------------------------------------------------------------------------

class TestCodeWriterConfig:
    def test_defaults(self):
        cfg = FlenseConfig()
        assert cfg.code_writer.enabled is False  # opt-in by default
        assert cfg.code_writer.model == "claude-haiku-4-5"
        assert cfg.code_writer.fallback == "gpt-4o-mini"
        assert cfg.code_writer.output_dir == "./generated"
        assert cfg.code_writer.allowed_ref_dir == "."

    def test_code_writer_strategy_in_enum(self):
        assert Strategy.CODE_WRITER.value == "code-writer"
