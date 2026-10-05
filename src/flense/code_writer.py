"""Code-writer bypass: route code generation tasks to a cheap model.

Triggered by X-Flense-Strategy: code-writer. Sends the request to a cheap
cloud model, writes the output to disk, and returns a synthetic ACK to the
client. The frontier model is never called.
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import httpx

from .config import CodeWriterConfig, FlenseConfig

logger = logging.getLogger(__name__)

_ANTHROPIC_UPSTREAM = "https://api.anthropic.com"
_OPENAI_UPSTREAM = "https://api.openai.com"


def _infer_provider(model: str) -> str:
    """Infer provider from model name prefix."""
    if model.startswith("claude"):
        return "anthropic"
    return "openai"


def _get_api_key(provider: str, headers: dict[str, str]) -> str | None:
    """Extract API key for the given provider from request headers, then env vars."""
    if provider == "anthropic":
        return headers.get("x-api-key") or os.environ.get("ANTHROPIC_API_KEY")
    if provider == "openai":
        auth = headers.get("authorization", "")
        key = auth.removeprefix("Bearer ").strip() or None
        return key or os.environ.get("OPENAI_API_KEY")
    return None


def _extract_spec(body: bytes) -> str:
    """Extract the spec from the last user message in the request body."""
    try:
        payload = json.loads(body)
        messages = payload.get("messages", [])
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    parts = [
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    ]
                    return "\n".join(parts)
    except Exception:
        pass
    return ""


def _build_prompt(spec: str, ref_file_path: str, ref_content: str) -> str:
    return (
        f"You are a code generator. Write code following the exact style and "
        f"conventions shown in the reference file.\n\n"
        f"Reference file ({ref_file_path}):\n"
        f"```\n{ref_content}\n```\n\n"
        f"Task: {spec}\n\n"
        f"Output only the raw code. No markdown formatting, no explanations, no preamble."
    )


def _resolve_output_path(
    output_file: str | None,
    ref_file_path: str | None,
    output_dir: str,
) -> Path:
    if output_file:
        p = Path(output_file)
        if not p.is_absolute():
            p = Path(output_dir) / p
        return p

    if ref_file_path:
        ref = Path(ref_file_path)
        suffix = ref.suffix or ".txt"
        timestamp = int(time.time())
        return Path(output_dir) / f"{ref.stem}_generated_{timestamp}{suffix}"

    timestamp = int(time.time())
    return Path(output_dir) / f"generated_{timestamp}.txt"


async def _call_anthropic(
    model: str,
    prompt: str,
    api_key: str,
    client: httpx.AsyncClient,
    upstream: str,
) -> str:
    resp = await client.post(
        f"{upstream}/v1/messages",
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        content=json.dumps({
            "model": model,
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["content"][0]["text"]


async def _call_openai(
    model: str,
    prompt: str,
    api_key: str,
    client: httpx.AsyncClient,
    upstream: str,
) -> str:
    resp = await client.post(
        f"{upstream}/v1/chat/completions",
        headers={
            "authorization": f"Bearer {api_key}",
            "content-type": "application/json",
        },
        content=json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }).encode(),
        timeout=120.0,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


async def _call_model(
    model: str,
    prompt: str,
    headers: dict[str, str],
    client: httpx.AsyncClient,
    config: FlenseConfig,
) -> str:
    provider = _infer_provider(model)
    api_key = _get_api_key(provider, headers)
    if not api_key:
        raise ValueError(
            f"No API key found for provider '{provider}' (model '{model}'). "
            f"Pass the key via request headers or set the appropriate env var."
        )

    provider_config = config.providers.get(provider)
    if provider == "anthropic":
        upstream = (
            provider_config.upstream
            if provider_config and provider_config.upstream
            else _ANTHROPIC_UPSTREAM
        )
        return await _call_anthropic(model, prompt, api_key, client, upstream)

    upstream = (
        provider_config.upstream
        if provider_config and provider_config.upstream
        else _OPENAI_UPSTREAM
    )
    return await _call_openai(model, prompt, api_key, client, upstream)


async def handle_code_writer(
    body: bytes,
    headers: dict[str, str],
    ref_file_header: str | None,
    output_file_header: str | None,
    cw_config: CodeWriterConfig,
    config: FlenseConfig,
    httpx_client: httpx.AsyncClient,
) -> tuple[str, str]:
    """Execute the code-writer bypass.

    Returns (generated_code, output_path_str). Raises on error.
    """
    if not ref_file_header:
        raise ValueError(
            "X-Flense-Ref-File header is required for the code-writer strategy"
        )

    ref_path = Path(ref_file_header)
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference file not found: {ref_file_header}")
    ref_content = ref_path.read_text(errors="replace")

    spec = _extract_spec(body)
    if not spec:
        raise ValueError("Could not extract a spec from the request messages")

    prompt = _build_prompt(spec, ref_file_header, ref_content)

    generated_code: str | None = None
    last_error: Exception | None = None
    for model in [cw_config.model, cw_config.fallback]:
        if not model:
            continue
        try:
            generated_code = await _call_model(model, prompt, headers, httpx_client, config)
            logger.info("Code-writer used model %s", model)
            break
        except Exception as exc:
            logger.warning("Code-writer model %s failed: %s", model, exc)
            last_error = exc

    if generated_code is None:
        raise RuntimeError(f"All code-writer models failed. Last error: {last_error}")

    output_path = _resolve_output_path(output_file_header, ref_file_header, cw_config.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(generated_code)
    logger.info("Code-writer wrote output to %s", output_path)

    return generated_code, str(output_path)
