from __future__ import annotations

import json
import logging
import time
from functools import partial

import anyio
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from .compression import compress_payload
from .compression.estimator import estimate_messages_tokens, estimate_tokens
from .compression.types import Strategy
from .providers.base import ProviderAdapter
from .telemetry import (
    RequestLog,
    build_request_log,
    build_telemetry_headers,
    emit_headless_log,
)

logger = logging.getLogger(__name__)


def _extract_model(body: bytes) -> str | None:
    """Best-effort extraction of the model name from a JSON request body."""
    try:
        payload = json.loads(body)
        if isinstance(payload, dict):
            model = payload.get("model")
            if isinstance(model, str):
                return model
    except Exception:
        pass
    return None


def _estimate_body_tokens(body: bytes, provider: str) -> int:
    """Estimate total input tokens for a request body."""
    try:
        payload = json.loads(body)
        messages = payload.get("messages", [])
        total = estimate_messages_tokens(messages, provider)
        system = payload.get("system")
        if system:
            text = system if isinstance(system, str) else json.dumps(system)
            total += estimate_tokens(text, provider)
        return total
    except Exception:
        return 0


async def proxy_request(
    request: Request,
    adapter: ProviderAdapter,
) -> StreamingResponse | JSONResponse:
    """Forward a request to the upstream provider, compressing if appropriate."""
    import hmac

    import httpx

    config = request.app.state.config

    # --- Authentication ---
    auth_token = getattr(config.server, "auth_token", None)
    if auth_token:
        provided = request.headers.get("x-flense-auth", "")
        if not hmac.compare_digest(provided, auth_token):
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Missing or invalid X-Flense-Auth token.",
                        "type": "flense_auth_error",
                    }
                },
            )

    # --- Request size cap ---
    max_body = getattr(config.server, "max_body_bytes", 0)
    content_length = request.headers.get("content-length")
    if max_body and content_length and content_length.isdigit():
        if int(content_length) > max_body:
            return JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "message": (
                            f"Request body exceeds the {max_body}-byte limit."
                        ),
                        "type": "flense_payload_too_large",
                    }
                },
            )

    upstream_url = adapter.rewrite_url(request.url.path)
    if request.url.query:
        upstream_url += f"?{request.url.query}"

    headers = adapter.filter_request_headers(dict(request.headers))
    # Flense control headers must never be forwarded upstream.
    headers.pop("x-flense-auth", None)

    body = await request.body()
    if max_body and len(body) > max_body:
        return JSONResponse(
            status_code=413,
            content={
                "error": {
                    "message": f"Request body exceeds the {max_body}-byte limit.",
                    "type": "flense_payload_too_large",
                }
            },
        )

    model = _extract_model(body)

    header_strategy = headers.pop("x-flense-strategy", None)

    # --- Code-writer bypass ---
    if header_strategy == Strategy.CODE_WRITER.value:
        if not getattr(config.code_writer, "enabled", False):
            return JSONResponse(
                status_code=403,
                content={
                    "error": {
                        "message": (
                            "The code-writer strategy is disabled. Enable it by "
                            "setting 'enabled = true' in the [code_writer] section "
                            "of your flense config."
                        ),
                        "type": "flense_code_writer_disabled",
                    }
                },
            )

        from .code_writer import CodeWriterError, handle_code_writer

        ref_file = headers.pop("x-flense-ref-file", None)
        output_file = headers.pop("x-flense-output-file", None)
        tokens_before = _estimate_body_tokens(body, adapter.name)

        try:
            _, output_path = await handle_code_writer(
                body=body,
                headers=headers,
                ref_file_header=ref_file,
                output_file_header=output_file,
                cw_config=config.code_writer,
                config=config,
                httpx_client=request.app.state.httpx_client,
            )
            ack_message = f"Generated and written to {output_path}"
            status = "ok"
        except (CodeWriterError, FileNotFoundError) as exc:
            # Validation errors carry client-safe messages.
            logger.info("Code-writer request rejected: %s", exc)
            return JSONResponse(
                status_code=400,
                content={"error": {"message": str(exc), "type": "flense_code_writer_error"}},
            )
        except Exception:
            # Unexpected failures: log the detail, return a generic message.
            logger.error("Code-writer bypass failed", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "The code-writer request failed. See server logs for details.",
                        "type": "flense_code_writer_error",
                    }
                },
            )

        ack = adapter.make_ack_response(ack_message, model)

        entry = RequestLog(
            timestamp=time.time(),
            provider=adapter.name,
            model=model,
            strategy=Strategy.CODE_WRITER,
            tokens_before=tokens_before,
            tokens_after=0,
            tokens_saved=tokens_before,
            est_cost_saved=None,
            compression_time_ms=0.0,
            status=status,
        )
        session_stats = getattr(request.app.state, "session_stats", None)
        if session_stats is not None:
            session_stats.record(entry)
        if getattr(config.server, "headless", False):
            emit_headless_log(entry)

        telemetry_headers = build_telemetry_headers(entry)
        telemetry_headers["X-Flense-Output-File"] = output_path
        return JSONResponse(content=ack, headers=telemetry_headers)
    # --- End code-writer bypass ---

    # --- Compression pipeline ---
    # Compression is CPU/IO-bound and synchronous (tiktoken, tree-sitter, and a
    # ctags subprocess), so run it in a worker thread to avoid blocking the
    # event loop and stalling other concurrent requests.
    provider_comp = adapter.compression_config
    compression_result = await anyio.to_thread.run_sync(
        partial(
            compress_payload,
            body=body,
            provider=adapter.name,
            threshold=(
                provider_comp.threshold
                if provider_comp
                else config.compression.threshold
            ),
            header_strategy=header_strategy,
            provider_strategy=(provider_comp.strategy if provider_comp else None),
            global_strategy=config.compression.strategy,
        )
    )
    body = compression_result.compressed_body

    if compression_result.was_compressed:
        logger.info(
            "Compressed %s request: %d -> %d tokens (%.1f ms, %s)",
            adapter.name,
            compression_result.tokens_before,
            compression_result.tokens_after,
            compression_result.compression_time_ms,
            compression_result.strategy_applied.value,
        )
    # --- End compression pipeline ---

    # --- Telemetry ---
    provider_config = config.providers.get(adapter.name)
    pricing_overrides = provider_config.pricing if provider_config else None

    entry = build_request_log(
        compression_result,
        provider=adapter.name,
        model=model,
        pricing_overrides=pricing_overrides or None,
    )

    session_stats = getattr(request.app.state, "session_stats", None)
    if session_stats is not None:
        session_stats.record(entry)

    if getattr(config.server, "headless", False):
        emit_headless_log(entry)

    telemetry_headers = build_telemetry_headers(entry)
    # --- End telemetry ---

    client: httpx.AsyncClient = request.app.state.httpx_client

    upstream_req = client.build_request(
        method=request.method,
        url=upstream_url,
        headers=headers,
        content=body,
    )

    try:
        upstream_resp = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as exc:
        logger.error("Upstream request to %s failed: %s", adapter.name, exc)
        error_body = adapter.make_error_response(
            f"flense could not reach the {adapter.name} upstream: {exc}"
        )
        return JSONResponse(
            status_code=502,
            content=error_body,
            headers=telemetry_headers,
        )

    response_headers = adapter.filter_response_headers(
        dict(upstream_resp.headers)
    )
    response_headers.update(telemetry_headers)

    async def stream_body():
        try:
            async for chunk in upstream_resp.aiter_raw():
                yield chunk
        finally:
            await upstream_resp.aclose()

    return StreamingResponse(
        content=stream_body(),
        status_code=upstream_resp.status_code,
        headers=response_headers,
    )
