from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .config import FlenseConfig
from .providers import get_adapter, register_providers
from .proxy import proxy_request
from .telemetry import SessionStats


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.httpx_client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=10.0,
            read=300.0,   # AI responses can be slow
            write=10.0,
            pool=10.0,
        ),
        follow_redirects=False,
        http2=True,
    )
    yield
    await app.state.httpx_client.aclose()


def create_app(config: FlenseConfig) -> FastAPI:
    app = FastAPI(title="flense", lifespan=_lifespan)
    app.state.config = config
    app.state.session_stats = SessionStats()
    register_providers(config)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.api_route(
        "/{provider}/{path:path}",
        methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    )
    async def proxy_route(provider: str, path: str, request: Request):
        adapter = get_adapter(provider)
        if adapter is None:
            return JSONResponse(
                {"error": f"Unknown provider: {provider}"},
                status_code=404,
            )
        return await proxy_request(request, adapter)

    return app
