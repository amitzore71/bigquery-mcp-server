"""FastAPI application factory with lifespan bootstrap."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from bqsaas import __version__
from bqsaas.api.routes import auth_routes, chat, health, kinds, query, tenants
from bqsaas.api import web as web_ui
from bqsaas.config import Settings, get_settings
from bqsaas.errors import BqSaaSError
from bqsaas.kinds.registry import KindRegistry, register_builtin_kinds
from bqsaas.services.bootstrap import bootstrap_demo
from bqsaas.storage.memory import MemoryStore

logger = logging.getLogger(__name__)


def create_app(
    settings: Optional[Settings] = None,
    store: Optional[MemoryStore] = None,
    *,
    bootstrap: bool = True,
    enable_web_ui: bool = True,
) -> FastAPI:
    """
    Application factory.

    Parameters
    ----------
    settings:
        Optional settings override (tests).
    store:
        Optional pre-built store (tests).
    bootstrap:
        When True (default), seed a demo tenant + API key on startup.
    enable_web_ui:
        Mount the HTMX chat UI at ``/`` when templates are available.
    """
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app_store: MemoryStore = store if store is not None else MemoryStore()
        registry = KindRegistry()
        register_builtin_kinds(registry)

        app.state.settings = settings
        app.state.store = app_store
        app.state.kinds = registry
        app.state.demo = None

        if bootstrap and len(app_store) == 0:
            try:
                app.state.demo = bootstrap_demo(app_store)
                raw = app.state.demo.get("raw_api_key") or ""
                logger.info("Demo tenant ready — API key prefix: %s…", raw[:16])
                # Always print once for local developer onboarding
                print(
                    "\n" + "=" * 60
                    + "\n  BigQuery SaaS demo tenant bootstrapped"
                    + f"\n  Tenant : {app.state.demo.get('tenant_id')}"
                    + f"\n  API key: {raw}"
                    + "\n  Use: Authorization: Bearer <api_key>"
                    + "\n  Docs: http://localhost:8000/docs"
                    + "\n" + "=" * 60 + "\n"
                )
            except Exception:
                logger.exception("Demo bootstrap failed")
                app_store.mark_ready()
        else:
            app_store.mark_ready()

        yield

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Multi-tenant BigQuery SaaS API — query, chat, workspaces, "
            "kind-based resources, and MCP-compatible tools."
        ),
        lifespan=lifespan,
        openapi_tags=[
            {"name": "health", "description": "Liveness / readiness"},
            {"name": "auth", "description": "API keys and identity"},
            {"name": "tenants", "description": "Tenants, workspaces, connections"},
            {"name": "chat", "description": "AI chat sessions and messages"},
            {"name": "kinds", "description": "Resource kind registry"},
            {"name": "query", "description": "Direct SQL execution"},
            {"name": "web", "description": "Browser chat UI (HTMX)"},
        ],
    )

    # CORS — allow all in development
    if getattr(settings, "app_env", "development") != "production":
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(BqSaaSError)
    async def bqsaas_error_handler(request: Request, exc: BqSaaSError) -> JSONResponse:
        return JSONResponse(status_code=exc.http_status, content=exc.to_dict())

    app.include_router(health.router)
    app.include_router(auth_routes.router)
    app.include_router(tenants.router)
    app.include_router(chat.router)
    app.include_router(kinds.router)
    app.include_router(query.router)

    if enable_web_ui:
        web_ui.mount_static(app)
        app.include_router(web_ui.router)

    return app


# Module-level app for ``uvicorn bqsaas.api.app:app --app-dir src``
app = create_app()
