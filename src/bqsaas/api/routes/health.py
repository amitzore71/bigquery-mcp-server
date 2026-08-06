"""Health and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Request

from bqsaas import __version__

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict:
    return {"status": "ok", "version": __version__}


@router.get("/ready")
async def ready(request: Request) -> dict:
    store = getattr(request.app.state, "store", None)
    if store is None or not getattr(store, "is_ready", False):
        return {"status": "not_ready", "store": False}
    return {"status": "ready", "store": True, "version": __version__}
