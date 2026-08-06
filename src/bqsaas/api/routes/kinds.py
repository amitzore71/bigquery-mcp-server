"""Kind registry endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from bqsaas.kinds.registry import KindRegistry, register_builtin_kinds

router = APIRouter(prefix="/v1", tags=["kinds"])


def _registry(request: Request) -> KindRegistry:
    reg = getattr(request.app.state, "kinds", None)
    if reg is None:
        reg = KindRegistry()
        register_builtin_kinds(reg)
    return reg


@router.get("/kinds")
async def list_kinds(request: Request) -> list[dict]:
    """List registered resource kinds."""
    return _registry(request).list_kinds()


@router.get("/kinds/{kind}")
async def get_kind(kind: str, request: Request) -> dict:
    reg = _registry(request)
    for meta in reg.list_kinds():
        if meta.get("kind") == kind:
            return meta
    raise HTTPException(status_code=404, detail=f"Unknown kind: {kind}")
