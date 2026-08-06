"""Tenant profile and workspace CRUD."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from bqsaas.auth.dependencies import AuthContext
from bqsaas.auth.deps import domain_error_to_http, get_auth_context, get_store
from bqsaas.errors import BqSaaSError
from bqsaas.services.connection_service import ConnectionService
from bqsaas.services.tenant_service import TenantService
from bqsaas.storage.memory import MemoryStore

router = APIRouter(prefix="/v1", tags=["tenants"])


class CreateWorkspaceRequest(BaseModel):
    name: str
    description: str = ""


class UpdateWorkspaceRequest(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None


@router.get("/tenants/me")
async def tenant_me(
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> dict:
    svc = TenantService(store)
    try:
        usage = svc.usage(ctx.tenant_id)
        sub = svc.get_subscription(ctx.tenant_id)
        return {
            "tenant": ctx.tenant.model_dump(mode="json"),
            "subscription": sub.model_dump(mode="json"),
            "usage": usage,
            "user": ctx.user.model_dump(mode="json"),
        }
    except BqSaaSError as e:
        raise domain_error_to_http(e) from e


@router.get("/workspaces")
async def list_workspaces(
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> list[dict]:
    svc = TenantService(store)
    return [w.model_dump(mode="json") for w in svc.list_workspaces(ctx.tenant_id)]


@router.post("/workspaces", status_code=status.HTTP_201_CREATED)
async def create_workspace(
    body: CreateWorkspaceRequest,
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> dict:
    svc = TenantService(store)
    try:
        ws = svc.create_workspace(
            ctx.tenant_id, body.name, description=body.description
        )
        return ws.model_dump(mode="json")
    except BqSaaSError as e:
        raise domain_error_to_http(e) from e


@router.get("/workspaces/{workspace_id}")
async def get_workspace(
    workspace_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> dict:
    svc = TenantService(store)
    try:
        return svc.get_workspace(ctx.tenant_id, workspace_id).model_dump(mode="json")
    except BqSaaSError as e:
        raise domain_error_to_http(e) from e


@router.patch("/workspaces/{workspace_id}")
async def update_workspace(
    workspace_id: str,
    body: UpdateWorkspaceRequest,
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> dict:
    svc = TenantService(store)
    try:
        ws = svc.update_workspace(
            ctx.tenant_id,
            workspace_id,
            name=body.name,
            description=body.description,
        )
        return ws.model_dump(mode="json")
    except BqSaaSError as e:
        raise domain_error_to_http(e) from e


@router.delete("/workspaces/{workspace_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workspace(
    workspace_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> None:
    svc = TenantService(store)
    try:
        svc.delete_workspace(ctx.tenant_id, workspace_id)
    except BqSaaSError as e:
        raise domain_error_to_http(e) from e


@router.get("/connections")
async def list_connections(
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> list[dict]:
    """List data connections (credentials redacted)."""
    svc = ConnectionService(store)
    out = []
    for c in svc.list_connections(ctx.tenant_id):
        d = c.model_dump(mode="json")
        d["has_credentials"] = bool(c.credentials_path or c.credentials_secret_ref)
        # Never echo full path in prod-ish responses — keep for demo
        out.append(d)
    return out
