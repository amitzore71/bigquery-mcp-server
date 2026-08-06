"""Auth: API key minting (admin) and current identity."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from bqsaas.auth.dependencies import AuthContext, create_api_key
from bqsaas.auth.deps import get_auth_context, get_store, require_admin
from bqsaas.billing.meter import UsageMeter
from bqsaas.billing.plans import get_plan
from bqsaas.services.tenant_service import TenantService
from bqsaas.storage.memory import MemoryStore

router = APIRouter(prefix="/v1", tags=["auth"])


class CreateApiKeyRequest(BaseModel):
    name: str = "default"
    user_email: Optional[str] = None
    scopes: list[str] = Field(default_factory=lambda: ["*"])


class CreateApiKeyResponse(BaseModel):
    id: str
    name: str
    api_key: str
    key_prefix: str
    tenant_id: str
    user_id: str
    scopes: list[str]


class MeResponse(BaseModel):
    user: dict
    tenant: dict
    plan: dict
    usage: dict
    is_admin: bool
    scopes: list[str]


@router.post("/auth/api-keys", response_model=CreateApiKeyResponse)
async def mint_api_key(
    body: CreateApiKeyRequest,
    ctx: AuthContext = Depends(require_admin),
    store: MemoryStore = Depends(get_store),
) -> CreateApiKeyResponse:
    """Mint a new API key for the current tenant (admin only)."""
    svc = TenantService(store)
    user = ctx.user
    if body.user_email:
        users = svc.list_users(ctx.tenant_id)
        found = next(
            (u for u in users if u.email.lower() == body.user_email.lower()),
            None,
        )
        if found is None:
            user = svc.add_user(
                ctx.tenant_id,
                email=body.user_email,
                name=body.user_email.split("@")[0],
                role="member",
            )
        else:
            user = found

    record, raw = create_api_key(
        store,
        tenant_id=ctx.tenant_id,
        user_id=user.id,
        name=body.name,
        scopes=body.scopes or ["*"],
        env="live",
    )
    return CreateApiKeyResponse(
        id=record.id,
        name=record.name,
        api_key=raw,
        key_prefix=record.key_prefix,
        tenant_id=record.tenant_id or ctx.tenant_id,
        user_id=record.user_id,
        scopes=list(record.scopes),
    )


@router.get("/me", response_model=MeResponse)
async def me(
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> MeResponse:
    """Current user + tenant + plan + usage."""
    meter = UsageMeter(store)
    try:
        plan = meter.get_plan_for_tenant(ctx.tenant_id)
        usage = meter.usage_snapshot(ctx.tenant_id)
        plan_dict = {
            "id": plan.id,
            "name": plan.name,
            "daily_query_limit": plan.daily_query_limit,
            "max_workspaces": plan.max_workspaces,
            "max_connections": plan.max_connections,
        }
    except Exception:
        plan_dict = {"id": getattr(ctx.tenant, "plan_id", "free")}
        usage = {}

    return MeResponse(
        user=ctx.user.model_dump(mode="json"),
        tenant=ctx.tenant.model_dump(mode="json"),
        plan=plan_dict,
        usage=usage if isinstance(usage, dict) else {},
        is_admin=ctx.is_admin,
        scopes=list(ctx.scopes),
    )
