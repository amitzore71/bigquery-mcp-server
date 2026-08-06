"""Seed a demo tenant for local development."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from bqsaas.auth.dependencies import create_api_key
from bqsaas.kinds import Kind, generate_id
from bqsaas.models import (
    DataConnection,
    PlanTier,
    Subscription,
    Tenant,
    User,
    UserRole,
    Workspace,
)
from bqsaas.models.subscription import DEFAULT_PLAN_LIMITS

if TYPE_CHECKING:
    from bqsaas.storage import MemoryStore

DEMO_TENANT_SLUG = "demo"
DEMO_TENANT_NAME = "Demo School District"
DEMO_USER_EMAIL = "demo@example.com"
DEMO_USER_NAME = "Demo Owner"
DEMO_WORKSPACE_NAME = "Main"
DEMO_API_KEY_NAME = "Local Dev Key"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _load_config() -> Any:
    try:
        from bqsaas.config import get_settings

        return get_settings()
    except Exception:
        return None


def _project_id_from_config() -> str:
    cfg = _load_config()
    if cfg is None:
        return "practice-project-481414"
    return str(getattr(cfg, "gcp_project_id", None) or "practice-project-481414")


def _dataset_id_from_config() -> str:
    cfg = _load_config()
    if cfg is None:
        return "school_data"
    return str(getattr(cfg, "dataset_id", None) or "school_data")


def _credentials_path_from_config() -> str:
    cfg = _load_config()
    path = "service-account.json"
    if cfg is not None:
        path = str(getattr(cfg, "service_account_path", None) or path)
    if not os.path.isabs(path):
        path = os.path.abspath(path)
    return path


def bootstrap_demo(store: MemoryStore, *, force: bool = False) -> dict[str, Any]:
    """Create demo tenant + user + workspace + connection + free sub + API key.

    Returns dict with tenant_id, user_id, raw_api_key, workspace_id, etc.
    """
    now = _utcnow()
    existing = _find_tenant_by_slug(store, DEMO_TENANT_SLUG)

    if existing is not None and not force:
        tenant = existing
        user = _ensure_owner_user(store, tenant.id, now)
        workspace = _ensure_workspace(store, tenant.id, now)
        connection = _ensure_connection(store, tenant.id, workspace.id, now)
        subscription = _ensure_subscription(store, tenant.id, now)
        api_key, raw_api_key = create_api_key(
            store,
            tenant_id=tenant.id,
            user_id=user.id,
            name=DEMO_API_KEY_NAME,
            scopes=["*"],
            env="dev",
        )
        store.mark_ready()
        return {
            "tenant_id": tenant.id,
            "user_id": user.id,
            "raw_api_key": raw_api_key,
            "api_key": raw_api_key,
            "workspace_id": workspace.id,
            "connection_id": connection.id,
            "subscription_id": subscription.id,
            "api_key_id": api_key.id,
            "reused": True,
        }

    tenant = Tenant(
        id=generate_id(Kind.TENANT),
        name=DEMO_TENANT_NAME,
        slug=DEMO_TENANT_SLUG,
        plan_id="free",
        created_at=now,
        updated_at=now,
    )
    store.save(tenant)

    user = User(
        id=generate_id(Kind.USER),
        tenant_id=tenant.id,
        email=DEMO_USER_EMAIL,
        name=DEMO_USER_NAME,
        role=UserRole.OWNER,
        created_at=now,
        updated_at=now,
    )
    store.save(user)

    tenant.owner_user_id = user.id
    store.save(tenant)

    workspace = Workspace(
        id=generate_id(Kind.WORKSPACE),
        tenant_id=tenant.id,
        name=DEMO_WORKSPACE_NAME,
        description="Default workspace for local development",
        created_at=now,
        updated_at=now,
    )
    store.save(workspace)

    creds_path = _credentials_path_from_config()
    connection = DataConnection(
        id=generate_id(Kind.DATA_CONNECTION),
        tenant_id=tenant.id,
        workspace_id=workspace.id,
        name="Demo BigQuery",
        provider="bigquery",
        project_id=_project_id_from_config(),
        dataset_id=_dataset_id_from_config(),
        credentials_path=creds_path,
        credentials_secret_ref=None if os.path.isfile(creds_path) else "env:ADC",
        created_at=now,
        updated_at=now,
    )
    # If file missing, use secret_ref so model validation passes
    if not os.path.isfile(creds_path):
        connection = DataConnection(
            id=connection.id,
            tenant_id=tenant.id,
            workspace_id=workspace.id,
            name="Demo BigQuery",
            provider="bigquery",
            project_id=_project_id_from_config(),
            dataset_id=_dataset_id_from_config(),
            credentials_path=None,
            credentials_secret_ref="env:ADC",
            created_at=now,
            updated_at=now,
        )
    store.save(connection)

    # Pro plan for local demo so workspace/session caps are comfortable
    subscription = Subscription.for_plan(
        tenant_id=tenant.id,
        plan=PlanTier.PRO,
        limits=DEFAULT_PLAN_LIMITS,
    )
    subscription.id = generate_id(Kind.SUBSCRIPTION)
    store.save(subscription)
    tenant.plan_id = "pro"
    store.save(tenant)

    api_key, raw_api_key = create_api_key(
        store,
        tenant_id=tenant.id,
        user_id=user.id,
        name=DEMO_API_KEY_NAME,
        scopes=["*"],
        env="dev",
    )

    store.mark_ready()
    return {
        "tenant_id": tenant.id,
        "user_id": user.id,
        "raw_api_key": raw_api_key,
        "api_key": raw_api_key,
        "workspace_id": workspace.id,
        "connection_id": connection.id,
        "subscription_id": subscription.id,
        "api_key_id": api_key.id,
        "reused": False,
    }


def create_demo_tenant(store: MemoryStore, *, force: bool = False) -> dict[str, Any]:
    """Alias for :func:`bootstrap_demo`."""
    return bootstrap_demo(store, force=force)


def _find_tenant_by_slug(store: MemoryStore, slug: str) -> Any | None:
    found = store.find_one(Kind.TENANT, slug=slug)
    if found is not None:
        return found
    for item in store.list(Kind.TENANT, platform_admin=True):
        if getattr(item, "slug", None) == slug:
            return item
    return None


def _list_by_tenant(store: MemoryStore, kind: Kind, tenant_id: str) -> list[Any]:
    return list(store.list_by_tenant(kind, tenant_id))


def _ensure_owner_user(store: MemoryStore, tenant_id: str, now: datetime) -> Any:
    users = _list_by_tenant(store, Kind.USER, tenant_id)
    for u in users:
        if getattr(u, "email", None) == DEMO_USER_EMAIL:
            return u
    if users:
        return users[0]

    user = User(
        id=generate_id(Kind.USER),
        tenant_id=tenant_id,
        email=DEMO_USER_EMAIL,
        name=DEMO_USER_NAME,
        role=UserRole.OWNER,
        created_at=now,
        updated_at=now,
    )
    store.save(user)
    return user


def _ensure_workspace(store: MemoryStore, tenant_id: str, now: datetime) -> Any:
    workspaces = _list_by_tenant(store, Kind.WORKSPACE, tenant_id)
    for ws in workspaces:
        if getattr(ws, "name", None) == DEMO_WORKSPACE_NAME:
            return ws
    if workspaces:
        return workspaces[0]

    workspace = Workspace(
        id=generate_id(Kind.WORKSPACE),
        tenant_id=tenant_id,
        name=DEMO_WORKSPACE_NAME,
        description="Default workspace for local development",
        created_at=now,
        updated_at=now,
    )
    store.save(workspace)
    return workspace


def _ensure_connection(
    store: MemoryStore,
    tenant_id: str,
    workspace_id: str,
    now: datetime,
) -> Any:
    connections = _list_by_tenant(store, Kind.DATA_CONNECTION, tenant_id)
    if connections:
        return connections[0]

    creds_path = _credentials_path_from_config()
    if os.path.isfile(creds_path):
        connection = DataConnection(
            id=generate_id(Kind.DATA_CONNECTION),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            name="Demo BigQuery",
            provider="bigquery",
            project_id=_project_id_from_config(),
            dataset_id=_dataset_id_from_config(),
            credentials_path=creds_path,
            created_at=now,
            updated_at=now,
        )
    else:
        connection = DataConnection(
            id=generate_id(Kind.DATA_CONNECTION),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            name="Demo BigQuery",
            provider="bigquery",
            project_id=_project_id_from_config(),
            dataset_id=_dataset_id_from_config(),
            credentials_secret_ref="env:ADC",
            created_at=now,
            updated_at=now,
        )
    store.save(connection)
    return connection


def _ensure_subscription(store: MemoryStore, tenant_id: str, now: datetime) -> Any:
    existing = store.find_subscription_by_tenant(tenant_id)
    if existing is not None:
        return existing

    subscription = Subscription.for_plan(tenant_id=tenant_id, plan=PlanTier.FREE)
    subscription.id = generate_id(Kind.SUBSCRIPTION)
    store.save(subscription)
    return subscription
