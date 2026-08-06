"""Tenant, user, workspace, and subscription management."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Sequence

from bqsaas.auth.dependencies import create_api_key
from bqsaas.billing.meter import UsageMeter
from bqsaas.billing.plans import get_plan
from bqsaas.errors import ForbiddenError, NotFoundError, ValidationError
from bqsaas.kinds import Kind, generate_id
from bqsaas.kinds.base import ResourceStatus
from bqsaas.models import PlanTier, Subscription, Tenant, User, UserRole, Workspace
from bqsaas.models.subscription import DEFAULT_PLAN_LIMITS

if TYPE_CHECKING:
    from bqsaas.models import ApiKey
    from bqsaas.storage import MemoryStore

_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _slugify(value: str) -> str:
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return s


class TenantService:
    """CRUD and membership operations for tenants and related resources."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._meter = UsageMeter(store)

    def create_tenant(
        self,
        name: str,
        *,
        slug: str | None = None,
        plan_id: str = "free",
        owner_email: str | None = None,
        owner_name: str | None = None,
    ) -> dict[str, Any]:
        """Create a tenant, subscription, and optional owner user."""
        if not name or not name.strip():
            raise ValidationError("Tenant name is required", field="name")

        resolved_slug = slug.strip().lower() if slug else _slugify(name)
        if not resolved_slug or not _SLUG_RE.match(resolved_slug):
            raise ValidationError(
                "Invalid slug; use lowercase letters, numbers, and hyphens",
                field="slug",
            )

        if self._find_tenant_by_slug(resolved_slug) is not None:
            raise ValidationError(
                f"Tenant slug '{resolved_slug}' is already taken",
                field="slug",
            )

        get_plan(plan_id)
        now = _utcnow()
        tenant = Tenant(
            id=generate_id(Kind.TENANT),
            name=name.strip(),
            slug=resolved_slug,
            plan_id=plan_id,
            created_at=now,
            updated_at=now,
        )
        self._store.save(tenant)

        tier = PlanTier(plan_id.lower())
        subscription = Subscription.for_plan(
            tenant_id=tenant.id,
            plan=tier,
            limits=DEFAULT_PLAN_LIMITS,
        )
        subscription.id = generate_id(Kind.SUBSCRIPTION)
        self._store.save(subscription)

        owner: User | None = None
        if owner_email:
            owner = self.add_user(
                tenant.id,
                email=owner_email,
                name=owner_name or owner_email.split("@")[0],
                role="owner",
            )
            tenant.owner_user_id = owner.id
            self._store.save(tenant)

        return {
            "tenant": tenant,
            "subscription": subscription,
            "owner": owner,
        }

    def get_tenant(self, tenant_id: str) -> Tenant:
        tenant = self._store.get(Kind.TENANT, tenant_id)
        if tenant is None:
            raise NotFoundError(
                "Tenant not found", resource="tenant", resource_id=tenant_id
            )
        return tenant  # type: ignore[return-value]

    def get_tenant_by_slug(self, slug: str) -> Tenant:
        tenant = self._find_tenant_by_slug(slug)
        if tenant is None:
            raise NotFoundError(
                "Tenant not found", resource="tenant", resource_id=slug
            )
        return tenant

    def list_tenants(self) -> list[Tenant]:
        return list(self._store.list(Kind.TENANT, platform_admin=True))  # type: ignore[return-value]

    def update_tenant(
        self,
        tenant_id: str,
        *,
        name: str | None = None,
        is_active: bool | None = None,
    ) -> Tenant:
        tenant = self.get_tenant(tenant_id)
        if name is not None:
            if not name.strip():
                raise ValidationError("Tenant name cannot be empty", field="name")
            tenant.name = name.strip()
        if is_active is not None:
            tenant.status = (
                ResourceStatus.ACTIVE if is_active else ResourceStatus.SUSPENDED
            )
        tenant.touch()
        self._store.save(tenant)
        return tenant

    def add_user(
        self,
        tenant_id: str,
        *,
        email: str,
        name: str,
        role: str = "member",
    ) -> User:
        self.get_tenant(tenant_id)
        if not email or "@" not in email:
            raise ValidationError("Valid email is required", field="email")
        if not name or not name.strip():
            raise ValidationError("User name is required", field="name")

        existing_users = self.list_users(tenant_id)
        email_lower = email.strip().lower()
        for u in existing_users:
            if getattr(u, "email", "").lower() == email_lower:
                raise ValidationError(
                    f"User with email '{email_lower}' already exists in tenant",
                    field="email",
                )

        self._meter.check_resource_limit(tenant_id, "users", len(existing_users))

        user = User(
            id=generate_id(Kind.USER),
            tenant_id=tenant_id,
            email=email_lower,
            name=name.strip(),
            role=role,
        )
        self._store.save(user)
        return user

    def get_user(self, tenant_id: str, user_id: str) -> User:
        user = self._store.get(Kind.USER, user_id)
        if user is None:
            raise NotFoundError("User not found", resource="user", resource_id=user_id)
        if getattr(user, "tenant_id", None) != tenant_id:
            raise ForbiddenError("User does not belong to tenant")
        return user  # type: ignore[return-value]

    def list_users(self, tenant_id: str) -> list[User]:
        self.get_tenant(tenant_id)
        return self._store.list_by_tenant(Kind.USER, tenant_id)  # type: ignore[return-value]

    def create_workspace(
        self,
        tenant_id: str,
        name: str,
        *,
        description: str = "",
    ) -> Workspace:
        self.get_tenant(tenant_id)
        if not name or not name.strip():
            raise ValidationError("Workspace name is required", field="name")

        existing = self.list_workspaces(tenant_id)
        self._meter.check_resource_limit(tenant_id, "workspaces", len(existing))

        workspace = Workspace(
            id=generate_id(Kind.WORKSPACE),
            tenant_id=tenant_id,
            name=name.strip(),
            description=description or "",
        )
        self._store.save(workspace)
        return workspace

    def get_workspace(self, tenant_id: str, workspace_id: str) -> Workspace:
        workspace = self._store.get(Kind.WORKSPACE, workspace_id)
        if workspace is None:
            raise NotFoundError(
                "Workspace not found",
                resource="workspace",
                resource_id=workspace_id,
            )
        if getattr(workspace, "tenant_id", None) != tenant_id:
            raise ForbiddenError("Workspace does not belong to tenant")
        return workspace  # type: ignore[return-value]

    def list_workspaces(self, tenant_id: str) -> list[Workspace]:
        self.get_tenant(tenant_id)
        return self._store.list_by_tenant(Kind.WORKSPACE, tenant_id)  # type: ignore[return-value]

    def update_workspace(
        self,
        tenant_id: str,
        workspace_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        is_active: bool | None = None,
    ) -> Workspace:
        workspace = self.get_workspace(tenant_id, workspace_id)
        if name is not None:
            if not name.strip():
                raise ValidationError("Workspace name cannot be empty", field="name")
            workspace.name = name.strip()
        if description is not None:
            workspace.description = description
        if is_active is not None:
            workspace.status = (
                ResourceStatus.ACTIVE if is_active else ResourceStatus.SUSPENDED
            )
        workspace.touch()
        self._store.save(workspace)
        return workspace

    def delete_workspace(self, tenant_id: str, workspace_id: str) -> None:
        self.get_workspace(tenant_id, workspace_id)
        self._store.soft_delete(Kind.WORKSPACE, workspace_id, tenant_id=tenant_id)

    def get_subscription(self, tenant_id: str) -> Subscription:
        self.get_tenant(tenant_id)
        return self._meter.get_subscription(tenant_id)

    def change_plan(self, tenant_id: str, plan_id: str) -> Subscription:
        self.get_tenant(tenant_id)
        plan = get_plan(plan_id)
        sub = self._meter.get_subscription(tenant_id)
        sub.plan = PlanTier(plan.id)
        sub.queries_limit = plan.daily_query_limit
        sub.touch()
        self._store.save(sub)
        return sub

    def create_api_key_for_user(
        self,
        tenant_id: str,
        user_id: str,
        name: str,
        scopes: Sequence[str],
        *,
        env: str = "live",
    ) -> tuple[ApiKey, str]:
        self.get_user(tenant_id, user_id)
        return create_api_key(
            self._store,
            tenant_id=tenant_id,
            user_id=user_id,
            name=name,
            scopes=scopes,
            env=env,
        )

    def usage(self, tenant_id: str) -> dict[str, Any]:
        self.get_tenant(tenant_id)
        return self._meter.usage_snapshot(tenant_id)

    def _find_tenant_by_slug(self, slug: str) -> Tenant | None:
        found = self._store.find_one(Kind.TENANT, slug=slug)
        if found is not None:
            return found  # type: ignore[return-value]
        for item in self._store.list(Kind.TENANT, platform_admin=True):
            if getattr(item, "slug", None) == slug:
                return item  # type: ignore[return-value]
        return None
