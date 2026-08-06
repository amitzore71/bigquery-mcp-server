"""Thread-safe in-memory resource store with tenant isolation."""

from __future__ import annotations

import threading
from typing import Any, TypeVar

from bqsaas.kinds.base import Kind, Resource, ResourceStatus, _utc_now

T = TypeVar("T", bound=Resource)


class ResourceNotFoundError(LookupError):
    """Raised when a required resource is missing from the store."""


class ResourceConflictError(ValueError):
    """Raised when creating a resource with a duplicate id."""


class TenantIsolationError(PermissionError):
    """Raised when a tenant-scoped operation lacks a tenant_id."""


class MemoryStore:
    """In-memory repository keyed by ``(kind, id)`` with tenant isolation.

    All public methods are protected by a single :class:`threading.RLock`.
    Returned resources are deep copies so callers cannot mutate the store
    without going through update/create/save.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._data: dict[tuple[str, str], Resource] = {}
        self._ready = False

    # ------------------------------------------------------------------ ready
    def mark_ready(self) -> None:
        self._ready = True

    @property
    def is_ready(self) -> bool:
        return self._ready

    def _key(self, kind: Kind, resource_id: str) -> tuple[str, str]:
        return (kind.value, resource_id)

    def _clone(self, resource: Resource) -> Resource:
        return resource.model_copy(deep=True)

    def _belongs_to_tenant(self, resource: Resource, tenant_id: str) -> bool:
        if resource.kind is Kind.TENANT:
            return resource.id == tenant_id
        return resource.tenant_id == tenant_id

    def create(self, resource: T) -> T:
        """Persist a new resource.

        Raises:
            ResourceConflictError: If ``(kind, id)`` already exists.
            TypeError: If resource is not a Resource instance.
        """
        if not isinstance(resource, Resource):
            raise TypeError(f"expected Resource, got {type(resource)!r}")

        with self._lock:
            key = self._key(resource.kind, resource.id)
            if key in self._data:
                raise ResourceConflictError(
                    f"resource already exists: {resource.kind.value}/{resource.id}"
                )
            stored = self._clone(resource)
            self._data[key] = stored
            return self._clone(stored)  # type: ignore[return-value]

    def save(self, resource: T) -> T:
        """Create or replace a resource (service-layer convenience)."""
        if not isinstance(resource, Resource):
            raise TypeError(f"expected Resource, got {type(resource)!r}")
        with self._lock:
            key = self._key(resource.kind, resource.id)
            existing = self._data.get(key)
            if existing is None:
                stored = self._clone(resource)
                self._data[key] = stored
                return self._clone(stored)  # type: ignore[return-value]
            if existing.tenant_id != resource.tenant_id:
                raise TenantIsolationError(
                    "cannot change tenant_id of an existing resource"
                )
            updated = self._clone(resource)
            if updated.updated_at <= existing.updated_at:
                updated.updated_at = _utc_now()
            self._data[key] = updated
            return self._clone(updated)  # type: ignore[return-value]

    def get(
        self,
        kind: Kind,
        resource_id: str,
        *,
        tenant_id: str | None = None,
        include_deleted: bool = False,
    ) -> Resource | None:
        """Fetch a single resource by kind and id with optional tenant check."""
        with self._lock:
            stored = self._data.get(self._key(kind, resource_id))
            if stored is None:
                return None
            if not include_deleted and stored.status is ResourceStatus.DELETED:
                return None
            if tenant_id is not None and not self._belongs_to_tenant(
                stored, tenant_id
            ):
                return None
            return self._clone(stored)

    def update(self, resource: T) -> T:
        """Replace an existing resource (must already exist)."""
        if not isinstance(resource, Resource):
            raise TypeError(f"expected Resource, got {type(resource)!r}")

        with self._lock:
            key = self._key(resource.kind, resource.id)
            existing = self._data.get(key)
            if existing is None:
                raise ResourceNotFoundError(
                    f"resource not found: {resource.kind.value}/{resource.id}"
                )
            if existing.tenant_id != resource.tenant_id:
                raise TenantIsolationError(
                    "cannot change tenant_id of an existing resource"
                )
            if existing.kind is not resource.kind:
                raise ValueError("cannot change kind of an existing resource")

            updated = self._clone(resource)
            if updated.updated_at <= existing.updated_at:
                updated.updated_at = _utc_now()
            self._data[key] = updated
            return self._clone(updated)  # type: ignore[return-value]

    def soft_delete(
        self,
        kind: Kind,
        resource_id: str,
        *,
        tenant_id: str | None = None,
    ) -> Resource | None:
        """Set status to DELETED. Returns the updated resource or None."""
        with self._lock:
            key = self._key(kind, resource_id)
            stored = self._data.get(key)
            if stored is None:
                return None
            if tenant_id is not None and not self._belongs_to_tenant(
                stored, tenant_id
            ):
                return None
            if stored.status is ResourceStatus.DELETED:
                return self._clone(stored)

            deleted = self._clone(stored)
            deleted.status = ResourceStatus.DELETED
            deleted.updated_at = _utc_now()
            self._data[key] = deleted
            return self._clone(deleted)

    def delete(self, kind: Kind, resource_id: str) -> bool:
        """Hard-delete a resource by kind/id. Returns True if removed."""
        with self._lock:
            key = self._key(kind, resource_id)
            if key in self._data:
                del self._data[key]
                return True
            return False

    def list(
        self,
        kind: Kind,
        *,
        tenant_id: str | None = None,
        status: ResourceStatus | None = ResourceStatus.ACTIVE,
        limit: int = 100,
        offset: int = 0,
        platform_admin: bool = False,
    ) -> list[Resource]:
        """List resources of a kind with tenant isolation and pagination."""
        if limit < 0 or offset < 0:
            raise ValueError("limit and offset must be non-negative")
        limit = min(limit, 1000)

        # Platform-level kinds or explicit admin can list without tenant scope.
        if (
            not platform_admin
            and kind is not Kind.TENANT
            and tenant_id is None
        ):
            # Soften for internal scans (API key auth, message listing):
            # allow unscoped list for API_KEY and MESSAGE kinds.
            if kind not in (Kind.API_KEY, Kind.MESSAGE, Kind.SUBSCRIPTION):
                raise TenantIsolationError(
                    "tenant_id is required for list() unless platform_admin=True "
                    "or kind is TENANT"
                )

        with self._lock:
            matches: list[Resource] = []
            for (kind_value, _), resource in self._data.items():
                if kind_value != kind.value:
                    continue
                if status is not None and resource.status is not status:
                    continue
                if tenant_id is not None and not self._belongs_to_tenant(
                    resource, tenant_id
                ):
                    continue
                matches.append(resource)

            matches.sort(key=lambda r: (r.created_at, r.id))
            page = matches[offset : offset + limit]
            return [self._clone(r) for r in page]

    def list_by_tenant(
        self,
        kind: Kind,
        tenant_id: str,
        *,
        status: ResourceStatus | None = ResourceStatus.ACTIVE,
        limit: int = 1000,
    ) -> list[Resource]:
        """List resources of ``kind`` belonging to ``tenant_id``."""
        return self.list(
            kind, tenant_id=tenant_id, status=status, limit=limit, offset=0
        )

    def get_by_field(
        self,
        kind: Kind,
        field: str,
        value: Any,
        *,
        tenant_id: str | None = None,
        include_deleted: bool = False,
    ) -> Resource | None:
        """Return the first resource where ``getattr(resource, field) == value``."""
        if not field or not isinstance(field, str):
            raise ValueError("field must be a non-empty string")
        if field.startswith("_"):
            raise ValueError("cannot query private fields")

        with self._lock:
            for (kind_value, _), resource in self._data.items():
                if kind_value != kind.value:
                    continue
                if not include_deleted and resource.status is ResourceStatus.DELETED:
                    continue
                if tenant_id is not None and not self._belongs_to_tenant(
                    resource, tenant_id
                ):
                    continue
                if not hasattr(resource, field):
                    continue
                if getattr(resource, field) == value:
                    return self._clone(resource)
            return None

    def find_one(self, kind: Kind, **fields: Any) -> Resource | None:
        """Return the first resource matching all keyword field filters."""
        if not fields:
            items = self.list(kind, platform_admin=True, limit=1)
            return items[0] if items else None

        with self._lock:
            for (kind_value, _), resource in self._data.items():
                if kind_value != kind.value:
                    continue
                if resource.status is ResourceStatus.DELETED:
                    continue
                if all(getattr(resource, k, None) == v for k, v in fields.items()):
                    return self._clone(resource)
            return None

    def find_many(self, kind: Kind, **fields: Any) -> list[Resource]:
        """Return all resources matching keyword field filters."""
        with self._lock:
            matches: list[Resource] = []
            for (kind_value, _), resource in self._data.items():
                if kind_value != kind.value:
                    continue
                if resource.status is ResourceStatus.DELETED:
                    continue
                if all(getattr(resource, k, None) == v for k, v in fields.items()):
                    matches.append(self._clone(resource))
            matches.sort(key=lambda r: (r.created_at, r.id))
            return matches

    def find_api_key_by_hash(self, key_hash: str) -> Resource | None:
        """Lookup an API key by its SHA-256 digest."""
        return self.find_one(Kind.API_KEY, key_hash=key_hash)

    def find_subscription_by_tenant(self, tenant_id: str) -> Resource | None:
        """Return the subscription for a tenant, if any."""
        return self.find_one(Kind.SUBSCRIPTION, tenant_id=tenant_id)

    def list_messages_for_session(self, session_id: str) -> list[Resource]:
        """Return messages for a chat session ordered by created_at."""
        return self.find_many(Kind.MESSAGE, session_id=session_id)

    def count(
        self,
        kind: Kind,
        *,
        tenant_id: str | None = None,
        status: ResourceStatus | None = ResourceStatus.ACTIVE,
        platform_admin: bool = False,
    ) -> int:
        """Count resources matching the same filters as :meth:`list`."""
        return len(
            self.list(
                kind,
                tenant_id=tenant_id,
                status=status,
                limit=1000,
                platform_admin=platform_admin,
            )
        )

    def clear(self) -> None:
        """Remove all resources (test helper)."""
        with self._lock:
            self._data.clear()
            self._ready = False

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)
