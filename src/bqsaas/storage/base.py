"""Abstract repository protocol for resource persistence."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from bqsaas.kinds.base import Kind, Resource, ResourceStatus

T = TypeVar("T", bound=Resource)


@runtime_checkable
class Repository(Protocol):
    """Storage protocol for typed SaaS resources.

    Implementations must enforce tenant isolation on list/query operations
    unless an explicit platform-admin path is used.
    """

    def create(self, resource: T) -> T:
        """Persist a new resource. Raises if id already exists."""
        ...

    def get(
        self,
        kind: Kind,
        resource_id: str,
        *,
        tenant_id: str | None = None,
        include_deleted: bool = False,
    ) -> Resource | None:
        """Fetch a single resource by kind and id.

        When ``tenant_id`` is provided, the resource must belong to that
        tenant (or be a Tenant whose id equals ``tenant_id``).
        """
        ...

    def update(self, resource: T) -> T:
        """Replace an existing resource. Raises if missing."""
        ...

    def soft_delete(
        self,
        kind: Kind,
        resource_id: str,
        *,
        tenant_id: str | None = None,
    ) -> Resource | None:
        """Mark a resource as deleted. Returns the updated resource or None."""
        ...

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
        """List resources of a kind with optional filters.

        Tenant isolation: when ``platform_admin`` is False, ``tenant_id``
        is required (except when listing Kind.TENANT for admin tooling).
        """
        ...

    def get_by_field(
        self,
        kind: Kind,
        field: str,
        value: Any,
        *,
        tenant_id: str | None = None,
        include_deleted: bool = False,
    ) -> Resource | None:
        """Return the first resource matching ``field == value``."""
        ...
