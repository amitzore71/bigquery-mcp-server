"""Tenant domain model — multi-tenant organization root."""

from __future__ import annotations

import re
from typing import Self

from pydantic import ConfigDict, Field, field_validator, model_validator

from bqsaas.kinds.base import Kind, Resource

_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")


class Tenant(Resource):
    """Organization root resource for multi-tenant isolation.

    ``tenant_id`` is always ``None`` for Tenant resources; the resource
    ``id`` itself is the tenant identifier used by child resources.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: Kind = Kind.TENANT
    tenant_id: str | None = None

    name: str = Field(..., min_length=1, max_length=200)
    slug: str = Field(
        ...,
        min_length=1,
        max_length=63,
        description="URL-safe unique slug within the platform",
    )
    plan_id: str = Field(
        default="free",
        description="Billing plan identifier (e.g. free, pro, enterprise)",
    )
    owner_user_id: str | None = Field(
        default=None,
        description="User id of the tenant owner (set after first user is created)",
    )

    @field_validator("slug")
    @classmethod
    def _normalize_slug(cls, value: str) -> str:
        slug = value.strip().lower()
        if not _SLUG_RE.match(slug):
            raise ValueError(
                "slug must be 1-63 chars, lowercase alphanumeric, "
                "hyphens allowed but not at start/end"
            )
        return slug

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("name must not be empty")
        return name

    @model_validator(mode="after")
    def _tenant_identity(self) -> Self:
        # Ensure kind is fixed and tenant_id stays unset for root tenants.
        if self.kind is not Kind.TENANT:
            raise ValueError("Tenant.kind must be Kind.TENANT")
        if self.tenant_id is not None and self.tenant_id != self.id:
            raise ValueError("Tenant.tenant_id must be None or equal to id")
        return self

    def as_tenant_id(self) -> str:
        """Return this tenant's id for use as ``tenant_id`` on child resources."""
        return self.id
