"""Core Kind system: resource taxonomy, status, refs, and base Resource model.

Mirrors Kubernetes/GCP-style typed resource identity for multi-tenant SaaS.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Kind(str, Enum):
    """Typed resource kinds in the BigQuery SaaS platform."""

    TENANT = "tenant"
    USER = "user"
    WORKSPACE = "workspace"
    DATA_CONNECTION = "data_connection"
    API_KEY = "api_key"
    CHAT_SESSION = "chat_session"
    MESSAGE = "message"
    # Alias for services written against CHAT_MESSAGE
    CHAT_MESSAGE = "message"
    SUBSCRIPTION = "subscription"
    USAGE_EVENT = "usage_event"
    QUERY_JOB = "query_job"


class ResourceStatus(str, Enum):
    """Lifecycle status for platform resources."""

    ACTIVE = "active"
    SUSPENDED = "suspended"
    DELETED = "deleted"


def _utc_now() -> datetime:
    """Return the current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _generate_raw_id() -> str:
    """Generate a unique raw identifier (ULID if available, else UUID4 hex)."""
    try:
        from ulid import ULID  # type: ignore[import-untyped]

        return str(ULID())
    except ImportError:
        return uuid4().hex


def generate_resource_id(kind: Kind) -> str:
    """Generate a fully-qualified resource ID.

    Format: ``{kind_value}_{raw_id}`` e.g. ``tenant_01HXYZ...`` or
    ``user_a1b2c3d4e5f6...`` when ULID is unavailable.

    Args:
        kind: Resource kind to prefix the ID with.

    Returns:
        A unique, kind-prefixed resource identifier string.
    """
    return f"{kind.value}_{_generate_raw_id()}"


class ResourceRef(BaseModel):
    """Lightweight cross-reference to another resource by kind and id."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Kind
    id: str

    @field_validator("id")
    @classmethod
    def _id_must_be_nonempty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("resource id must be a non-empty string")
        return value.strip()

    def __str__(self) -> str:
        return f"{self.kind.value}/{self.id}"

    @classmethod
    def from_resource(cls, resource: Resource) -> ResourceRef:
        """Build a ref from a full Resource instance."""
        return cls(kind=resource.kind, id=resource.id)

    @classmethod
    def parse(cls, value: str) -> ResourceRef:
        """Parse ``kind/id`` or a bare kind-prefixed id into a ResourceRef.

        Accepts:
          - ``tenant/tenant_01HXYZ`` (kind/id form)
          - ``tenant_01HXYZ`` (id form; kind inferred from prefix)
        """
        if "/" in value:
            kind_str, resource_id = value.split("/", 1)
            return cls(kind=Kind(kind_str), id=resource_id)
        # Infer kind from the leading segment of the id.
        for kind in Kind:
            prefix = f"{kind.value}_"
            if value.startswith(prefix):
                return cls(kind=kind, id=value)
        raise ValueError(f"cannot parse ResourceRef from {value!r}")


class Resource(BaseModel):
    """Base model for all platform resources.

    Every resource carries a typed ``kind``, a kind-prefixed ``id``, optional
    ``tenant_id`` (required for all kinds except :attr:`Kind.TENANT`),
    lifecycle timestamps, status, and free-form labels/annotations.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: Kind
    id: str = ""
    tenant_id: str | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)
    status: ResourceStatus = ResourceStatus.ACTIVE
    labels: dict[str, str] = Field(default_factory=dict)
    annotations: dict[str, str] = Field(default_factory=dict)

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _ensure_aware_utc(cls, value: Any) -> datetime:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, datetime):
            if value.tzinfo is None:
                return value.replace(tzinfo=timezone.utc)
            return value.astimezone(timezone.utc)
        raise TypeError(f"expected datetime, got {type(value)!r}")

    @model_validator(mode="after")
    def _finalize_identity(self) -> Self:
        if not self.id:
            object.__setattr__(self, "id", generate_resource_id(self.kind))
        if self.kind is Kind.TENANT:
            # Platform root: no parent tenant. Optionally self-ref via id.
            if self.tenant_id is not None and self.tenant_id != self.id:
                raise ValueError(
                    "Tenant resources must have tenant_id=None "
                    "(or equal to their own id for self-reference)"
                )
        else:
            if not self.tenant_id:
                raise ValueError(
                    f"tenant_id is required for kind {self.kind.value!r}"
                )
        return self

    def touch(self) -> Self:
        """Bump ``updated_at`` to now (UTC). Returns self for chaining."""
        self.updated_at = _utc_now()
        return self

    def soft_delete(self) -> Self:
        """Mark resource as deleted and bump timestamps."""
        self.status = ResourceStatus.DELETED
        return self.touch()

    def to_ref(self) -> ResourceRef:
        """Return a ResourceRef pointing at this resource."""
        return ResourceRef(kind=self.kind, id=self.id)

    def is_active(self) -> bool:
        """True when status is ACTIVE."""
        return self.status is ResourceStatus.ACTIVE
