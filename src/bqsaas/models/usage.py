"""Usage event domain model for metering and billing."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import ConfigDict, Field, field_validator

from bqsaas.kinds.base import Kind, Resource, ResourceRef


class UsageEventType(str, Enum):
    """Categories of billable or metered platform events."""

    QUERY = "query"
    CHAT_MESSAGE = "chat_message"
    TOOL_CALL = "tool_call"
    BYTES_BILLED = "bytes_billed"
    API_REQUEST = "api_request"
    OTHER = "other"


class UsageEvent(Resource):
    """Immutable-style usage event for metering, quotas, and analytics.

    ``resource_ref`` optionally points at the subject of the event
    (e.g. a chat session or data connection).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: Kind = Kind.USAGE_EVENT

    event_type: UsageEventType = UsageEventType.OTHER
    units: float = Field(
        default=1.0,
        ge=0,
        description="Quantity consumed (queries, bytes, tokens, etc.)",
    )
    resource_ref: ResourceRef | None = Field(
        default=None,
        description="Optional reference to the resource that generated usage",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured non-secret context (job id, model, etc.)",
    )
    user_id: str | None = None
    workspace_id: str | None = None

    @field_validator("event_type", mode="before")
    @classmethod
    def _coerce_event_type(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("resource_ref", mode="before")
    @classmethod
    def _coerce_ref(cls, value: object) -> object:
        if value is None or isinstance(value, ResourceRef):
            return value
        if isinstance(value, str):
            return ResourceRef.parse(value)
        if isinstance(value, dict):
            return ResourceRef.model_validate(value)
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def _ensure_metadata_dict(cls, value: object) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("metadata must be a dict")
        return value
