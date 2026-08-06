"""Workspace domain model."""

from __future__ import annotations

from pydantic import ConfigDict, Field, field_validator

from bqsaas.kinds.base import Kind, Resource


class Workspace(Resource):
    """Logical workspace grouping connections and chat sessions for a tenant."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: Kind = Kind.WORKSPACE

    name: str = Field(..., min_length=1, max_length=200)
    description: str = Field(default="", max_length=2000)

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("name must not be empty")
        return name

    @field_validator("description", mode="before")
    @classmethod
    def _coerce_description(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)
