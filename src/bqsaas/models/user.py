"""User domain model."""

from __future__ import annotations

from enum import Enum

from pydantic import ConfigDict, EmailStr, Field, field_validator

from bqsaas.kinds.base import Kind, Resource


class UserRole(str, Enum):
    """Role of a user within a tenant."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class User(Resource):
    """Platform user belonging to exactly one tenant."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: Kind = Kind.USER

    email: str = Field(..., min_length=3, max_length=320)
    name: str = Field(..., min_length=1, max_length=200)
    role: UserRole = UserRole.MEMBER

    @field_validator("email")
    @classmethod
    def _normalize_email(cls, value: str) -> str:
        email = value.strip().lower()
        if "@" not in email or email.startswith("@") or email.endswith("@"):
            raise ValueError("email must be a valid email address")
        local, _, domain = email.partition("@")
        if not local or not domain or "." not in domain:
            raise ValueError("email must be a valid email address")
        return email

    @field_validator("name")
    @classmethod
    def _strip_name(cls, value: str) -> str:
        name = value.strip()
        if not name:
            raise ValueError("name must not be empty")
        return name

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value
