"""BqSaaS error hierarchy for auth, billing, and service layers."""

from __future__ import annotations

from typing import Any


class BqSaaSError(Exception):
    """Base exception for all BqSaaS domain errors."""

    code: str = "error"
    http_status: int = 500

    def __init__(
        self,
        message: str = "An unexpected error occurred",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error": self.code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:
        return self.message


class NotFoundError(BqSaaSError):
    """Resource was not found."""

    code = "not_found"
    http_status = 404

    def __init__(
        self,
        message: str = "Resource not found",
        *,
        resource: str | None = None,
        resource_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(details or {})
        if resource is not None:
            merged.setdefault("resource", resource)
        if resource_id is not None:
            merged.setdefault("resource_id", resource_id)
        super().__init__(message, details=merged)


class AuthError(BqSaaSError):
    """Authentication failed (missing/invalid credentials)."""

    code = "auth_error"
    http_status = 401

    def __init__(
        self,
        message: str = "Authentication failed",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)


class ForbiddenError(BqSaaSError):
    """Authenticated but not allowed to perform the action."""

    code = "forbidden"
    http_status = 403

    def __init__(
        self,
        message: str = "Forbidden",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, details=details)


class QuotaExceededError(BqSaaSError):
    """Tenant has exceeded a plan quota (e.g. daily query limit)."""

    code = "quota_exceeded"
    http_status = 429

    def __init__(
        self,
        message: str = "Quota exceeded",
        *,
        event_type: str | None = None,
        limit: int | None = None,
        used: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(details or {})
        if event_type is not None:
            merged.setdefault("event_type", event_type)
        if limit is not None:
            merged.setdefault("limit", limit)
        if used is not None:
            merged.setdefault("used", used)
        super().__init__(message, details=merged)


class ValidationError(BqSaaSError):
    """Request or domain validation failed."""

    code = "validation_error"
    http_status = 422

    def __init__(
        self,
        message: str = "Validation failed",
        *,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged = dict(details or {})
        if field is not None:
            merged.setdefault("field", field)
        super().__init__(message, details=merged)
