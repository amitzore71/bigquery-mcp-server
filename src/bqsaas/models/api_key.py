"""API key domain model — hashed secrets only."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timezone
from typing import ClassVar

from pydantic import ConfigDict, Field, field_validator

from bqsaas.kinds.base import Kind, Resource


def hash_api_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of a raw API key string."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_raw_api_key(*, prefix: str = "bqs", nbytes: int = 32) -> str:
    """Generate a high-entropy raw API key string.

    Format: ``{prefix}_{urlsafe_token}``. The full value is shown once to
    the user; only the hash and short prefix are stored.
    """
    token = secrets.token_urlsafe(nbytes)
    return f"{prefix}_{token}"


class ApiKey(Resource):
    """API key for programmatic access.

    Stores only ``key_prefix`` (for display) and ``key_hash`` (SHA-256).
    The raw key material is never persisted on this model.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    DEFAULT_SCOPES: ClassVar[list[str]] = ["query:read", "chat:write"]

    kind: Kind = Kind.API_KEY

    user_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=200)
    key_prefix: str = Field(
        ...,
        min_length=4,
        max_length=32,
        description="First characters of the raw key for identification",
    )
    key_hash: str = Field(
        ...,
        min_length=64,
        max_length=64,
        description="SHA-256 hex digest of the raw API key",
    )
    scopes: list[str] = Field(default_factory=lambda: list(ApiKey.DEFAULT_SCOPES))
    expires_at: datetime | None = Field(
        default=None,
        description="Optional expiry; None means non-expiring",
    )
    last_used_at: datetime | None = None

    @field_validator("name", "user_id", "key_prefix")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field must not be empty")
        return stripped

    @field_validator("key_hash")
    @classmethod
    def _validate_hash(cls, value: str) -> str:
        digest = value.strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("key_hash must be a 64-character hex SHA-256 digest")
        return digest

    @field_validator("scopes", mode="before")
    @classmethod
    def _normalize_scopes(cls, value: object) -> list[str]:
        if value is None:
            return list(ApiKey.DEFAULT_SCOPES)
        if not isinstance(value, (list, tuple)):
            raise TypeError("scopes must be a list of strings")
        scopes = [str(s).strip() for s in value if str(s).strip()]
        if not scopes:
            raise ValueError("scopes must contain at least one scope")
        return scopes

    @field_validator("expires_at", "last_used_at", mode="before")
    @classmethod
    def _ensure_aware(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    def is_expired(self, *, now: datetime | None = None) -> bool:
        """Return True if the key has an expiry in the past."""
        if self.expires_at is None:
            return False
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return current >= self.expires_at

    def verify(self, raw_key: str) -> bool:
        """Constant-time compare of raw key against stored hash."""
        candidate = hash_api_key(raw_key)
        return secrets.compare_digest(candidate, self.key_hash)

    @classmethod
    def mint(
        cls,
        *,
        tenant_id: str,
        user_id: str,
        name: str,
        scopes: list[str] | None = None,
        expires_at: datetime | None = None,
        prefix: str = "bqs",
    ) -> tuple[ApiKey, str]:
        """Create an ApiKey and return ``(model, raw_key)``.

        The raw key is returned only here and must not be re-stored.
        """
        raw = generate_raw_api_key(prefix=prefix)
        # Use a short identifiable prefix (prefix + first 8 of token body).
        display_prefix = raw[: max(len(prefix) + 9, 12)]
        key = cls(
            tenant_id=tenant_id,
            user_id=user_id,
            name=name,
            key_prefix=display_prefix,
            key_hash=hash_api_key(raw),
            scopes=scopes if scopes is not None else list(cls.DEFAULT_SCOPES),
            expires_at=expires_at,
        )
        return key, raw
