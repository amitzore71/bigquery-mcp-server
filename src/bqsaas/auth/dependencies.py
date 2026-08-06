"""Resolve tenant/user auth context from API keys and enforce scopes."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Sequence

from bqsaas.auth.security import (
    generate_api_key,
    hash_api_key,
    key_prefix,
    parse_api_key_format,
    verify_api_key,
)
from bqsaas.errors import AuthError, ForbiddenError, NotFoundError, ValidationError
from bqsaas.kinds import Kind, generate_id
from bqsaas.kinds.base import ResourceStatus
from bqsaas.models import ApiKey, Tenant, User

if TYPE_CHECKING:
    from bqsaas.storage import MemoryStore

# Known scopes. ``*`` grants everything.
SCOPE_CHAT_READ = "chat:read"
SCOPE_CHAT_WRITE = "chat:write"
SCOPE_QUERY_EXECUTE = "query:execute"
SCOPE_ADMIN_READ = "admin:read"
SCOPE_ADMIN_WRITE = "admin:write"
SCOPE_ALL = "*"

ALL_SCOPES: frozenset[str] = frozenset(
    {
        SCOPE_CHAT_READ,
        SCOPE_CHAT_WRITE,
        SCOPE_QUERY_EXECUTE,
        SCOPE_ADMIN_READ,
        SCOPE_ADMIN_WRITE,
        SCOPE_ALL,
    }
)

_API_KEY_HEADER_SCHEMES = ("Bearer ", "ApiKey ", "apikey ")
API_KEY_PREFIX_FRAGMENT = "bqs"


@dataclass(slots=True)
class AuthContext:
    """Resolved identity for a request authenticated via API key."""

    tenant: Tenant
    user: User
    api_key: ApiKey
    scopes: list[str] = field(default_factory=list)

    @property
    def tenant_id(self) -> str:
        return self.tenant.id

    @property
    def user_id(self) -> str:
        return self.user.id

    @property
    def is_admin(self) -> bool:
        if SCOPE_ALL in self.scopes or SCOPE_ADMIN_WRITE in self.scopes:
            return True
        role = getattr(self.user, "role", None)
        role_val = role.value if hasattr(role, "value") else str(role or "")
        return role_val in ("owner", "admin")

    def has_scope(self, required: str) -> bool:
        if SCOPE_ALL in self.scopes:
            return True
        return required in self.scopes

    def require_scopes(self, *required: str) -> None:
        """Raise ForbiddenError if any required scope is missing."""
        require_scopes(self, required)


def require_scopes(ctx: AuthContext, required: Sequence[str]) -> None:
    """Ensure ``ctx`` has every scope in ``required`` (or ``*``)."""
    if SCOPE_ALL in ctx.scopes:
        return
    missing = [s for s in required if s not in ctx.scopes]
    if missing:
        raise ForbiddenError(
            "Insufficient scopes",
            details={
                "required": list(required),
                "missing": missing,
                "have": list(ctx.scopes),
            },
        )


def _validate_scopes(scopes: Sequence[str]) -> list[str]:
    if not scopes:
        raise ValidationError("At least one scope is required", field="scopes")
    normalized: list[str] = []
    for scope in scopes:
        if scope not in ALL_SCOPES:
            raise ValidationError(
                f"Unknown scope '{scope}'",
                field="scopes",
                details={"allowed": sorted(ALL_SCOPES)},
            )
        if scope not in normalized:
            normalized.append(scope)
    return normalized


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def create_api_key(
    store: MemoryStore,
    tenant_id: str,
    user_id: str,
    name: str,
    scopes: Sequence[str],
    *,
    env: str = "live",
) -> tuple[ApiKey, str]:
    """Create an API key record and return ``(ApiKey model, raw_key_once)``."""
    if not name or not name.strip():
        raise ValidationError("API key name is required", field="name")

    tenant = store.get(Kind.TENANT, tenant_id)
    if tenant is None:
        raise NotFoundError(
            "Tenant not found", resource="tenant", resource_id=tenant_id
        )

    user = store.get(Kind.USER, user_id)
    if user is None:
        raise NotFoundError("User not found", resource="user", resource_id=user_id)
    if getattr(user, "tenant_id", None) != tenant_id:
        raise ForbiddenError("User does not belong to tenant")

    normalized_scopes = _validate_scopes(scopes)
    raw_key = generate_api_key(env=env)  # type: ignore[arg-type]

    api_key = ApiKey(
        id=generate_id(Kind.API_KEY),
        tenant_id=tenant_id,
        user_id=user_id,
        name=name.strip(),
        key_hash=hash_api_key(raw_key),
        key_prefix=key_prefix(raw_key, 12),
        scopes=list(normalized_scopes),
    )
    store.save(api_key)
    return api_key, raw_key


def authenticate_api_key(store: MemoryStore, raw_key: str) -> AuthContext:
    """Authenticate a raw API key and return an :class:`AuthContext`."""
    parse_api_key_format(raw_key)
    digest = hash_api_key(raw_key)

    api_key: ApiKey | None = None
    finder = getattr(store, "find_api_key_by_hash", None)
    if callable(finder):
        api_key = finder(digest)  # type: ignore[assignment]
    if api_key is None:
        api_key = store.find_one(Kind.API_KEY, key_hash=digest)  # type: ignore[assignment]

    if api_key is None:
        raise AuthError("Invalid API key")

    if getattr(api_key, "status", ResourceStatus.ACTIVE) is ResourceStatus.DELETED:
        raise AuthError("API key has been revoked")
    if getattr(api_key, "is_expired", lambda: False)():
        raise AuthError("API key has expired")

    if not verify_api_key(raw_key, api_key.key_hash):
        raise AuthError("Invalid API key")

    tenant = store.get(Kind.TENANT, api_key.tenant_id)
    if tenant is None:
        raise AuthError("API key tenant no longer exists")

    user = store.get(Kind.USER, api_key.user_id)
    if user is None:
        raise AuthError("API key user no longer exists")

    if hasattr(api_key, "last_used_at"):
        api_key.last_used_at = _utcnow()
        store.save(api_key)

    scopes = list(getattr(api_key, "scopes", []) or [])
    return AuthContext(tenant=tenant, user=user, api_key=api_key, scopes=scopes)  # type: ignore[arg-type]


def extract_raw_key_from_header(authorization: str | None) -> str:
    """Extract raw API key from an Authorization header value."""
    if not authorization or not authorization.strip():
        raise AuthError("Missing Authorization header")

    value = authorization.strip()
    for scheme in _API_KEY_HEADER_SCHEMES:
        if value.lower().startswith(scheme.lower()):
            raw = value[len(scheme) :].strip()
            if not raw:
                raise AuthError("Empty API key in Authorization header")
            return raw

    if value.startswith(f"{API_KEY_PREFIX_FRAGMENT}_"):
        return value

    raise AuthError("Unrecognized Authorization header format")


def resolve_auth_from_header(
    store: MemoryStore, authorization: str | None
) -> AuthContext:
    """High-level helper: Authorization header → AuthContext."""
    raw_key = extract_raw_key_from_header(authorization)
    return authenticate_api_key(store, raw_key)


def extract_raw_key_from_request(
    authorization: str | None, x_api_key: str | None
) -> str:
    """Extract key from Bearer or X-API-Key."""
    if x_api_key and x_api_key.strip():
        return x_api_key.strip()
    return extract_raw_key_from_header(authorization)
