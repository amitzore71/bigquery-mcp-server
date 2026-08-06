"""FastAPI auth dependencies."""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from bqsaas.auth.dependencies import (
    AuthContext,
    authenticate_api_key,
    extract_raw_key_from_request,
)
from bqsaas.errors import AuthError, ForbiddenError
from bqsaas.storage.memory import MemoryStore


def get_store(request: Request) -> MemoryStore:
    store: Optional[MemoryStore] = getattr(request.app.state, "store", None)
    if store is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Store not initialized",
        )
    return store


def extract_api_key(request: Request) -> Optional[str]:
    """Extract API key from Authorization Bearer or X-API-Key header."""
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    x_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    try:
        return extract_raw_key_from_request(auth, x_key)
    except AuthError:
        if x_key:
            return x_key.strip()
        if auth:
            parts = auth.split(None, 1)
            if len(parts) == 2 and parts[0].lower() == "bearer":
                return parts[1].strip()
            if auth.strip().startswith("bqs_"):
                return auth.strip()
        return None


async def get_auth_context(request: Request) -> AuthContext:
    """
    Resolve the current caller.

    Accepts:
      - ``Authorization: Bearer bqs_...``
      - ``X-API-Key: bqs_...``
    """
    raw = extract_api_key(request)
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Use Authorization: Bearer bqs_... or X-API-Key.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    store = get_store(request)
    try:
        return authenticate_api_key(store, raw)
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


async def require_admin(
    ctx: AuthContext = Depends(get_auth_context),
) -> AuthContext:
    if not ctx.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return ctx


def domain_error_to_http(exc: Exception) -> HTTPException:
    """Map domain errors to HTTPException."""
    if isinstance(exc, AuthError):
        return HTTPException(status_code=401, detail=exc.message)
    if isinstance(exc, ForbiddenError):
        return HTTPException(status_code=403, detail=exc.message)
    from bqsaas.errors import NotFoundError, QuotaExceededError, ValidationError

    if isinstance(exc, NotFoundError):
        return HTTPException(status_code=404, detail=exc.message)
    if isinstance(exc, QuotaExceededError):
        return HTTPException(status_code=429, detail=exc.message)
    if isinstance(exc, ValidationError):
        return HTTPException(status_code=422, detail=exc.message)
    raise exc
