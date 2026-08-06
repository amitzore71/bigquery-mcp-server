"""API key hashing/generation and optional signed token helpers."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from typing import Any, Literal

from bqsaas.errors import AuthError, ValidationError

# Key format: bqs_{env}_{random32 hex}  e.g. bqs_live_a1b2c3...
API_KEY_PREFIX = "bqs"
API_KEY_RANDOM_BYTES = 16  # 32 hex chars
VALID_ENVS = frozenset({"live", "test", "dev"})

DEFAULT_TOKEN_TTL_SECONDS = 3600


def generate_api_key(env: Literal["live", "test", "dev"] = "live") -> str:
    """Generate a raw API key once. Format: ``bqs_{env}_{random32}``.

    The full raw key is returned only at creation time and must not be stored.
    """
    if env not in VALID_ENVS:
        raise ValidationError(
            f"Invalid API key env '{env}'",
            field="env",
            details={"allowed": sorted(VALID_ENVS)},
        )
    random_part = secrets.token_hex(API_KEY_RANDOM_BYTES)
    return f"{API_KEY_PREFIX}_{env}_{random_part}"


def hash_api_key(raw_key: str) -> str:
    """Return the SHA-256 hex digest of the full raw API key."""
    if not raw_key or not isinstance(raw_key, str):
        raise ValidationError("API key must be a non-empty string", field="api_key")
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def key_prefix(raw_key: str, length: int = 8) -> str:
    """First ``length`` characters of the raw key (for display / lookup hints)."""
    if not raw_key:
        raise ValidationError("API key must be a non-empty string", field="api_key")
    return raw_key[:length]


def verify_api_key(raw_key: str, key_hash: str) -> bool:
    """Constant-time comparison of raw key against a stored SHA-256 hash."""
    if not raw_key or not key_hash:
        return False
    computed = hash_api_key(raw_key)
    return hmac.compare_digest(computed, key_hash.lower())


def parse_api_key_format(raw_key: str) -> tuple[str, str]:
    """Validate and parse ``bqs_{env}_{secret}`` into (env, secret).

    Raises:
        AuthError: if the key format is invalid.
    """
    if not raw_key or not isinstance(raw_key, str):
        raise AuthError("Missing or invalid API key")

    parts = raw_key.split("_", 2)
    if len(parts) != 3 or parts[0] != API_KEY_PREFIX:
        raise AuthError("Invalid API key format")
    env, secret = parts[1], parts[2]
    if env not in VALID_ENVS:
        raise AuthError("Invalid API key environment segment")
    if len(secret) < 16:
        raise AuthError("Invalid API key format")
    return env, secret


# ---------------------------------------------------------------------------
# Optional JWT-like signed tokens (HMAC-SHA256, no external dependency)
# ---------------------------------------------------------------------------


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def sign_token(
    payload: dict[str, Any],
    secret: str,
    *,
    ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS,
) -> str:
    """Create a compact signed token: ``base64url(header).base64url(payload).sig``.

    Payload is JSON-serialized. Adds ``iat`` and ``exp`` claims automatically.
    """
    if not secret:
        raise ValidationError("Token secret is required", field="secret")
    if ttl_seconds <= 0:
        raise ValidationError("ttl_seconds must be positive", field="ttl_seconds")

    now = int(time.time())
    body = dict(payload)
    body.setdefault("iat", now)
    body.setdefault("exp", now + ttl_seconds)

    header = {"alg": "HS256", "typ": "BQS"}
    header_b64 = _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8"))
    payload_b64 = _b64url_encode(json.dumps(body, separators=(",", ":")).encode("utf-8"))
    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    sig = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    return f"{header_b64}.{payload_b64}.{_b64url_encode(sig)}"


def verify_token(token: str, secret: str) -> dict[str, Any]:
    """Verify a token from :func:`sign_token` and return its payload.

    Raises:
        AuthError: if signature is invalid, token is malformed, or expired.
    """
    if not token or not secret:
        raise AuthError("Invalid token")

    try:
        header_b64, payload_b64, sig_b64 = token.split(".")
    except ValueError as exc:
        raise AuthError("Malformed token") from exc

    signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
    expected = hmac.new(secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
    try:
        actual = _b64url_decode(sig_b64)
    except Exception as exc:
        raise AuthError("Malformed token signature") from exc

    if not hmac.compare_digest(expected, actual):
        raise AuthError("Invalid token signature")

    try:
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise AuthError("Malformed token payload") from exc

    if not isinstance(payload, dict):
        raise AuthError("Malformed token payload")

    exp = payload.get("exp")
    if exp is not None and int(time.time()) >= int(exp):
        raise AuthError("Token expired")

    return payload
