"""Tests for API key creation, hashing, and authentication."""

from __future__ import annotations

import pytest

from bqsaas.auth.security import generate_api_key, hash_api_key, verify_api_key
from bqsaas.kinds.base import Kind, generate_resource_id
from bqsaas.storage.memory import MemoryStore


def _seed_tenant_and_user(store: MemoryStore):
    """Create minimal tenant + user required by create_api_key."""
    from bqsaas.models.tenant import Tenant
    from bqsaas.models.user import User

    tenant = Tenant(name="Auth Test Org", slug=f"auth-test-{generate_resource_id(Kind.TENANT)[-8:]}")
    if hasattr(store, "create"):
        tenant = store.create(tenant)
    elif hasattr(store, "save"):
        store.save(tenant)

    user = User(
        tenant_id=tenant.id,
        email=f"owner-{tenant.id[-6:]}@example.com",
        name="Owner",
    )
    if hasattr(store, "create"):
        user = store.create(user)
    elif hasattr(store, "save"):
        store.save(user)

    return tenant, user


def _create_api_key(store: MemoryStore, tenant_id: str, user_id: str):
    from bqsaas.auth.dependencies import create_api_key

    return create_api_key(
        store,
        tenant_id=tenant_id,
        user_id=user_id,
        name="test-key",
        scopes=["*"],
        env="test",
    )


def _authenticate(store: MemoryStore, raw_key: str):
    from bqsaas.auth.dependencies import authenticate_api_key

    return authenticate_api_key(store, raw_key)


class TestCreateApiKey:
    def test_create_api_key_returns_raw_key_starting_with_bqs_(self, store: MemoryStore):
        tenant, user = _seed_tenant_and_user(store)
        api_key, raw = _create_api_key(store, tenant.id, user.id)
        assert isinstance(raw, str)
        assert raw.startswith("bqs_"), f"Expected raw key to start with 'bqs_', got {raw!r}"
        assert api_key is not None

    def test_generate_api_key_format(self):
        raw = generate_api_key(env="live")
        assert raw.startswith("bqs_live_")


class TestAuthenticate:
    def test_authenticate_succeeds_with_raw_key(self, store: MemoryStore):
        tenant, user = _seed_tenant_and_user(store)
        _api_key, raw = _create_api_key(store, tenant.id, user.id)

        ctx = _authenticate(store, raw)
        assert ctx is not None
        assert ctx.tenant_id == tenant.id
        assert ctx.user_id == user.id

    def test_authenticate_fails_with_wrong_key(self, store: MemoryStore):
        tenant, user = _seed_tenant_and_user(store)
        _create_api_key(store, tenant.id, user.id)

        wrong = "bqs_test_ffffffffffffffffffffffffffffffff"
        with pytest.raises(Exception) as exc_info:
            _authenticate(store, wrong)
        # AuthError or similar
        assert exc_info.value is not None


class TestKeyHash:
    def test_hash_is_not_equal_to_raw_key(self, store: MemoryStore):
        tenant, user = _seed_tenant_and_user(store)
        api_key, raw = _create_api_key(store, tenant.id, user.id)

        assert api_key.key_hash is not None
        assert api_key.key_hash != raw
        assert not api_key.key_hash.startswith("bqs_")
        # hash_api_key agrees
        assert hash_api_key(raw) == api_key.key_hash
        assert verify_api_key(raw, api_key.key_hash) is True
        assert verify_api_key("bqs_test_wrong", api_key.key_hash) is False
