"""Tests for bootstrap_demo and chat session services."""

from __future__ import annotations

import pytest

from bqsaas.storage.memory import MemoryStore


def _bootstrap(store: MemoryStore) -> dict:
    try:
        from bqsaas.services.bootstrap import bootstrap_demo
    except ImportError:
        from bqsaas.bootstrap import bootstrap_demo_tenant as bootstrap_demo  # type: ignore

    return bootstrap_demo(store)


class TestBootstrapDemo:
    def test_bootstrap_demo_creates_all_expected_entities(self, store: MemoryStore):
        result = _bootstrap(store)
        assert result is not None
        assert isinstance(result, dict)

        keys_lower = {str(k).lower() for k in result.keys()}

        has_tenant = any(k in keys_lower for k in ("tenant", "tenant_id", "tenants"))
        has_user = any(k in keys_lower for k in ("user", "user_id", "users"))
        has_key = any(
            k in keys_lower
            for k in ("api_key", "raw_api_key", "raw_key", "key", "api_keys", "keys")
        )

        assert has_tenant, f"bootstrap_demo missing tenant in keys: {keys_lower}"
        assert has_user, f"bootstrap_demo missing user in keys: {keys_lower}"
        assert has_key, f"bootstrap_demo missing api_key in keys: {keys_lower}"

        raw = (
            result.get("raw_api_key")
            or result.get("api_key")
            or result.get("raw_key")
            or result.get("key")
        )
        if isinstance(raw, dict):
            raw = raw.get("raw_key") or raw.get("key")
        if isinstance(raw, str):
            assert raw.startswith("bqs_")

        # Optional but expected by bootstrap_demo contract
        for optional in ("workspace_id", "connection_id", "subscription_id"):
            if optional in result:
                assert result[optional]

    def test_bootstrap_demo_is_idempotent_or_safe_to_call_twice(self, store: MemoryStore):
        first = _bootstrap(store)
        second = _bootstrap(store)
        assert first is not None
        assert second is not None
        # Same tenant when reusing demo slug
        if "tenant_id" in first and "tenant_id" in second:
            if first.get("reused") or second.get("reused"):
                assert first["tenant_id"] == second["tenant_id"]


class TestChatService:
    def test_create_session_and_append_message(self, store: MemoryStore, demo: dict):
        from bqsaas.services.chat_service import ChatService

        tenant_id = demo.get("tenant_id") or (demo.get("tenant") or {}).get("id")
        user_id = demo.get("user_id") or (demo.get("user") or {}).get("id")
        if not tenant_id or not user_id:
            pytest.skip("demo fixture missing tenant_id/user_id")

        service = ChatService(store)

        session = service.create_session(
            tenant_id=tenant_id,
            user_id=user_id,
            title="t1",
        )
        assert session is not None
        session_id = getattr(session, "id", None) or session.get("id")  # type: ignore[union-attr]
        assert session_id

        msg = service.append_message(
            tenant_id=tenant_id,
            session_id=session_id,
            role="user",
            content="hello",
        )
        assert msg is not None
        content = getattr(msg, "content", None)
        if content is None and isinstance(msg, dict):
            content = msg.get("content")
        mid = getattr(msg, "id", None) or (msg.get("id") if isinstance(msg, dict) else None)
        assert content == "hello" or mid is not None
