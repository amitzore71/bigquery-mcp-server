"""HTTP API smoke tests with in-memory store."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from bqsaas.api.app import create_app
from bqsaas.storage.memory import MemoryStore


@pytest.fixture
def store() -> MemoryStore:
    return MemoryStore()


@pytest.fixture
def app(store: MemoryStore):
    return create_app(store=store, bootstrap=True)


@pytest.fixture
async def client_and_key(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        async with app.router.lifespan_context(app):
            demo = app.state.demo or {}
            raw_key = demo.get("raw_api_key") or demo.get("api_key")
            assert raw_key, "bootstrap must expose raw_api_key"
            yield ac, raw_key


@pytest.mark.asyncio
async def test_health(client_and_key):
    client, _ = client_and_key
    r = await client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "version" in body


@pytest.mark.asyncio
async def test_ready(client_and_key):
    client, _ = client_and_key
    r = await client.get("/ready")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_me_requires_auth(client_and_key):
    client, _ = client_and_key
    r = await client.get("/v1/me")
    assert r.status_code == 401


@pytest.mark.asyncio
async def test_me_with_demo_key(client_and_key):
    client, raw_key = client_and_key
    r = await client.get(
        "/v1/me",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["user"]["email"] == "demo@example.com"
    assert body["tenant"]["slug"] == "demo"
    assert body["is_admin"] is True


@pytest.mark.asyncio
async def test_kinds(client_and_key):
    client, _ = client_and_key
    r = await client.get("/v1/kinds")
    assert r.status_code == 200
    kinds = r.json()
    assert any(k["kind"] == "tenant" for k in kinds)


@pytest.mark.asyncio
async def test_workspaces_crud(client_and_key):
    client, raw_key = client_and_key
    headers = {"X-API-Key": raw_key}
    r = await client.get("/v1/workspaces", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) >= 1

    r = await client.post(
        "/v1/workspaces",
        headers=headers,
        json={"name": "Analytics", "description": "test"},
    )
    # free plan may only allow 1 workspace
    assert r.status_code in (201, 429), r.text
    if r.status_code == 201:
        ws = r.json()
        assert ws["name"] == "Analytics"
        r = await client.get(f"/v1/workspaces/{ws['id']}", headers=headers)
        assert r.status_code == 200


@pytest.mark.asyncio
async def test_sessions(client_and_key):
    client, raw_key = client_and_key
    headers = {"Authorization": f"Bearer {raw_key}"}
    r = await client.post(
        "/v1/sessions",
        headers=headers,
        json={"title": "Test session"},
    )
    assert r.status_code == 201, r.text
    session = r.json()

    r = await client.get("/v1/sessions", headers=headers)
    assert r.status_code == 200
    assert any(s["id"] == session["id"] for s in r.json())

    r = await client.get(f"/v1/sessions/{session['id']}/messages", headers=headers)
    assert r.status_code == 200
    assert r.json() == []


@pytest.mark.asyncio
async def test_connections(client_and_key):
    client, raw_key = client_and_key
    headers = {"X-API-Key": raw_key}
    r = await client.get("/v1/connections", headers=headers)
    assert r.status_code == 200
    assert len(r.json()) >= 1
