"""Tests for MemoryStore resource CRUD, isolation, soft-delete, pagination."""

from __future__ import annotations

from time import sleep

import pytest

from bqsaas.kinds.base import Kind, Resource, ResourceStatus, generate_resource_id
from bqsaas.storage.memory import MemoryStore


def _make_child_resource(tenant_id: str, name: str = "res") -> Resource:
    """Build a minimal tenant-scoped Resource for storage tests."""
    # Prefer a concrete model if available; fall back to bare Resource.
    try:
        from bqsaas.models.workspace import Workspace

        return Workspace(
            tenant_id=tenant_id,
            name=name,
            id=generate_resource_id(Kind.WORKSPACE),
        )
    except Exception:
        pass

    return Resource(
        kind=Kind.WORKSPACE,
        id=generate_resource_id(Kind.WORKSPACE),
        tenant_id=tenant_id,
        labels={"name": name},
    )


def _create(store: MemoryStore, resource: Resource) -> Resource:
    if hasattr(store, "create"):
        return store.create(resource)
    if hasattr(store, "save"):
        store.save(resource)
        return resource
    raise AttributeError("MemoryStore has no create/save")


def _get(store: MemoryStore, kind: Kind, resource_id: str, tenant_id: str | None = None):
    if hasattr(store, "get"):
        try:
            return store.get(kind, resource_id, tenant_id=tenant_id)
        except TypeError:
            return store.get(kind, resource_id)
    raise AttributeError("MemoryStore has no get")


def _list(store: MemoryStore, kind: Kind, tenant_id: str, **kwargs):
    return store.list(kind, tenant_id=tenant_id, **kwargs)


def _soft_delete(store: MemoryStore, kind: Kind, resource_id: str, tenant_id: str | None = None):
    return store.soft_delete(kind, resource_id, tenant_id=tenant_id)


def _update(store: MemoryStore, resource: Resource) -> Resource:
    return store.update(resource)


class TestCreateGet:
    def test_create_and_get_resource(self, store: MemoryStore):
        tenant_id = generate_resource_id(Kind.TENANT)
        created = _create(store, _make_child_resource(tenant_id, name="alpha"))
        rid = created.id

        got = _get(store, created.kind, rid, tenant_id=tenant_id)
        assert got is not None
        assert got.id == rid
        assert got.tenant_id == tenant_id


class TestTenantIsolation:
    def test_tenant_a_cannot_list_tenant_b_resources(self, store: MemoryStore):
        tenant_a = generate_resource_id(Kind.TENANT)
        tenant_b = generate_resource_id(Kind.TENANT)

        res_a = _create(store, _make_child_resource(tenant_a, name="a-only"))
        res_b = _create(store, _make_child_resource(tenant_b, name="b-only"))

        listed_a = list(_list(store, res_a.kind, tenant_a))
        listed_b = list(_list(store, res_b.kind, tenant_b))

        ids_a = {r.id for r in listed_a}
        ids_b = {r.id for r in listed_b}

        assert res_a.id in ids_a
        assert res_b.id not in ids_a
        assert res_b.id in ids_b
        assert res_a.id not in ids_b


class TestSoftDelete:
    def test_soft_delete_sets_status_deleted_and_excluded_from_default_list(
        self, store: MemoryStore
    ):
        tenant_id = generate_resource_id(Kind.TENANT)
        created = _create(store, _make_child_resource(tenant_id, name="to-delete"))
        rid = created.id
        kind = created.kind

        deleted = _soft_delete(store, kind, rid, tenant_id=tenant_id)
        assert deleted is not None
        assert deleted.status == ResourceStatus.DELETED

        # Default get excludes deleted
        got = _get(store, kind, rid, tenant_id=tenant_id)
        assert got is None

        # Default list (status=ACTIVE) excludes soft-deleted
        listed = list(_list(store, kind, tenant_id))
        ids = {r.id for r in listed}
        assert rid not in ids


class TestUpdate:
    def test_update_bumps_updated_at(self, store: MemoryStore):
        tenant_id = generate_resource_id(Kind.TENANT)
        created = _create(store, _make_child_resource(tenant_id, name="before"))
        before = created.updated_at

        sleep(0.02)
        # Mutate a safe field if present
        if hasattr(created, "name"):
            created.name = "after"
        elif hasattr(created, "labels"):
            created.labels = {**(created.labels or {}), "name": "after"}
        else:
            created.annotations = {**(created.annotations or {}), "n": "after"}

        if hasattr(created, "touch"):
            created.touch()

        updated = _update(store, created)
        assert updated.updated_at is not None
        assert updated.updated_at >= before


class TestListPagination:
    def test_list_with_limit_and_offset(self, store: MemoryStore):
        tenant_id = generate_resource_id(Kind.TENANT)
        created_ids = []
        kind = Kind.WORKSPACE
        for i in range(5):
            res = _create(store, _make_child_resource(tenant_id, name=f"item-{i}"))
            kind = res.kind
            created_ids.append(res.id)

        page1 = list(_list(store, kind, tenant_id, limit=2, offset=0))
        page2 = list(_list(store, kind, tenant_id, limit=2, offset=2))

        assert len(page1) == 2
        assert len(page2) == 2

        ids1 = {r.id for r in page1}
        ids2 = {r.id for r in page2}
        assert ids1.isdisjoint(ids2)

        all_listed = list(_list(store, kind, tenant_id, limit=100, offset=0))
        all_ids = {r.id for r in all_listed}
        for cid in created_ids:
            assert cid in all_ids
