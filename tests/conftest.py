"""Shared pytest fixtures for the bqsaas test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _make_store():
    from bqsaas.storage.memory import MemoryStore

    return MemoryStore()


def _bootstrap(store):
    try:
        from bqsaas.services.bootstrap import bootstrap_demo
    except ImportError:
        from bqsaas.bootstrap import bootstrap_demo_tenant as bootstrap_demo  # type: ignore

    return bootstrap_demo(store)


def _create_app(store):
    from bqsaas.api.app import create_app

    try:
        return create_app(store=store)
    except TypeError:
        app = create_app()
        if hasattr(app, "state"):
            app.state.store = store
        return app


@pytest.fixture
def store():
    """Fresh in-memory store for each test."""
    return _make_store()


@pytest.fixture
def demo(store) -> dict:
    """Bootstrap demo tenant/user/api-key into the store; return result dict."""
    result = _bootstrap(store)
    assert isinstance(result, dict)
    return result


@pytest.fixture
def client(store, demo: dict):
    """
    FastAPI TestClient wired to an isolated MemoryStore with demo data.

    If create_app accepts a store argument, pass it; otherwise attach
    store onto app.state after creation.
    """
    app = _create_app(store)

    app_store = getattr(getattr(app, "state", None), "store", None)
    if app_store is not None and app_store is not store:
        # App built its own store — re-bootstrap so demo key exists there.
        demo_on_app = _bootstrap(app_store)
        bound_store = app_store
        bound_demo = demo_on_app
    else:
        if hasattr(app, "state"):
            app.state.store = store
        bound_store = store
        bound_demo = demo

    with TestClient(app) as test_client:
        test_client.store = bound_store  # type: ignore[attr-defined]
        test_client.demo = bound_demo  # type: ignore[attr-defined]
        yield test_client
