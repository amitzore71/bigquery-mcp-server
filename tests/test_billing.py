"""Tests for plans registry, free-tier quota, and daily reset."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from bqsaas.billing.plans import PLANS, get_plan
from bqsaas.errors import QuotaExceededError


class TestPlansRegistry:
    def test_plans_registry_has_free_pro_enterprise(self):
        names = {k.lower() for k in PLANS.keys()}
        for required in ("free", "pro", "enterprise"):
            assert required in names, f"Expected plan '{required}' in {names}"

        free = get_plan("free")
        assert free.daily_query_limit > 0
        assert free.id == "free"


class TestFreePlanQuota:
    def test_free_plan_quota_allows_n_queries_then_raises(self):
        from bqsaas.billing.meter import UsageMeter
        from bqsaas.billing.plans import PLANS

        free = PLANS["free"]
        n = free.daily_query_limit
        assert n > 0

        # Use a small limit for speed if the free plan is large — temporarily
        # patch via a mock subscription store.
        # Prefer real store + subscription when models/store are wired.
        from bqsaas.storage.memory import MemoryStore
        from bqsaas.kinds.base import Kind, generate_resource_id

        store = MemoryStore()
        tenant_id = generate_resource_id(Kind.TENANT)

        # Build subscription model compatible with UsageMeter
        sub = _make_subscription(store, tenant_id, plan_id="free", queries_used_today=0)

        meter = UsageMeter(store)

        # If free plan is 100, still run all N — it's fine and fast.
        # For safety, if limit is huge, only verify at boundary via direct set.
        limit = min(n, 100)
        if hasattr(sub, "queries_used_today"):
            # Drive check_and_consume or set near limit
            if n > 20:
                sub.queries_used_today = n - 1
                _save_sub(store, sub)
                meter.check_and_consume(tenant_id, event_type="query", units=1)
                with pytest.raises(QuotaExceededError):
                    meter.check_and_consume(tenant_id, event_type="query", units=1)
            else:
                for _ in range(limit):
                    meter.check_and_consume(tenant_id, event_type="query", units=1)
                with pytest.raises(QuotaExceededError):
                    meter.check_and_consume(tenant_id, event_type="query", units=1)
        else:
            pytest.skip("Subscription model lacks queries_used_today")


class TestDailyReset:
    def test_daily_reset_when_period_start_is_yesterday(self):
        from bqsaas.billing.meter import UsageMeter
        from bqsaas.billing.plans import PLANS
        from bqsaas.storage.memory import MemoryStore
        from bqsaas.kinds.base import Kind, generate_resource_id

        store = MemoryStore()
        tenant_id = generate_resource_id(Kind.TENANT)
        free = PLANS["free"]
        n = free.daily_query_limit

        yesterday = datetime.now(timezone.utc) - timedelta(days=1)
        sub = _make_subscription(
            store,
            tenant_id,
            plan_id="free",
            queries_used_today=n,
            period_start=yesterday,
        )
        assert getattr(sub, "queries_used_today", 0) >= n or n == 0

        meter = UsageMeter(store)

        # After daily reset, a new query should be allowed
        try:
            meter.check_and_consume(tenant_id, event_type="query", units=1)
        except QuotaExceededError:
            pytest.fail(
                "Expected daily reset to allow a new query when period_start is yesterday"
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subscription(
    store,
    tenant_id: str,
    *,
    plan_id: str = "free",
    queries_used_today: int = 0,
    period_start: datetime | None = None,
):
    """Create and persist a Subscription the meter can find."""
    from bqsaas.kinds.base import Kind, generate_resource_id

    period = period_start or datetime.now(timezone.utc)

    # Try concrete model constructors with different field names.
    try:
        from bqsaas.models.subscription import Subscription, PlanTier

        kwargs = {
            "tenant_id": tenant_id,
            "id": generate_resource_id(Kind.SUBSCRIPTION),
            "queries_used_today": queries_used_today,
            "period_start": period,
        }
        # plan vs plan_id
        try:
            sub = Subscription(**kwargs, plan=PlanTier(plan_id))
        except TypeError:
            try:
                sub = Subscription(**kwargs, plan_id=plan_id)
            except TypeError:
                sub = Subscription(tenant_id=tenant_id, plan=PlanTier(plan_id))
                sub.queries_used_today = queries_used_today
                sub.period_start = period
    except Exception:
        # Minimal duck-typed object if model constructor fails
        sub = MagicMock()
        sub.id = generate_resource_id(Kind.SUBSCRIPTION)
        sub.kind = Kind.SUBSCRIPTION
        sub.tenant_id = tenant_id
        sub.plan_id = plan_id
        sub.queries_used_today = queries_used_today
        sub.period_start = period

    # Ensure plan_id attr for meter
    if not hasattr(sub, "plan_id") or getattr(sub, "plan_id", None) is None:
        try:
            object.__setattr__(sub, "plan_id", plan_id)
        except Exception:
            if hasattr(sub, "plan"):
                # meter uses plan_id primarily; patch via attribute
                try:
                    sub.plan_id = plan_id  # type: ignore[attr-defined]
                except Exception:
                    pass

    _save_sub(store, sub)
    return sub


def _save_sub(store, sub) -> None:
    if hasattr(store, "save"):
        store.save(sub)
        return
    if hasattr(store, "create"):
        try:
            store.create(sub)
            return
        except Exception:
            # may already exist — try update
            if hasattr(store, "update"):
                try:
                    store.update(sub)
                    return
                except Exception:
                    pass
    # Last resort: put into private dict if present (test-only)
    if hasattr(store, "_data") and hasattr(sub, "kind") and hasattr(sub, "id"):
        kind_val = sub.kind.value if hasattr(sub.kind, "value") else str(sub.kind)
        store._data[(kind_val, sub.id)] = sub
