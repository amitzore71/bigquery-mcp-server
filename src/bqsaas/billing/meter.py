"""Usage metering and daily quota enforcement."""

from __future__ import annotations

from datetime import date, datetime, timezone
from typing import TYPE_CHECKING

from bqsaas.billing.plans import PLANS, Plan, get_plan
from bqsaas.errors import NotFoundError, QuotaExceededError, ValidationError
from bqsaas.kinds.base import Kind
from bqsaas.models import Subscription, UsageEvent, UsageEventType
from bqsaas.models.subscription import PlanTier, SubscriptionStatus

if TYPE_CHECKING:
    from bqsaas.storage import MemoryStore

EVENT_QUERY = "query"
EVENT_CHAT = "chat"
SUPPORTED_EVENTS = frozenset({EVENT_QUERY, EVENT_CHAT})


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _as_date(value: datetime | date | None) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).date()
        return value.date()
    return value


class UsageMeter:
    """Record usage and enforce plan quotas for a tenant.

    Tracks counters on the tenant's :class:`Subscription` (primarily
    ``queries_used_today``). When ``period_start.date() < today``, the
    daily counter is reset before checking/consuming.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def get_subscription(self, tenant_id: str) -> Subscription:
        """Load the active subscription for a tenant."""
        sub = self._find_subscription(tenant_id)
        if sub is None:
            raise NotFoundError(
                "No subscription for tenant",
                resource="subscription",
                resource_id=tenant_id,
            )
        return sub

    def get_plan_for_tenant(self, tenant_id: str) -> Plan:
        sub = self.get_subscription(tenant_id)
        return get_plan(sub.plan)

    def maybe_reset_daily_counters(self, subscription: Subscription) -> Subscription:
        """Reset daily counters if the subscription period day is stale.

        If ``period_start.date() < today`` (UTC), set ``queries_used_today = 0``
        and ``period_start = now``.
        """
        today = _utcnow().date()
        period_date = _as_date(subscription.period_start)

        if period_date is None or period_date < today:
            subscription.reset_daily_usage(period_start=_utcnow())
            subscription = self._store.update(subscription)  # type: ignore[assignment]
            assert isinstance(subscription, Subscription)
        return subscription

    def remaining_queries(self, tenant_id: str) -> int:
        """Queries remaining today."""
        sub = self.maybe_reset_daily_counters(self.get_subscription(tenant_id))
        return sub.remaining_queries()

    def check_quota(
        self,
        tenant_id: str,
        event_type: str = EVENT_QUERY,
        units: int = 1,
    ) -> None:
        """Raise :class:`QuotaExceededError` if consuming ``units`` would exceed limit.

        Does not consume.
        """
        if units < 0:
            raise ValidationError("units must be >= 0", field="units")
        if event_type not in SUPPORTED_EVENTS:
            raise ValidationError(
                f"Unsupported event_type '{event_type}'",
                field="event_type",
                details={"supported": sorted(SUPPORTED_EVENTS)},
            )
        if units == 0:
            return

        sub = self.maybe_reset_daily_counters(self.get_subscription(tenant_id))

        if sub.subscription_status in (
            SubscriptionStatus.CANCELED,
            SubscriptionStatus.PAST_DUE,
        ):
            raise QuotaExceededError(
                f"Subscription is {sub.subscription_status.value}",
                event_type=event_type,
                limit=sub.queries_limit,
                used=sub.queries_used_today,
                details={"subscription_status": sub.subscription_status.value},
            )

        if event_type == EVENT_QUERY:
            plan = get_plan(sub.plan)
            if plan.is_unlimited_queries:
                return
            if not sub.can_query(units):
                raise QuotaExceededError(
                    f"Daily query limit exceeded for plan '{sub.plan.value}'",
                    event_type=event_type,
                    limit=sub.queries_limit,
                    used=sub.queries_used_today,
                    details={"units_requested": units, "plan_id": sub.plan.value},
                )

    def record_usage(
        self,
        tenant_id: str,
        event_type: str = EVENT_QUERY,
        units: int = 1,
        *,
        user_id: str | None = None,
        workspace_id: str | None = None,
        metadata: dict | None = None,
    ) -> Subscription:
        """Increment usage counters without a prior quota check.

        Prefer :meth:`check_and_consume` for atomic check+consume.
        """
        if units < 0:
            raise ValidationError("units must be >= 0", field="units")
        if event_type not in SUPPORTED_EVENTS:
            raise ValidationError(
                f"Unsupported event_type '{event_type}'",
                field="event_type",
                details={"supported": sorted(SUPPORTED_EVENTS)},
            )

        sub = self.maybe_reset_daily_counters(self.get_subscription(tenant_id))
        if event_type == EVENT_QUERY and units:
            sub.queries_used_today = int(sub.queries_used_today or 0) + int(units)
            sub.touch()
            sub = self._store.update(sub)  # type: ignore[assignment]
            assert isinstance(sub, Subscription)

        # Persist a usage event for audit/metering
        try:
            usage_type = (
                UsageEventType.QUERY
                if event_type == EVENT_QUERY
                else UsageEventType.CHAT_MESSAGE
            )
            event = UsageEvent(
                tenant_id=tenant_id,
                event_type=usage_type,
                units=float(units),
                user_id=user_id,
                workspace_id=workspace_id,
                metadata=dict(metadata or {}),
            )
            self._store.create(event)
        except Exception:
            # Metering events must not break the primary consume path.
            pass

        return sub

    def check_and_consume(
        self,
        tenant_id: str,
        event_type: str = EVENT_QUERY,
        units: int = 1,
        *,
        user_id: str | None = None,
        workspace_id: str | None = None,
        metadata: dict | None = None,
    ) -> Subscription:
        """Check quota then consume units. Raises QuotaExceededError if over limit."""
        self.check_quota(tenant_id, event_type=event_type, units=units)
        return self.record_usage(
            tenant_id,
            event_type=event_type,
            units=units,
            user_id=user_id,
            workspace_id=workspace_id,
            metadata=metadata,
        )

    def check_resource_limit(
        self,
        tenant_id: str,
        resource: str,
        current_count: int,
    ) -> None:
        """Check soft resource caps (workspaces, connections, users).

        Raises QuotaExceededError when ``current_count >= plan max``.
        """
        plan = self.get_plan_for_tenant(tenant_id)
        limits = {
            "workspaces": plan.max_workspaces,
            "connections": plan.max_connections,
            "users": plan.max_users,
            "chat_sessions": plan.max_chat_sessions,
        }
        if resource not in limits:
            raise ValidationError(
                f"Unknown resource '{resource}'",
                field="resource",
                details={"known": sorted(limits.keys())},
            )
        limit = limits[resource]
        if current_count >= limit:
            raise QuotaExceededError(
                f"Plan limit reached for {resource}",
                event_type=resource,
                limit=limit,
                used=current_count,
                details={"plan_id": plan.id},
            )

    def usage_snapshot(self, tenant_id: str) -> dict:
        """Return a simple usage summary for dashboards/APIs."""
        sub = self.maybe_reset_daily_counters(self.get_subscription(tenant_id))
        plan = get_plan(sub.plan)
        used = int(sub.queries_used_today or 0)
        return {
            "tenant_id": tenant_id,
            "plan_id": plan.id,
            "plan_name": plan.name,
            "queries_used_today": used,
            "daily_query_limit": sub.queries_limit,
            "queries_remaining": sub.remaining_queries(),
            "period_start": sub.period_start,
            "subscription_status": sub.subscription_status.value,
            "max_workspaces": plan.max_workspaces,
            "max_connections": plan.max_connections,
            "max_users": plan.max_users,
        }

    def _find_subscription(self, tenant_id: str) -> Subscription | None:
        found = self._store.get_by_field(
            Kind.SUBSCRIPTION,
            "tenant_id",
            tenant_id,
            tenant_id=tenant_id,
        )
        if found is not None and isinstance(found, Subscription):
            return found

        for item in self._store.list(Kind.SUBSCRIPTION, tenant_id=tenant_id, limit=100):
            if isinstance(item, Subscription):
                return item
        return None
