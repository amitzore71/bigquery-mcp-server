"""Subscription domain model — plan, status, and daily query quotas."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import ConfigDict, Field, field_validator

from bqsaas.kinds.base import Kind, Resource, _utc_now


class PlanTier(str, Enum):
    """Supported commercial plan tiers."""

    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Billing lifecycle status for a subscription."""

    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    TRIALING = "trialing"


# Default daily query limits by plan (overridable via Settings).
DEFAULT_PLAN_LIMITS: dict[PlanTier, int] = {
    PlanTier.FREE: 100,
    PlanTier.PRO: 10_000,
    PlanTier.ENTERPRISE: 1_000_000,
}


class Subscription(Resource):
    """Tenant subscription binding plan tier to query quotas."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: Kind = Kind.SUBSCRIPTION

    plan: PlanTier = PlanTier.FREE
    subscription_status: SubscriptionStatus = Field(
        default=SubscriptionStatus.ACTIVE,
        description="Billing status (distinct from resource lifecycle status)",
    )
    queries_used_today: int = Field(default=0, ge=0)
    queries_limit: int = Field(default=100, ge=0)
    period_start: datetime = Field(default_factory=_utc_now)

    @field_validator("plan", mode="before")
    @classmethod
    def _coerce_plan(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("subscription_status", mode="before")
    @classmethod
    def _coerce_sub_status(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("period_start", mode="before")
    @classmethod
    def _ensure_aware(cls, value: object) -> object:
        if isinstance(value, str):
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @classmethod
    def for_plan(
        cls,
        *,
        tenant_id: str,
        plan: PlanTier | str = PlanTier.FREE,
        limits: dict[PlanTier, int] | None = None,
    ) -> Subscription:
        """Construct a subscription with the default daily limit for ``plan``."""
        tier = PlanTier(plan) if not isinstance(plan, PlanTier) else plan
        table = limits if limits is not None else DEFAULT_PLAN_LIMITS
        return cls(
            tenant_id=tenant_id,
            plan=tier,
            queries_limit=table.get(tier, DEFAULT_PLAN_LIMITS[PlanTier.FREE]),
        )

    def remaining_queries(self) -> int:
        """Queries left for the current day."""
        return max(0, self.queries_limit - self.queries_used_today)

    def can_query(self, units: int = 1) -> bool:
        """Return True if the tenant may consume ``units`` queries."""
        if units < 0:
            raise ValueError("units must be non-negative")
        if self.subscription_status in (
            SubscriptionStatus.CANCELED,
            SubscriptionStatus.PAST_DUE,
        ):
            return False
        return self.queries_used_today + units <= self.queries_limit

    def record_query(self, units: int = 1) -> Subscription:
        """Increment daily usage after a successful query authorization."""
        if units < 0:
            raise ValueError("units must be non-negative")
        if not self.can_query(units):
            raise ValueError(
                f"query quota exceeded: used={self.queries_used_today} "
                f"limit={self.queries_limit} requested={units}"
            )
        self.queries_used_today += units
        return self.touch()

    def reset_daily_usage(self, *, period_start: datetime | None = None) -> Subscription:
        """Reset the daily counter and set a new period start."""
        self.queries_used_today = 0
        self.period_start = period_start or _utc_now()
        return self.touch()
