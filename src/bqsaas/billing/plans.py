"""Subscription plan definitions and limits."""

from __future__ import annotations

from dataclasses import dataclass

from bqsaas.errors import NotFoundError, ValidationError
from bqsaas.models.subscription import DEFAULT_PLAN_LIMITS, PlanTier


@dataclass(frozen=True, slots=True)
class Plan:
    """A billing plan with resource limits."""

    id: str
    name: str
    daily_query_limit: int
    max_workspaces: int
    max_connections: int
    max_users: int
    max_chat_sessions: int = 1_000
    description: str = ""

    @property
    def is_unlimited_queries(self) -> bool:
        return self.daily_query_limit < 0 or self.daily_query_limit >= 1_000_000_000

    def to_tier(self) -> PlanTier:
        return PlanTier(self.id)


PLANS: dict[str, Plan] = {
    "free": Plan(
        id="free",
        name="Free",
        daily_query_limit=DEFAULT_PLAN_LIMITS.get(PlanTier.FREE, 100),
        max_workspaces=1,
        max_connections=1,
        max_users=3,
        max_chat_sessions=50,
        description="Free tier for evaluation and small pilots",
    ),
    "pro": Plan(
        id="pro",
        name="Pro",
        daily_query_limit=DEFAULT_PLAN_LIMITS.get(PlanTier.PRO, 10_000),
        max_workspaces=10,
        max_connections=10,
        max_users=50,
        max_chat_sessions=5_000,
        description="Pro tier for production school districts",
    ),
    "enterprise": Plan(
        id="enterprise",
        name="Enterprise",
        daily_query_limit=DEFAULT_PLAN_LIMITS.get(PlanTier.ENTERPRISE, 1_000_000),
        max_workspaces=1_000,
        max_connections=1_000,
        max_users=10_000,
        max_chat_sessions=1_000_000,
        description="Enterprise tier with high limits",
    ),
}


def get_plan(plan_id: str | PlanTier) -> Plan:
    """Return a plan by id or raise NotFoundError."""
    if isinstance(plan_id, PlanTier):
        key = plan_id.value
    else:
        if not plan_id:
            raise ValidationError("plan_id is required", field="plan_id")
        key = str(plan_id).strip().lower()
    plan = PLANS.get(key)
    if plan is None:
        raise NotFoundError(
            f"Unknown plan '{plan_id}'",
            resource="plan",
            resource_id=str(plan_id),
            details={"available": sorted(PLANS.keys())},
        )
    return plan


def list_plans() -> list[Plan]:
    """Return all plans in stable order: free, pro, enterprise."""
    order = ("free", "pro", "enterprise")
    return [PLANS[k] for k in order if k in PLANS]
