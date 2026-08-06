"""Domain models for the BigQuery SaaS platform."""

from bqsaas.models.api_key import ApiKey, generate_raw_api_key, hash_api_key
from bqsaas.models.connection import DataConnection
from bqsaas.models.session import ChatSession, Message, MessageRole, ToolCall

# Alias used by chat_service and older call sites
ChatMessage = Message
from bqsaas.models.subscription import (
    DEFAULT_PLAN_LIMITS,
    PlanTier,
    Subscription,
    SubscriptionStatus,
)
from bqsaas.models.tenant import Tenant
from bqsaas.models.usage import UsageEvent, UsageEventType
from bqsaas.models.user import User, UserRole
from bqsaas.models.workspace import Workspace

__all__ = [
    "ApiKey",
    "ChatMessage",
    "ChatSession",
    "DataConnection",
    "DEFAULT_PLAN_LIMITS",
    "Message",
    "MessageRole",
    "PlanTier",
    "Subscription",
    "SubscriptionStatus",
    "Tenant",
    "ToolCall",
    "UsageEvent",
    "UsageEventType",
    "User",
    "UserRole",
    "Workspace",
    "generate_raw_api_key",
    "hash_api_key",
]
