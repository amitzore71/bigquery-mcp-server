"""BigQuery SaaS platform — multi-tenant Kind system and domain models."""

from __future__ import annotations

__version__ = "0.2.0"

from bqsaas.config import Settings, get_settings
from bqsaas.kinds.base import (
    Kind,
    Resource,
    ResourceRef,
    ResourceStatus,
    generate_resource_id,
)
from bqsaas.kinds.registry import (
    KindRegistry,
    default_registry,
    register_builtin_kinds,
)
from bqsaas.models import (
    ApiKey,
    ChatSession,
    DataConnection,
    Message,
    Subscription,
    Tenant,
    UsageEvent,
    User,
    Workspace,
)
from bqsaas.storage.memory import MemoryStore

# Register domain models with the default kind registry on import.
register_builtin_kinds(default_registry)

__all__ = [
    "ApiKey",
    "ChatSession",
    "DataConnection",
    "Kind",
    "KindRegistry",
    "MemoryStore",
    "Message",
    "Resource",
    "ResourceRef",
    "ResourceStatus",
    "Settings",
    "Subscription",
    "Tenant",
    "UsageEvent",
    "User",
    "Workspace",
    "__version__",
    "default_registry",
    "generate_resource_id",
    "get_settings",
    "register_builtin_kinds",
]
