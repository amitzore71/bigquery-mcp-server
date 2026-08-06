"""Kind system: typed resource taxonomy for multi-tenant SaaS."""

from bqsaas.kinds.base import (
    Kind,
    Resource,
    ResourceRef,
    ResourceStatus,
    generate_resource_id,
)
from bqsaas.kinds.registry import (
    KindMeta,
    KindRegistry,
    default_registry,
    register_builtin_kinds,
)

# Alias used by services/auth layers written in parallel.
generate_id = generate_resource_id

__all__ = [
    "Kind",
    "KindMeta",
    "KindRegistry",
    "Resource",
    "ResourceRef",
    "ResourceStatus",
    "default_registry",
    "generate_id",
    "generate_resource_id",
    "register_builtin_kinds",
]
