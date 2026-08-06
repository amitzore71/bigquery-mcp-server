"""Kind registry: maps kinds to model classes and provides ID/ref utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bqsaas.kinds.base import Kind, Resource, ResourceRef


@dataclass(frozen=True, slots=True)
class KindMeta:
    """Metadata describing a registered kind."""

    kind: Kind
    model_class: type[Resource]
    description: str
    schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata for API/discovery responses."""
        return {
            "kind": self.kind.value,
            "description": self.description,
            "model": self.model_class.__name__,
            "schema": self.schema,
        }


class KindRegistry:
    """Central registry of resource kinds and their Pydantic models.

    Thread-safe for reads after initial registration (typically at import
    time). Registration is expected during application bootstrap.
    """

    def __init__(self) -> None:
        self._entries: dict[Kind, KindMeta] = {}

    def register(
        self,
        kind: Kind,
        model_class: type[Resource],
        description: str = "",
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a kind with its model class and human description.

        Args:
            kind: The Kind enum value.
            model_class: Pydantic Resource subclass for this kind.
            description: Short documentation string.
            overwrite: If True, replace an existing registration.

        Raises:
            TypeError: If model_class is not a Resource subclass.
            ValueError: If kind is already registered and overwrite is False.
        """
        if not isinstance(model_class, type) or not issubclass(
            model_class, Resource
        ):
            raise TypeError(
                f"model_class must be a subclass of Resource, got {model_class!r}"
            )
        if kind in self._entries and not overwrite:
            raise ValueError(f"kind {kind.value!r} is already registered")

        schema: dict[str, Any] = {}
        if hasattr(model_class, "model_json_schema"):
            schema = model_class.model_json_schema()

        first_line = (model_class.__doc__ or "").strip().split("\n")[0]
        self._entries[kind] = KindMeta(
            kind=kind,
            model_class=model_class,
            description=description or first_line,
            schema=schema,
        )

    def get_model(self, kind: Kind) -> type[Resource]:
        """Return the registered model class for ``kind``.

        Raises:
            KeyError: If the kind has not been registered.
        """
        meta = self._entries.get(kind)
        if meta is None:
            raise KeyError(f"kind {kind.value!r} is not registered")
        return meta.model_class

    def get_meta(self, kind: Kind) -> KindMeta:
        """Return full metadata for a registered kind.

        Raises:
            KeyError: If the kind has not been registered.
        """
        meta = self._entries.get(kind)
        if meta is None:
            raise KeyError(f"kind {kind.value!r} is not registered")
        return meta

    def is_registered(self, kind: Kind) -> bool:
        """Return True if ``kind`` has been registered."""
        return kind in self._entries

    def parse_id(self, resource_id: str) -> tuple[Kind, str]:
        """Parse a kind-prefixed resource id into ``(Kind, raw_id)``.

        Example: ``tenant_01HXYZ...`` → ``(Kind.TENANT, "01HXYZ...")``

        Args:
            resource_id: Fully-qualified resource identifier.

        Returns:
            Tuple of (Kind, raw portion after the kind prefix).

        Raises:
            ValueError: If the id does not match any known kind prefix.
        """
        if not resource_id or not isinstance(resource_id, str):
            raise ValueError("resource_id must be a non-empty string")

        # Prefer longest matching prefix so compound names resolve correctly.
        candidates = sorted(Kind, key=lambda k: len(k.value), reverse=True)
        for kind in candidates:
            prefix = f"{kind.value}_"
            if resource_id.startswith(prefix):
                raw = resource_id[len(prefix) :]
                if not raw:
                    raise ValueError(
                        f"resource_id {resource_id!r} has empty raw id after prefix"
                    )
                return kind, raw

        raise ValueError(
            f"resource_id {resource_id!r} does not match any known Kind prefix"
        )

    def validate_ref(self, ref: ResourceRef | dict[str, Any] | str) -> ResourceRef:
        """Validate and normalize a resource reference.

        Accepts a ResourceRef, a dict with kind/id keys, or a string form
        understood by :meth:`ResourceRef.parse`.

        Raises:
            ValueError: If the ref is malformed or kind is unknown.
            TypeError: If the input type is unsupported.
        """
        if isinstance(ref, ResourceRef):
            parsed = ref
        elif isinstance(ref, str):
            parsed = ResourceRef.parse(ref)
        elif isinstance(ref, dict):
            kind_val = ref.get("kind")
            id_val = ref.get("id")
            if kind_val is None or id_val is None:
                raise ValueError("ref dict must contain 'kind' and 'id'")
            kind = kind_val if isinstance(kind_val, Kind) else Kind(str(kind_val))
            parsed = ResourceRef(kind=kind, id=str(id_val))
        else:
            raise TypeError(
                f"unsupported ref type {type(ref)!r}; expected ResourceRef, dict, or str"
            )

        # Ensure the id prefix is consistent with the declared kind when present.
        try:
            parsed_kind, _ = self.parse_id(parsed.id)
            if parsed_kind is not parsed.kind:
                raise ValueError(
                    f"ref kind {parsed.kind.value!r} does not match id prefix "
                    f"for {parsed.id!r} (parsed as {parsed_kind.value!r})"
                )
        except ValueError as exc:
            msg = str(exc)
            if "does not match id prefix" in msg:
                raise
            if "does not match any known Kind prefix" not in msg:
                raise

        return parsed

    def list_kinds(self) -> list[dict[str, Any]]:
        """Return metadata for all registered kinds, sorted by kind value."""
        return [
            meta.to_dict()
            for meta in sorted(self._entries.values(), key=lambda m: m.kind.value)
        ]

    def clear(self) -> None:
        """Remove all registrations (primarily for tests)."""
        self._entries.clear()


# Module-level singleton used by the package.
default_registry = KindRegistry()


def register_builtin_kinds(registry: KindRegistry | None = None) -> KindRegistry:
    """Register all built-in domain models with the given (or default) registry.

    Import is deferred to avoid circular imports between kinds and models.
    """
    reg = registry if registry is not None else default_registry

    from bqsaas.models.api_key import ApiKey
    from bqsaas.models.connection import DataConnection
    from bqsaas.models.session import ChatSession, Message
    from bqsaas.models.subscription import Subscription
    from bqsaas.models.tenant import Tenant
    from bqsaas.models.usage import UsageEvent
    from bqsaas.models.user import User
    from bqsaas.models.workspace import Workspace

    builtins: list[tuple[Kind, type[Resource], str]] = [
        (Kind.TENANT, Tenant, "Multi-tenant organization root resource"),
        (Kind.USER, User, "Platform user belonging to a tenant"),
        (Kind.WORKSPACE, Workspace, "Logical workspace within a tenant"),
        (
            Kind.DATA_CONNECTION,
            DataConnection,
            "BigQuery (or other) data connection with credentials reference",
        ),
        (Kind.API_KEY, ApiKey, "Hashed API key for programmatic access"),
        (Kind.CHAT_SESSION, ChatSession, "AI chat session within a workspace"),
        (Kind.MESSAGE, Message, "Single message in a chat session"),
        (Kind.SUBSCRIPTION, Subscription, "Tenant plan and quota subscription"),
        (Kind.USAGE_EVENT, UsageEvent, "Metered usage event for billing/limits"),
    ]

    for kind, model_cls, desc in builtins:
        if not reg.is_registered(kind):
            reg.register(kind, model_cls, desc)

    return reg
