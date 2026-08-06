"""Tests for Kind system: Kind enum, Resource, IDs, KindRegistry."""

from __future__ import annotations

import pytest

from bqsaas.kinds.base import (
    Kind,
    ResourceStatus,
    generate_resource_id,
)


def _parse_id(rid: str):
    """parse_id if available, else ResourceRef.parse."""
    try:
        from bqsaas.kinds.base import parse_id  # type: ignore

        return parse_id(rid)
    except ImportError:
        pass
    try:
        from bqsaas.kinds import parse_id  # type: ignore

        return parse_id(rid)
    except ImportError:
        pass
    from bqsaas.kinds.base import ResourceRef

    return ResourceRef.parse(rid)


def _parsed_kind(parsed):
    if isinstance(parsed, Kind):
        return parsed
    if isinstance(parsed, tuple):
        k = parsed[0]
        return Kind(k) if isinstance(k, str) else k
    kind_attr = getattr(parsed, "kind", None)
    if isinstance(kind_attr, Kind):
        return kind_attr
    if isinstance(kind_attr, str):
        return Kind(kind_attr)
    raise AssertionError(f"Cannot extract kind from parse result: {parsed!r}")


class TestKindEnum:
    def test_kind_values_exist(self):
        """Core Kind enum members must be defined."""
        expected = {
            "tenant",
            "user",
            "api_key",
            "workspace",
            "data_connection",
            "chat_session",
            "message",
            "subscription",
        }
        values = {k.value for k in Kind}
        missing = expected - values
        assert not missing, f"Missing Kind values: {missing}"

    def test_kind_values_are_unique(self):
        values = [k.value for k in Kind]
        assert len(values) == len(set(values))

    def test_kind_is_string_enum(self):
        for k in Kind:
            assert isinstance(k.value, str)
            assert k.value


class TestGenerateResourceId:
    def test_starts_with_kind_value_and_underscore(self):
        for kind in Kind:
            rid = generate_resource_id(kind)
            assert rid.startswith(f"{kind.value}_"), (
                f"generate_resource_id({kind!r}) -> {rid!r} "
                f"should start with '{kind.value}_'"
            )

    def test_ids_are_unique(self):
        ids = {generate_resource_id(Kind.TENANT) for _ in range(50)}
        assert len(ids) == 50

    def test_id_has_suffix(self):
        rid = generate_resource_id(Kind.USER)
        suffix = rid[len("user_") :]
        assert len(suffix) >= 8


class TestParseId:
    def test_parse_id_roundtrip(self):
        for kind in Kind:
            rid = generate_resource_id(kind)
            parsed = _parse_id(rid)
            assert _parsed_kind(parsed) == kind

    def test_parse_id_known_format(self):
        sample = f"{Kind.TENANT.value}_abc123xyz"
        parsed = _parse_id(sample)
        assert _parsed_kind(parsed) == Kind.TENANT


class TestResourceStatus:
    def test_status_enum_has_active_and_deleted(self):
        assert ResourceStatus.ACTIVE.value == "active"
        assert ResourceStatus.DELETED.value == "deleted"

    def test_status_values_unique(self):
        values = [s.value for s in ResourceStatus]
        assert len(values) == len(set(values))


class TestKindRegistry:
    def test_list_kinds_non_empty_after_registration(self):
        try:
            from bqsaas.kinds.registry import KindRegistry
            from bqsaas import default_registry, register_builtin_kinds
        except ImportError:
            from bqsaas.kinds.registry import KindRegistry

            registry = KindRegistry()
            kinds = registry.list_kinds()
            assert len(list(kinds)) > 0
            return

        # Prefer default_registry with builtins
        try:
            register_builtin_kinds(default_registry)
        except Exception:
            pass
        kinds = default_registry.list_kinds()
        assert len(list(kinds)) > 0

    def test_list_kinds_includes_tenant_or_bigquery(self):
        """Registry may catalog resource kinds and/or connection kinds."""
        try:
            from bqsaas import default_registry

            registry = default_registry
        except ImportError:
            from bqsaas.kinds.registry import KindRegistry

            registry = KindRegistry()

        kinds = list(registry.list_kinds())
        names: set[str] = set()
        for item in kinds:
            if isinstance(item, Kind):
                names.add(item.value)
            elif isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict):
                names.add(
                    str(item.get("kind") or item.get("name") or item.get("value"))
                )
            else:
                names.add(
                    str(
                        getattr(item, "kind", None)
                        or getattr(item, "value", None)
                        or getattr(item, "name", item)
                    )
                )

        # Either platform resource kinds or connection kinds are fine
        assert names, "list_kinds returned empty"
        assert (
            "tenant" in names
            or "bigquery" in names
            or Kind.TENANT in kinds
            or any("tenant" in n or "bigquery" in n for n in names)
        )
