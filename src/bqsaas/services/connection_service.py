"""Data connection management (BigQuery project/dataset bindings)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bqsaas.billing.meter import UsageMeter
from bqsaas.errors import ForbiddenError, NotFoundError, ValidationError
from bqsaas.kinds import Kind, generate_id
from bqsaas.kinds.base import ResourceStatus
from bqsaas.models import DataConnection

if TYPE_CHECKING:
    from bqsaas.storage import MemoryStore

SUPPORTED_PROVIDERS = frozenset({"bigquery"})


class ConnectionService:
    """CRUD for tenant-scoped data connections with ownership checks."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._meter = UsageMeter(store)

    def create_connection(
        self,
        tenant_id: str,
        workspace_id: str,
        *,
        name: str,
        project_id: str,
        dataset_id: str | None = None,
        provider: str = "bigquery",
        credentials_path: str | None = None,
        credentials_secret_ref: str | None = None,
        location: str = "US",
    ) -> DataConnection:
        self._assert_tenant_exists(tenant_id)
        self._assert_workspace_in_tenant(tenant_id, workspace_id)

        if not name or not name.strip():
            raise ValidationError("Connection name is required", field="name")
        if not project_id or not project_id.strip():
            raise ValidationError("project_id is required", field="project_id")
        if provider not in SUPPORTED_PROVIDERS:
            raise ValidationError(
                f"Unsupported provider '{provider}'",
                field="provider",
                details={"supported": sorted(SUPPORTED_PROVIDERS)},
            )
        if not credentials_path and not credentials_secret_ref:
            raise ValidationError(
                "either credentials_path or credentials_secret_ref is required",
                field="credentials_path",
            )
        if not dataset_id:
            raise ValidationError("dataset_id is required", field="dataset_id")

        existing = self.list_connections(tenant_id)
        self._meter.check_resource_limit(tenant_id, "connections", len(existing))

        connection = DataConnection(
            id=generate_id(Kind.DATA_CONNECTION),
            tenant_id=tenant_id,
            workspace_id=workspace_id,
            name=name.strip(),
            provider=provider,
            project_id=project_id.strip(),
            dataset_id=dataset_id.strip(),
            credentials_path=credentials_path,
            credentials_secret_ref=credentials_secret_ref,
            location=location,
        )
        self._store.save(connection)
        return connection

    def get_connection(self, tenant_id: str, connection_id: str) -> DataConnection:
        connection = self._store.get(Kind.DATA_CONNECTION, connection_id)
        if connection is None:
            raise NotFoundError(
                "Data connection not found",
                resource="data_connection",
                resource_id=connection_id,
            )
        if getattr(connection, "tenant_id", None) != tenant_id:
            raise ForbiddenError("Data connection does not belong to tenant")
        return connection  # type: ignore[return-value]

    def list_connections(
        self,
        tenant_id: str,
        *,
        workspace_id: str | None = None,
        active_only: bool = False,
    ) -> list[DataConnection]:
        self._assert_tenant_exists(tenant_id)
        items = self._store.list_by_tenant(Kind.DATA_CONNECTION, tenant_id)
        result: list[DataConnection] = []
        for c in items:
            if workspace_id is not None and getattr(c, "workspace_id", None) != workspace_id:
                continue
            if active_only and getattr(c, "status", None) is not ResourceStatus.ACTIVE:
                continue
            result.append(c)  # type: ignore[arg-type]
        result.sort(key=lambda c: getattr(c, "created_at", None) or 0, reverse=True)
        return result

    def update_connection(
        self,
        tenant_id: str,
        connection_id: str,
        *,
        name: str | None = None,
        project_id: str | None = None,
        dataset_id: str | None = None,
        credentials_path: str | None = None,
        credentials_secret_ref: str | None = None,
        is_active: bool | None = None,
    ) -> DataConnection:
        connection = self.get_connection(tenant_id, connection_id)

        if name is not None:
            if not name.strip():
                raise ValidationError("Connection name cannot be empty", field="name")
            connection.name = name.strip()
        if project_id is not None:
            if not project_id.strip():
                raise ValidationError("project_id cannot be empty", field="project_id")
            connection.project_id = project_id.strip()
        if dataset_id is not None:
            connection.dataset_id = dataset_id.strip()
        if credentials_path is not None:
            connection.credentials_path = credentials_path or None
        if credentials_secret_ref is not None:
            connection.credentials_secret_ref = credentials_secret_ref or None
        if is_active is not None:
            connection.status = (
                ResourceStatus.ACTIVE if is_active else ResourceStatus.SUSPENDED
            )

        connection.touch()
        self._store.save(connection)
        return connection

    def delete_connection(self, tenant_id: str, connection_id: str) -> None:
        self.get_connection(tenant_id, connection_id)
        self._store.soft_delete(
            Kind.DATA_CONNECTION, connection_id, tenant_id=tenant_id
        )

    def get_connection_for_workspace(
        self,
        tenant_id: str,
        workspace_id: str,
    ) -> DataConnection | None:
        """Return the first active connection for a workspace, if any."""
        connections = self.list_connections(
            tenant_id,
            workspace_id=workspace_id,
            active_only=True,
        )
        return connections[0] if connections else None

    def _assert_tenant_exists(self, tenant_id: str) -> None:
        tenant = self._store.get(Kind.TENANT, tenant_id)
        if tenant is None:
            raise NotFoundError(
                "Tenant not found", resource="tenant", resource_id=tenant_id
            )

    def _assert_workspace_in_tenant(self, tenant_id: str, workspace_id: str) -> None:
        workspace = self._store.get(Kind.WORKSPACE, workspace_id)
        if workspace is None:
            raise NotFoundError(
                "Workspace not found",
                resource="workspace",
                resource_id=workspace_id,
            )
        if getattr(workspace, "tenant_id", None) != tenant_id:
            raise ForbiddenError("Workspace does not belong to tenant")
