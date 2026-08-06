"""BigQuery client factory — lazy, mock-friendly, soft-fail without credentials."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Optional, Protocol, runtime_checkable

from bqsaas.config import Settings, get_settings
from bqsaas.models import DataConnection

logger = logging.getLogger(__name__)


@runtime_checkable
class BigQueryClientLike(Protocol):
    """Minimal protocol so tests can inject mocks."""

    def list_tables(self, dataset: str) -> Any: ...
    def get_table(self, table: str) -> Any: ...
    def query(self, sql: str, job_config: Any = None) -> Any: ...


class ClientError(Exception):
    """Raised when a client cannot be constructed."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _load_credentials(connection: DataConnection):
    """Load google.oauth2 service-account credentials if available."""
    try:
        from google.oauth2 import service_account
    except ImportError as e:
        raise ClientError(f"google-auth not installed: {e}") from e

    path = getattr(connection, "credentials_path", None)
    if path and os.path.isfile(path):
        return service_account.Credentials.from_service_account_file(path)

    # credentials_secret_ref / ADC fallback
    return None


def get_client_from_connection(
    connection: DataConnection,
) -> BigQueryClientLike:
    """
    Build a ``google.cloud.bigquery.Client`` from a ``DataConnection``.

    Soft-fails with ``ClientError`` (never crashes import).
    """
    try:
        from google.cloud import bigquery
    except ImportError as e:
        raise ClientError(f"google-cloud-bigquery not installed: {e}") from e

    try:
        credentials = _load_credentials(connection)
    except Exception as e:
        raise ClientError(f"Failed to load credentials: {e}") from e

    kwargs: dict[str, Any] = {"project": connection.project_id}
    if credentials is not None:
        kwargs["credentials"] = credentials
    location = getattr(connection, "location", None)
    if location:
        kwargs["location"] = location

    try:
        return bigquery.Client(**kwargs)
    except Exception as e:
        raise ClientError(f"Failed to create BigQuery client: {e}") from e


def get_client(
    project_id: Optional[str] = None,
    credentials_path: Optional[str] = None,
    location: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> BigQueryClientLike:
    """Convenience factory using Settings defaults (legacy single-tenant path)."""
    settings = settings or get_settings()
    path = credentials_path or settings.service_account_path
    if path and not os.path.isabs(path):
        path = os.path.abspath(path)

    # DataConnection requires workspace_id + credentials
    if path and os.path.isfile(path):
        conn = DataConnection(
            tenant_id="system",
            workspace_id="system",
            name="default",
            project_id=project_id or settings.gcp_project_id,
            dataset_id=settings.dataset_id,
            credentials_path=path,
            location=location or "US",
        )
    else:
        conn = DataConnection(
            tenant_id="system",
            workspace_id="system",
            name="default",
            project_id=project_id or settings.gcp_project_id,
            dataset_id=settings.dataset_id,
            credentials_secret_ref="env:ADC",
            location=location or "US",
        )
    return get_client_from_connection(conn)


@lru_cache(maxsize=1)
def get_default_client_cached() -> Optional[BigQueryClientLike]:
    """Lazily create the process-wide default client. Returns None on soft-fail."""
    try:
        return get_client()
    except ClientError as e:
        logger.warning("Default BigQuery client unavailable: %s", e.message)
        return None


def clear_client_cache() -> None:
    get_default_client_cached.cache_clear()


def connection_error_result(message: str) -> dict[str, Any]:
    return {"status": "error", "message": message}
