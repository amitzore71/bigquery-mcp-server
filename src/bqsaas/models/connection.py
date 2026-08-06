"""Data connection domain model (BigQuery credentials reference)."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict, Field, field_validator, model_validator

from bqsaas.kinds.base import Kind, Resource


class DataConnection(Resource):
    """Connection to an external data source (default: BigQuery).

    Secrets are never stored as raw values on this model for logging safety.
    Use ``credentials_secret_ref`` (preferred) or a filesystem
    ``credentials_path`` pointing at a service-account JSON file.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: Kind = Kind.DATA_CONNECTION

    workspace_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1, max_length=200)
    provider: str = Field(default="bigquery", min_length=1, max_length=64)
    project_id: str = Field(..., min_length=1, max_length=128)
    dataset_id: str = Field(..., min_length=1, max_length=128)
    credentials_path: str | None = Field(
        default=None,
        description="Filesystem path to service-account JSON (dev/local only)",
    )
    credentials_secret_ref: str | None = Field(
        default=None,
        description="Secret manager / vault reference for credentials (preferred)",
    )
    location: str = Field(
        default="US",
        description="BigQuery dataset/job location",
        max_length=64,
    )

    @field_validator("name", "project_id", "dataset_id", "workspace_id")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field must not be empty")
        return stripped

    @field_validator("provider")
    @classmethod
    def _normalize_provider(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def _require_credentials_ref(self) -> DataConnection:
        if not self.credentials_path and not self.credentials_secret_ref:
            raise ValueError(
                "either credentials_path or credentials_secret_ref is required"
            )
        return self

    def credentials_descriptor(self) -> dict[str, Any]:
        """Return a log-safe summary of how credentials are configured.

        Never includes raw secret material.
        """
        if self.credentials_secret_ref:
            return {
                "mode": "secret_ref",
                "ref": self.credentials_secret_ref,
            }
        return {
            "mode": "path",
            "path": self.credentials_path,
        }

    def __repr__(self) -> str:
        # Avoid accidental secret path dumps in noisy reprs when path is long.
        return (
            f"DataConnection(id={self.id!r}, name={self.name!r}, "
            f"provider={self.provider!r}, project_id={self.project_id!r}, "
            f"dataset_id={self.dataset_id!r}, "
            f"credentials={self.credentials_descriptor()!r})"
        )
