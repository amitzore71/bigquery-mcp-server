"""Application settings loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the BigQuery SaaS platform.

    Values are read from environment variables (case-insensitive) and
    optionally from a local ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "BigQuery SaaS"
    app_env: Literal["development", "staging", "production", "test"] = "development"

    gcp_project_id: str = "practice-project-481414"
    dataset_id: str = "school_data"
    service_account_path: str = "service-account.json"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    secret_key: str = "dev-secret-change-me"

    max_query_bytes_billed: int = Field(
        default=10 * 1024**3,
        ge=0,
        description="Maximum bytes billed per BigQuery job",
    )
    max_query_rows: int = Field(default=10_000, ge=1)
    free_daily_query_limit: int = Field(default=100, ge=0)
    pro_daily_query_limit: int = Field(default=10_000, ge=0)
    enterprise_daily_query_limit: int = Field(default=1_000_000, ge=0)

    @field_validator("app_env", mode="before")
    @classmethod
    def _normalize_env(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    def daily_query_limit_for_plan(self, plan: str) -> int:
        """Return the configured daily query limit for a plan name."""
        key = plan.strip().lower()
        mapping = {
            "free": self.free_daily_query_limit,
            "pro": self.pro_daily_query_limit,
            "enterprise": self.enterprise_daily_query_limit,
        }
        if key not in mapping:
            raise ValueError(f"unknown plan {plan!r}")
        return mapping[key]

    @property
    def is_production(self) -> bool:
        """True when running in the production environment."""
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance."""
    return Settings()
