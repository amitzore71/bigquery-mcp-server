"""Tests for pure BigQuery MCP tools (mocked client, no network)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from bqsaas.mcp import tools as tools_mod
from bqsaas.mcp.tools import (
    describe_table,
    execute_query,
    list_tables,
    validate_identifier,
)


class TestInvalidTableNames:
    @pytest.mark.parametrize(
        "bad_name",
        [
            "attendance; DROP",
            "attendance; DROP TABLE users--",
            "schools'; DROP TABLE schools; --",
            "foo bar",
            "table-name;rm",
            "../../etc/passwd",
            "a`b",
            "x;y",
            "",
        ],
    )
    def test_reject_invalid_table_names(self, bad_name: str):
        err = validate_identifier(bad_name, "table_name")
        assert err is not None, f"validate_identifier should reject {bad_name!r}"

        # describe_table should also return error status without calling client
        mock_client = MagicMock()
        result = describe_table(mock_client, "proj", "dataset", bad_name)
        assert result["status"] == "error"
        assert "message" in result
        mock_client.get_table.assert_not_called()


def _patch_settings():
    """Settings field names differ across agent drafts — normalize for tools."""
    from unittest.mock import patch

    mock_settings = MagicMock()
    mock_settings.max_bytes_billed = 10 * 1024 * 1024 * 1024
    mock_settings.max_query_bytes_billed = 10 * 1024 * 1024 * 1024
    mock_settings.max_result_rows = 1000
    mock_settings.max_query_rows = 1000
    mock_settings.query_timeout_seconds = 30
    return patch.object(tools_mod, "get_settings", return_value=mock_settings)


class TestExecuteQueryMocked:
    def test_execute_query_success_format(self):
        mock_client = MagicMock()
        mock_job = MagicMock()
        mock_job.result.return_value = [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]
        mock_job.total_bytes_processed = 1024
        mock_job.total_bytes_billed = 1024
        mock_client.query.return_value = mock_job

        with _patch_settings():
            result = execute_query(
                mock_client,
                "practice_project",
                "school_data",
                "SELECT 1 AS id",
            )

        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert "data" in result or "rows" in result or "row_count" in result

    def test_execute_query_error_format(self):
        mock_client = MagicMock()
        mock_client.query.side_effect = RuntimeError("boom: connection refused")

        with _patch_settings():
            result = execute_query(
                mock_client,
                "practice_project",
                "school_data",
                "SELECT 1",
            )

        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "message" in result
        assert "boom" in result["message"].lower() or "connection" in result["message"].lower()

    def test_execute_query_empty_sql_error(self):
        mock_client = MagicMock()
        with _patch_settings():
            result = execute_query(mock_client, "p", "d", "   ")
        assert result["status"] == "error"


class TestListTablesMocked:
    def test_list_tables_success_format(self):
        mock_table = MagicMock()
        mock_table.table_id = "attendance"
        mock_table.table_type = "TABLE"

        mock_client = MagicMock()
        mock_client.list_tables.return_value = [mock_table]

        result = list_tables(mock_client, "myproject", "school_data")

        assert isinstance(result, dict)
        assert result["status"] == "success"
        tables = result.get("tables") or []
        assert isinstance(tables, list)
        assert len(tables) >= 1
        assert tables[0]["table_id"] == "attendance"

    def test_list_tables_error_format(self):
        mock_client = MagicMock()
        mock_client.list_tables.side_effect = RuntimeError("no credentials")

        result = list_tables(mock_client, "myproject", "school_data")

        assert isinstance(result, dict)
        assert result["status"] == "error"
        assert "message" in result
