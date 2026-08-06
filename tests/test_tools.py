"""Unit tests for pure BigQuery tools (mocked client)."""

from __future__ import annotations

from types import SimpleNamespace

from bqsaas.mcp.tools import (
    describe_table,
    execute_query,
    get_sample_data,
    list_tables,
    validate_identifier,
)


class FakeField:
    def __init__(self, name, field_type="STRING", mode="NULLABLE", description=None):
        self.name = name
        self.field_type = field_type
        self.mode = mode
        self.description = description


class FakeTable:
    def __init__(self):
        self.table_id = "attendance"
        self.num_rows = 42
        self.num_bytes = 1024
        self.created = "2026-01-01"
        self.modified = "2026-01-02"
        self.schema = [FakeField("school_id"), FakeField("status")]


class FakeClient:
    def list_tables(self, dataset):
        return [
            SimpleNamespace(table_id="attendance", table_type="TABLE"),
            SimpleNamespace(table_id="schools", table_type="TABLE"),
        ]

    def get_table(self, table_ref):
        return FakeTable()

    def query(self, sql, job_config=None):
        class Job:
            total_bytes_processed = 100
            total_bytes_billed = 100

            def result(self, timeout=None):
                return [{"school_id": "S1", "status": "present"}]

        return Job()


def test_validate_identifier():
    assert validate_identifier("attendance") is None
    assert validate_identifier("bad-name") is not None
    assert validate_identifier("1bad") is not None
    assert validate_identifier("drop;table") is not None


def test_list_tables():
    result = list_tables(FakeClient(), "practice-project-481414", "school_data")
    assert result["status"] == "success"
    assert result["count"] == 2


def test_describe_table_rejects_injection():
    result = describe_table(
        FakeClient(), "proj", "ds", "attendance; DROP TABLE"
    )
    assert result["status"] == "error"


def test_describe_table_ok():
    result = describe_table(FakeClient(), "proj", "ds", "attendance")
    assert result["status"] == "success"
    assert result["num_rows"] == 42
    assert len(result["schema"]) == 2


def test_get_sample_data():
    result = get_sample_data(FakeClient(), "proj", "ds", "attendance", limit=5)
    assert result["status"] == "success"
    assert result["row_count"] == 1
