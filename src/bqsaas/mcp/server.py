"""FastMCP server — multi-tenant-ready tools with lazy BigQuery client."""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastmcp import FastMCP

from bqsaas.config import get_settings
from bqsaas.mcp import tools as bq_tools
from bqsaas.mcp.bigquery_client import (
    ClientError,
    connection_error_result,
    get_default_client_cached,
)

logger = logging.getLogger(__name__)

mcp = FastMCP("BigQuery Attendance Schools Server")


def _ctx() -> tuple[Any, str, str] | tuple[None, None, None]:
    """Resolve default client + project/dataset for MCP stdio mode."""
    settings = get_settings()
    try:
        client = get_default_client_cached()
    except Exception as e:
        logger.warning("Client resolution failed: %s", e)
        client = None
    if client is None:
        return None, None, None
    return client, settings.gcp_project_id, settings.dataset_id


def _no_client() -> dict[str, Any]:
    return connection_error_result(
        "BigQuery client unavailable. Set GCP credentials "
        "(service-account.json or ADC) and GCP_PROJECT_ID / DATASET_ID."
    )


def format_schema(schema: list) -> str:
    return bq_tools.format_schema(schema)


@mcp.tool()
def list_tables() -> dict[str, Any]:
    """Lists all available tables in the dataset."""
    client, project_id, dataset_id = _ctx()
    if client is None:
        return _no_client()
    return bq_tools.list_tables(client, project_id, dataset_id)


@mcp.tool()
def describe_table(table_name: str) -> dict[str, Any]:
    """
    Gets the schema and metadata for a specific table.

    Args:
        table_name: Name of the table (e.g., 'attendance' or 'schools')
    """
    client, project_id, dataset_id = _ctx()
    if client is None:
        return _no_client()
    return bq_tools.describe_table(client, project_id, dataset_id, table_name)


@mcp.tool()
def execute_query(sql_query: str) -> dict[str, Any]:
    """
    Executes a SQL query on BigQuery and returns results.

    Args:
        sql_query: The SQL query to execute (BigQuery SQL dialect)
    """
    client, project_id, dataset_id = _ctx()
    if client is None:
        return _no_client()
    return bq_tools.execute_query(client, project_id, dataset_id, sql_query)


@mcp.tool()
def get_sample_data(table_name: str, limit: int = 10) -> dict[str, Any]:
    """
    Retrieves sample rows from a specified table.

    Args:
        table_name: Name of the table ('attendance' or 'schools')
        limit: Number of rows to retrieve (default: 10, max: 100)
    """
    client, project_id, dataset_id = _ctx()
    if client is None:
        return _no_client()
    return bq_tools.get_sample_data(
        client, project_id, dataset_id, table_name, limit=limit
    )


@mcp.tool()
def get_table_stats(table_name: str) -> dict[str, Any]:
    """
    Gets statistical information about a table.

    Args:
        table_name: Name of the table ('attendance' or 'schools')
    """
    client, project_id, dataset_id = _ctx()
    if client is None:
        return _no_client()
    return bq_tools.get_table_stats(client, project_id, dataset_id, table_name)


@mcp.tool()
def join_attendance_schools(
    select_fields: str = "*",
    where_clause: str = "",
    order_by: str = "",
    limit: int = 100,
) -> dict[str, Any]:
    """
    Performs a JOIN between attendance and schools tables.

    Args:
        select_fields: Comma-separated fields to select (default: "*")
        where_clause: Optional WHERE clause without the WHERE keyword
        order_by: Optional ORDER BY clause without ORDER BY keyword
        limit: Maximum rows to return (default: 100)
    """
    client, project_id, dataset_id = _ctx()
    if client is None:
        return _no_client()
    return bq_tools.join_attendance_schools(
        client,
        project_id,
        dataset_id,
        select_fields=select_fields,
        where_clause=where_clause,
        order_by=order_by,
        limit=limit,
    )


# ==================== RESOURCES ====================


@mcp.resource("schema://attendance")
def get_attendance_schema() -> str:
    """Provides the complete schema for the attendance table."""
    result = describe_table("attendance")
    if result["status"] == "success":
        return f"Attendance Table Schema:\n\n{format_schema(result['schema'])}"
    return f"Error: {result.get('message', 'unknown')}"


@mcp.resource("schema://schools")
def get_schools_schema() -> str:
    """Provides the complete schema for the schools table."""
    result = describe_table("schools")
    if result["status"] == "success":
        return f"Schools Table Schema:\n\n{format_schema(result['schema'])}"
    return f"Error: {result.get('message', 'unknown')}"


@mcp.resource("help://query-examples")
def get_query_examples() -> str:
    """Provides example queries for common use cases."""
    settings = get_settings()
    ds = f"{settings.gcp_project_id}.{settings.dataset_id}"
    return f"""
Common Query Examples:

1. Get all attendance records for a specific school:
   SELECT * FROM `{ds}.attendance` WHERE school_id = 'SCHOOL_123'

2. Count students by school:
   SELECT s.school_name, COUNT(a.student_id) as student_count
   FROM `{ds}.attendance` a
   JOIN `{ds}.schools` s ON a.school_id = s.school_id
   GROUP BY s.school_name

3. Get attendance rate by school:
   SELECT school_id,
          AVG(CASE WHEN status = 'present' THEN 1.0 ELSE 0.0 END) as attendance_rate
   FROM `{ds}.attendance`
   GROUP BY school_id

4. List all schools with their details:
   SELECT * FROM `{ds}.schools` ORDER BY school_name
"""


# ==================== PROMPTS ====================


@mcp.prompt()
def analyze_attendance(school_name: str = "") -> str:
    """Generate a prompt for analyzing attendance patterns."""
    if school_name:
        return f"""
Analyze attendance patterns for {school_name}:
1. Get the overall attendance rate
2. Identify trends over time
3. Compare with other schools if applicable
4. Provide actionable insights
"""
    return """
Perform a comprehensive attendance analysis:
1. Calculate overall attendance rates across all schools
2. Identify schools with lowest attendance
3. Find temporal patterns (day of week, time of year)
4. Provide recommendations for improvement
"""


@mcp.prompt()
def school_comparison() -> str:
    """Generate a prompt for comparing schools."""
    return """
Compare schools based on:
1. Total student enrollment
2. Attendance rates
3. Geographic distribution
4. Any other available metrics

Present findings in a clear, actionable format.
"""


def create_mcp(name: str = "BigQuery Attendance Schools Server") -> FastMCP:
    """Factory for tests / multi-instance use (returns the module-level mcp)."""
    # Tools already registered on module-level ``mcp``; name is for API parity
    _ = name
    return mcp


def run(transport: str = "stdio") -> None:
    """Entrypoint used by ``main.py`` and ``python -m bqsaas``."""
    mcp.run(transport=transport)


if __name__ == "__main__":
    run()
