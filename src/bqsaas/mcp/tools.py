"""
Pure BigQuery tool functions — all take an explicit client + project/dataset.

No module-level client init. Safe to import without credentials.
Return consistent ``{"status": "success"|"error", ...}`` dicts.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from bqsaas.config import get_settings

# SQL identifier safety: table / dataset style names only
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Select list / order-by: allow common SQL identifier chars and punctuation
_SAFE_SELECT_RE = re.compile(r"^[a-zA-Z0-9_.*,\s`]+$")
_SAFE_ORDER_RE = re.compile(r"^[a-zA-Z0-9_.,\s`]+$")


def validate_identifier(name: str, label: str = "identifier") -> Optional[str]:
    """Return error message if ``name`` is not a safe SQL identifier, else None."""
    if not name or not _IDENTIFIER_RE.match(name):
        return (
            f"Invalid {label} '{name}': must match "
            r"^[a-zA-Z_][a-zA-Z0-9_]*$"
        )
    return None


def full_table_id(project_id: str, dataset_id: str, table_name: str) -> str:
    return f"{project_id}.{dataset_id}.{table_name}"


def format_schema(schema: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {f['name']}: {f['type']} ({f['mode']}) - {f.get('description', '')}"
        for f in schema
    )


def _validate_project_id(project_id: str) -> Optional[str]:
    """GCP project ids allow lowercase letters, digits, and hyphens."""
    if not project_id or not re.match(r"^[a-zA-Z0-9][a-zA-Z0-9\-]*$", project_id):
        return f"Invalid project_id '{project_id}'"
    return None


def list_tables(
    client: Any,
    project_id: str,
    dataset_id: str,
) -> dict[str, Any]:
    """List tables in the given dataset."""
    err = _validate_project_id(project_id) or validate_identifier(
        dataset_id, "dataset_id"
    )
    if err:
        return {"status": "error", "message": err}

    try:
        tables = [
            {
                "table_id": t.table_id,
                "full_table_id": full_table_id(project_id, dataset_id, t.table_id),
                "table_type": t.table_type,
            }
            for t in client.list_tables(f"{project_id}.{dataset_id}")
        ]
        return {"status": "success", "tables": tables, "count": len(tables)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def describe_table(
    client: Any,
    project_id: str,
    dataset_id: str,
    table_name: str,
) -> dict[str, Any]:
    """Schema and metadata for a table."""
    err = validate_identifier(table_name, "table_name")
    if err:
        return {"status": "error", "message": err}

    try:
        table_ref = full_table_id(project_id, dataset_id, table_name)
        table = client.get_table(table_ref)
        schema_info = [
            {
                "name": f.name,
                "type": f.field_type,
                "mode": f.mode,
                "description": f.description or "No description",
            }
            for f in table.schema
        ]
        return {
            "status": "success",
            "table_id": table.table_id,
            "full_table_id": table_ref,
            "num_rows": table.num_rows,
            "num_bytes": table.num_bytes,
            "created": str(table.created),
            "modified": str(table.modified),
            "schema": schema_info,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def execute_query(
    client: Any,
    project_id: str,
    dataset_id: str,
    sql_query: str,
    *,
    max_bytes_billed: Optional[int] = None,
    max_rows: Optional[int] = None,
    timeout: Optional[int] = None,
) -> dict[str, Any]:
    """
    Execute SQL with safety limits.

    ``project_id`` / ``dataset_id`` are accepted for API symmetry (and default
    project context) but the SQL itself may reference any allowed tables.
    """
    settings = get_settings()

    def _int_setting(name: str, default: int) -> int:
        val = getattr(settings, name, default)
        try:
            return int(val)
        except (TypeError, ValueError):
            return default

    max_bytes = (
        max_bytes_billed
        if max_bytes_billed is not None
        else _int_setting("max_query_bytes_billed", 10 * 1024**3)
    )
    row_limit = (
        max_rows if max_rows is not None else _int_setting("max_query_rows", 10_000)
    )
    query_timeout = (
        timeout if timeout is not None else _int_setting("query_timeout_seconds", 180)
    )

    if not sql_query or not sql_query.strip():
        return {"status": "error", "message": "sql_query must not be empty"}

    try:
        from google.api_core.exceptions import GoogleAPIError
        from google.cloud import bigquery
    except ImportError as e:
        return {"status": "error", "message": f"BigQuery libraries missing: {e}"}

    try:
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            maximum_bytes_billed=max_bytes,
        )
        query_job = client.query(sql_query, job_config=job_config)
        results = query_job.result(timeout=query_timeout)
        rows = [dict(row) for row in results]
        truncated = len(rows) > row_limit
        return {
            "status": "success",
            "row_count": len(rows),
            "returned_rows": min(len(rows), row_limit),
            "truncated": truncated,
            "total_bytes_processed": query_job.total_bytes_processed,
            "total_bytes_billed": query_job.total_bytes_billed,
            "data": rows[:row_limit],
            "project_id": project_id,
            "dataset_id": dataset_id,
        }
    except GoogleAPIError as e:
        return {"status": "error", "message": f"BigQuery API Error: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_sample_data(
    client: Any,
    project_id: str,
    dataset_id: str,
    table_name: str,
    limit: int = 10,
) -> dict[str, Any]:
    """Sample rows from a table (identifier-validated)."""
    err = validate_identifier(table_name, "table_name")
    if err:
        return {"status": "error", "message": err}

    safe_limit = max(1, min(int(limit), 100))
    table_ref = full_table_id(project_id, dataset_id, table_name)
    query = f"SELECT * FROM `{table_ref}` LIMIT {safe_limit}"
    return execute_query(client, project_id, dataset_id, query, max_rows=safe_limit)


def get_table_stats(
    client: Any,
    project_id: str,
    dataset_id: str,
    table_name: str,
) -> dict[str, Any]:
    """Row count / size stats for a table."""
    err = validate_identifier(table_name, "table_name")
    if err:
        return {"status": "error", "message": err}

    try:
        table_ref = full_table_id(project_id, dataset_id, table_name)
        table = client.get_table(table_ref)
        count_result = execute_query(
            client,
            project_id,
            dataset_id,
            f"SELECT COUNT(*) AS total_rows FROM `{table_ref}`",
            max_rows=1,
        )
        total_rows = (
            count_result["data"][0]["total_rows"]
            if count_result.get("status") == "success" and count_result.get("data")
            else table.num_rows
        )
        return {
            "status": "success",
            "table_name": table_name,
            "total_rows": total_rows,
            "total_columns": len(table.schema),
            "size_bytes": table.num_bytes,
            "size_mb": round((table.num_bytes or 0) / (1024 * 1024), 2),
            "created": str(table.created),
            "last_modified": str(table.modified),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def join_attendance_schools(
    client: Any,
    project_id: str,
    dataset_id: str,
    select_fields: str = "*",
    where_clause: str = "",
    order_by: str = "",
    limit: int = 100,
) -> dict[str, Any]:
    """
    Convenience JOIN between attendance and schools.

    ``select_fields`` and ``order_by`` are lightly validated; ``where_clause``
    is still user-controlled (use with trusted callers / AI only).
    """
    for table in ("attendance", "schools"):
        err = validate_identifier(table, "table_name")
        if err:
            return {"status": "error", "message": err}

    select_fields = (select_fields or "*").strip()
    if select_fields != "*" and not _SAFE_SELECT_RE.match(select_fields):
        return {
            "status": "error",
            "message": "select_fields contains invalid characters",
        }

    if order_by and not _SAFE_ORDER_RE.match(order_by):
        return {"status": "error", "message": "order_by contains invalid characters"}

    # Basic injection guard on where: reject semicolons / comments
    if where_clause and re.search(r";|--|/\*", where_clause):
        return {
            "status": "error",
            "message": "where_clause contains forbidden sequences",
        }

    safe_limit = max(1, min(int(limit), 1000))
    att = full_table_id(project_id, dataset_id, "attendance")
    sch = full_table_id(project_id, dataset_id, "schools")
    where_part = f"WHERE {where_clause}" if where_clause else ""
    order_part = f"ORDER BY {order_by}" if order_by else ""

    query = f"""
        SELECT {select_fields}
        FROM `{att}` a
        JOIN `{sch}` s ON a.school_id = s.school_id
        {where_part}
        {order_part}
        LIMIT {safe_limit}
    """
    return execute_query(
        client, project_id, dataset_id, query, max_rows=safe_limit
    )


# Tool metadata for Gemini / OpenAPI-style discovery
TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "list_tables",
        "description": "Lists all available tables in the dataset.",
        "parameters": {},
    },
    {
        "name": "describe_table",
        "description": "Gets the schema and metadata for a specific table.",
        "parameters": {
            "table_name": "Name of the table (e.g., 'attendance' or 'schools')"
        },
    },
    {
        "name": "execute_query",
        "description": (
            "Executes a SQL query on BigQuery and returns results. "
            "Use this for any custom data retrieval."
        ),
        "parameters": {
            "sql_query": "The SQL query to execute (BigQuery SQL dialect)"
        },
    },
    {
        "name": "get_sample_data",
        "description": "Retrieves sample rows from a specified table.",
        "parameters": {
            "table_name": "Name of the table",
            "limit": "Number of rows (default: 10, max: 100)",
        },
    },
    {
        "name": "get_table_stats",
        "description": "Gets statistical information about a table.",
        "parameters": {"table_name": "Name of the table"},
    },
    {
        "name": "join_attendance_schools",
        "description": (
            "Performs a JOIN between attendance and schools tables. "
            "Use this for queries involving both tables."
        ),
        "parameters": {
            "select_fields": 'Comma-separated fields to select (default: "*")',
            "where_clause": "Optional WHERE clause without the WHERE keyword",
            "order_by": "Optional ORDER BY clause without ORDER BY keyword",
            "limit": "Maximum rows to return (default: 100)",
        },
    },
]


def call_tool(
    name: str,
    arguments: dict[str, Any],
    client: Any,
    project_id: str,
    dataset_id: str,
) -> dict[str, Any]:
    """Dispatch a tool by name with explicit client context."""
    dispatch = {
        "list_tables": lambda: list_tables(client, project_id, dataset_id),
        "describe_table": lambda: describe_table(
            client, project_id, dataset_id, arguments.get("table_name", "")
        ),
        "execute_query": lambda: execute_query(
            client,
            project_id,
            dataset_id,
            arguments.get("sql_query") or arguments.get("sql", ""),
        ),
        "get_sample_data": lambda: get_sample_data(
            client,
            project_id,
            dataset_id,
            arguments.get("table_name", ""),
            limit=int(arguments.get("limit", 10)),
        ),
        "get_table_stats": lambda: get_table_stats(
            client, project_id, dataset_id, arguments.get("table_name", "")
        ),
        "join_attendance_schools": lambda: join_attendance_schools(
            client,
            project_id,
            dataset_id,
            select_fields=arguments.get("select_fields", "*"),
            where_clause=arguments.get("where_clause", ""),
            order_by=arguments.get("order_by", ""),
            limit=int(arguments.get("limit", 100)),
        ),
    }
    fn = dispatch.get(name)
    if fn is None:
        return {"status": "error", "message": f"Unknown tool: {name}"}
    try:
        return fn()
    except Exception as e:
        return {"status": "error", "message": f"Tool execution failed: {e}"}
