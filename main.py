import os
from typing import Any

from fastmcp import FastMCP
from google.api_core.exceptions import GoogleAPIError
from google.cloud import bigquery
from google.oauth2 import service_account

# Initialize MCP server
mcp = FastMCP("BigQuery Attendance Schools Server")

# Configuration
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "practice-project-481414")
DATASET_ID = os.environ.get("DATASET_ID", "school_data")
SERVICE_ACCOUNT_PATH = "service-account.json"

# Initialize BigQuery client
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_PATH)
client = bigquery.Client(project=PROJECT_ID, credentials=credentials)


def get_full_table_id(table_name: str) -> str:
    """Returns fully qualified table ID."""
    return f"{PROJECT_ID}.{DATASET_ID}.{table_name}"


def format_schema(schema: list) -> str:
    """Formats table schema as readable text."""
    return "\n".join(
        f"- {f['name']}: {f['type']} ({f['mode']}) - {f['description']}"
        for f in schema
    )


# ==================== TOOLS ====================

@mcp.tool()
def list_tables() -> dict[str, Any]:
    """Lists all available tables in the dataset."""
    try:
        tables = [
            {
                "table_id": t.table_id,
                "full_table_id": get_full_table_id(t.table_id),
                "table_type": t.table_type
            }
            for t in client.list_tables(f"{PROJECT_ID}.{DATASET_ID}")
        ]
        return {"status": "success", "tables": tables, "count": len(tables)}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def describe_table(table_name: str) -> dict[str, Any]:
    """
    Gets the schema and metadata for a specific table.
    
    Args:
        table_name: Name of the table (e.g., 'attendance' or 'schools')
    """
    try:
        table_ref = get_full_table_id(table_name)
        table = client.get_table(table_ref)
        
        schema_info = [
            {
                "name": f.name,
                "type": f.field_type,
                "mode": f.mode,
                "description": f.description or "No description"
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
            "schema": schema_info
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def execute_query(sql_query: str) -> dict[str, Any]:
    """
    Executes a SQL query on BigQuery and returns results.
    
    Args:
        sql_query: The SQL query to execute (BigQuery SQL dialect)
    """
    try:
        job_config = bigquery.QueryJobConfig(
            use_query_cache=True,
            maximum_bytes_billed=10 * 1024 * 1024 * 1024  # 10 GB limit
        )
        
        query_job = client.query(sql_query, job_config=job_config)
        results = query_job.result(timeout=180)
        rows = [dict(row) for row in results]
        
        return {
            "status": "success",
            "row_count": len(rows),
            "total_bytes_processed": query_job.total_bytes_processed,
            "total_bytes_billed": query_job.total_bytes_billed,
            "data": rows[:10000]
        }
    except GoogleAPIError as e:
        return {"status": "error", "message": f"BigQuery API Error: {e}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def get_sample_data(table_name: str, limit: int = 10) -> dict[str, Any]:
    """
    Retrieves sample rows from a specified table.
    
    Args:
        table_name: Name of the table ('attendance' or 'schools')
        limit: Number of rows to retrieve (default: 10, max: 100)
    """
    query = f"SELECT * FROM `{get_full_table_id(table_name)}` LIMIT {min(limit, 100)}"
    return execute_query(query)


@mcp.tool()
def get_table_stats(table_name: str) -> dict[str, Any]:
    """
    Gets statistical information about a table.
    
    Args:
        table_name: Name of the table ('attendance' or 'schools')
    """
    try:
        table_ref = get_full_table_id(table_name)
        table = client.get_table(table_ref)
        
        result = execute_query(f"SELECT COUNT(*) as total_rows FROM `{table_ref}`")
        total_rows = (
            result["data"][0]["total_rows"]
            if result["status"] == "success"
            else table.num_rows
        )
        
        return {
            "status": "success",
            "table_name": table_name,
            "total_rows": total_rows,
            "total_columns": len(table.schema),
            "size_bytes": table.num_bytes,
            "size_mb": round(table.num_bytes / (1024 * 1024), 2),
            "created": str(table.created),
            "last_modified": str(table.modified)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def join_attendance_schools(
    select_fields: str = "*",
    where_clause: str = "",
    order_by: str = "",
    limit: int = 100
) -> dict[str, Any]:
    """
    Performs a JOIN between attendance and schools tables.
    
    Args:
        select_fields: Comma-separated fields to select (default: "*")
        where_clause: Optional WHERE clause without the WHERE keyword
        order_by: Optional ORDER BY clause without ORDER BY keyword
        limit: Maximum rows to return (default: 100)
    """
    where_part = f"WHERE {where_clause}" if where_clause else ""
    order_part = f"ORDER BY {order_by}" if order_by else ""
    
    query = f"""
        SELECT {select_fields}
        FROM `{get_full_table_id('attendance')}` a
        JOIN `{get_full_table_id('schools')}` s ON a.school_id = s.school_id
        {where_part}
        {order_part}
        LIMIT {limit}
    """
    return execute_query(query)


# ==================== RESOURCES ====================

@mcp.resource("schema://attendance")
def get_attendance_schema() -> str:
    """Provides the complete schema for the attendance table."""
    result = describe_table("attendance")
    if result["status"] == "success":
        return f"Attendance Table Schema:\n\n{format_schema(result['schema'])}"
    return f"Error: {result['message']}"


@mcp.resource("schema://schools")
def get_schools_schema() -> str:
    """Provides the complete schema for the schools table."""
    result = describe_table("schools")
    if result["status"] == "success":
        return f"Schools Table Schema:\n\n{format_schema(result['schema'])}"
    return f"Error: {result['message']}"


@mcp.resource("help://query-examples")
def get_query_examples() -> str:
    """Provides example queries for common use cases."""
    return """
Common Query Examples:

1. Get all attendance records for a specific school:
   SELECT * FROM attendance WHERE school_id = 'SCHOOL_123'

2. Count students by school:
   SELECT s.school_name, COUNT(a.student_id) as student_count
   FROM attendance a
   JOIN schools s ON a.school_id = s.school_id
   GROUP BY s.school_name

3. Get attendance rate by school:
   SELECT school_id, 
          AVG(CASE WHEN status = 'present' THEN 1.0 ELSE 0.0 END) as attendance_rate
   FROM attendance
   GROUP BY school_id

4. List all schools with their details:
   SELECT * FROM schools ORDER BY school_name
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


if __name__ == "__main__":
    mcp.run(transport="stdio")
