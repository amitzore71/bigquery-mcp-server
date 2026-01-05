# BigQuery MCP Server

The MCP server component provides the backend infrastructure for BigQuery interactions. Built with FastMCP, it exposes a collection of tools that enable AI-driven data queries and analysis.

## Overview

The server (`main.py`) implements a set of tools for interacting with BigQuery databases without requiring direct SQL knowledge. It follows the Model Context Protocol specification, providing an AI-compatible interface for database operations.

## Available Tools

The server provides the following tools:

### 1. `list_tables()`

Returns a list of all tables in the configured dataset.

**Use case:** Discovering available data sources

### 2. `describe_table(table_name)`

Retrieves comprehensive metadata about a specified table, including schema, row count, size, and timestamps.

**Parameters:**

- `table_name` - Name of the table (e.g., `attendance` or `schools`)

**Use case:** Understanding table structure and contents

### 3. `execute_query(sql_query)`

Executes arbitrary SQL queries against BigQuery.

**Parameters:**

- `sql_query` - SQL query in BigQuery SQL dialect

**Example:** `SELECT * FROM attendance WHERE date = '2026-01-05'`

**Safety features:**

- 10 GB query billing limit
- Query result caching
- 180-second timeout
- 10,000 row result limit

### 4. `get_sample_data(table_name, limit=10)`

Retrieves sample rows from a specified table for data exploration.

**Parameters:**

- `table_name` - Target table name
- `limit` - Number of rows to return (default: 10, maximum: 100)

**Use case:** Quick data inspection

### 5. `get_table_stats(table_name)`

Returns statistical information about a table, including row count, size, and modification timestamps.

**Parameters:**

- `table_name` - Name of the table to analyze

**Use case:** Understanding table size and scope

### 6. `join_attendance_schools(...)`

Performs JOIN operations between attendance and schools tables with customizable parameters.

**Parameters:**

- `select_fields` - Comma-separated column names (default: all columns)
- `where_clause` - Optional filter conditions (without WHERE keyword)
- `order_by` - Optional sorting specification (without ORDER BY keyword)
- `limit` - Maximum rows to return (default: 100)

**Use case:** Combining data from multiple tables

## Resources

The server exposes the following resources for AI context:

- `schema://attendance` - Complete schema for the attendance table
- `schema://schools` - Complete schema for the schools table
- `help://query-examples` - Common query patterns and examples

## Prompts

Pre-configured prompts for common analysis tasks:

- `analyze_attendance(school_name)` - Generate attendance pattern analysis
- `school_comparison()` - Compare schools across multiple metrics

## Configuration

### Environment Variables

- `GCP_PROJECT_ID` - Google Cloud project ID
- `DATASET_ID` - BigQuery dataset name

### Service Account Setup

A `service-account.json` file is required in the root directory with appropriate BigQuery permissions.

**Setup steps:**

1. Navigate to Google Cloud Console
2. Go to IAM & Admin > Service Accounts
3. Create a new service account
4. Assign BigQuery User and BigQuery Data Editor roles
5. Generate a JSON key and save as `service-account.json`

## Running the Server

### Standalone Mode

To run the MCP server independently:

```bash
uv run python main.py
```

This starts the server in stdio mode for standard MCP client communication.

### Integrated Mode

The chat client (`client/app.py`) imports the server directly and invokes tools as Python functions. No separate server process is required when using the chat client.

## Troubleshooting

**google.auth.exceptions.DefaultCredentialsError**

- Verify `service-account.json` exists in the root directory
- Confirm file name matches exactly

**Permission Denied Errors**

- Check service account has BigQuery User and BigQuery Data Editor roles
- Verify project ID is correct

**Table Not Found**

- Confirm `GCP_PROJECT_ID` and `DATASET_ID` environment variables
- Verify table exists in BigQuery console

**Query Exceeded Billing Limit**

- Query processes more than 10 GB of data
- Add WHERE clauses to reduce data scanned
- Consider table partitioning
