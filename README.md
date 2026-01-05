# BigQuery MCP Server & Chat Client

A natural language interface for querying BigQuery data using Google's Gemini AI and the Model Context Protocol (MCP). This project enables conversational data analysis without requiring SQL knowledge.

## Overview

This project consists of two main components:

1. **MCP Server** (`main.py`) - A FastMCP-based server that provides tools for interacting with BigQuery databases
2. **Chat Client** (`client/`) - A web-based chat interface powered by Google's Gemini AI for natural language queries

The system uses the Model Context Protocol to enable AI-driven interactions with your BigQuery tables.

## Features

- Natural language querying of BigQuery data
- Automatic data visualizations (charts, graphs, KPIs)
- Multi-table query support without manual SQL
- Attendance pattern analysis and school statistics
- Modern, professional web interface

## Quick Start

### Prerequisites

- Python 3.13 or newer
- Google Cloud project with BigQuery enabled
- Gemini API key (obtain from [Google AI Studio](https://aistudio.google.com/app/apikey))
- BigQuery service account credentials

### Installation

1. **Navigate to the project directory:**

   ```bash
   cd d:\.\bigquery-mcp-server
   ```

2. **Configure environment variables:**

   ```powershell
   $env:GEMINI_API_KEY = "your-api-key-here"
   ```

3. **Install dependencies:**

   ```bash
   uv sync
   ```

4. **Add service account credentials:**

   Place your `service-account.json` file in the root directory for BigQuery authentication.

5. **Start the MCP server:**

   ```bash
   uv run python main.py
   ```

6. **Launch the chat client (in a separate terminal):**

   ```bash
   uv run uvicorn client.app:app --reload --host 0.0.0.0 --port 8000
   ```

7. **Access the application:**
   ```
   http://localhost:8000
   ```

## Project Structure

```
bigquery-mcp-server/
├── main.py                    # MCP server with BigQuery tools
├── client/                    # Chat web application
│   ├── app.py                # FastAPI backend + Gemini integration
│   ├── templates/            # HTML templates
│   └── static/               # CSS and other static files
├── service-account.json      # BigQuery credentials (do not commit)
├── pyproject.toml            # Python dependencies
└── README.md                 # This file
```

## Example Queries

The system supports natural language questions such as:

- "Show me all the tables we have"
- "What's in the attendance table?"
- "How many students are in each school?"
- "What's the attendance rate for last week?"
- "Give me a breakdown of attendance by school"
- "Show me some sample data from the schools table"

The AI interprets your questions and retrieves the appropriate data automatically.

## System Architecture

1. User submits a question via the chat interface
2. Message is sent to Gemini AI for interpretation
3. Gemini determines if BigQuery access is required
4. If needed, appropriate MCP tools are invoked
5. MCP server executes queries on BigQuery
6. Results are returned to Gemini
7. Gemini formats the response (including visualizations if applicable)
8. Formatted response is displayed in the chat interface

## Configuration

The following environment variables can be configured:

- `GCP_PROJECT_ID` - Google Cloud project ID
- `DATASET_ID` - BigQuery dataset name
- `GEMINI_API_KEY` - Gemini API key (required for chat functionality)

## Additional Documentation

For detailed information about specific components:

- [Server README](./README_SERVER.md) - Comprehensive MCP server documentation
- [Client README](./client/README.md) - Chat interface details and customization

## Troubleshooting

**Authentication Errors**

- Verify `service-account.json` is in the correct location
- Confirm service account has appropriate BigQuery permissions

**Chat Not Responding**

- Ensure `GEMINI_API_KEY` environment variable is set
- Verify MCP server is running
- Check terminal output for error messages

**JSON Responses Instead of Formatted Output**

- Try rephrasing your question
- Confirm both server and client are running
- Check network connectivity
