# BigQuery Chat Client

A web-based chat interface for querying BigQuery data using natural language. This client leverages Google's Gemini AI to interpret questions and retrieve data without requiring SQL knowledge.

## Overview

The chat client provides an intuitive interface for interacting with BigQuery databases through conversational queries. It integrates with the MCP server to execute database operations and presents results in a clean, professional format.

## Features

- **Natural Language Processing** - Query data using plain English
- **Automatic Visualizations** - Charts and graphs generated based on query results
- **Professional Interface** - Clean, modern design optimized for data analysis
- **Real-time Updates** - HTMX-powered dynamic content without page reloads
- **Session Management** - Persistent chat history and conversation tracking
- **Responsive Design** - Optimized for various screen sizes

## Getting Started

### Prerequisites

1. Gemini API key ([obtain here](https://aistudio.google.com/app/apikey))
2. MCP server configured and accessible
3. Python 3.13 or newer

### Setup Instructions

**Step 1: Configure API Key**

```powershell
$env:GEMINI_API_KEY = "your-actual-api-key-here"
```

**Step 2: Install Dependencies**

From the project root directory:

```bash
uv sync
```

**Step 3: Start the Server**

```bash
uv run uvicorn client.app:app --reload --host 0.0.0.0 --port 8000
```

**Step 4: Access the Application**

Navigate to: `http://localhost:8000`

## Example Queries

### Basic Operations

- "What tables do we have?"
- "Show me the attendance table structure"
- "Give me 5 rows from the schools table"

### Data Analysis

- "How many students are in each school?"
- "What's the attendance rate this week?"
- "Show me attendance trends over time"
- "Which school has the best attendance?"

### Advanced Queries

- "Compare attendance rates across all schools"
- "Find schools with attendance below 80%"
- "Show me a breakdown of attendance by day of week"

The AI interprets natural language and generates appropriate queries automatically.

## System Architecture

### Request Flow

1. **User Input** - Question submitted via chat interface
2. **AI Processing** - Gemini analyzes the request
3. **Tool Selection** - AI determines required MCP tools
4. **Query Execution** - MCP server executes BigQuery operations
5. **Data Retrieval** - Raw results returned from BigQuery
6. **Response Formatting** - Gemini formats results with visualizations
7. **Display** - Formatted response shown in chat interface

Processing typically completes within seconds.

## Technology Stack

- **FastAPI** - Modern web framework for Python
- **HTMX** - Dynamic HTML updates without JavaScript
- **Jinja2** - Server-side template rendering
- **Google Gemini** - AI model (gemini-3-flash-preview)
- **Chart.js** - Client-side data visualization
- **Custom CSS** - Professional styling

## Project Structure

```
client/
├── app.py                 # FastAPI application
│                          # - Message handling
│                          # - Gemini integration
│                          # - MCP tool invocation
│                          # - Markdown to HTML conversion
│
├── templates/
│   └── index.html        # Chat interface
│                          # - Professional design
│                          # - HTMX integration
│                          # - Chart.js visualizations
│
├── static/
│   └── styles.css        # Custom styles
│
└── README.md             # This file
```

## Advanced Features

### Intelligent Visualizations

The AI automatically generates appropriate visualizations:

- **Bar charts** - Category comparisons
- **Line charts** - Time series data
- **Pie charts** - Proportional data
- **KPI cards** - Key metrics
- **Tables** - Detailed multi-column data
