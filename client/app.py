"""
BigQuery MCP Chat Client - A clean FastAPI + HTMX chat application
for querying BigQuery attendance data using natural language.
"""

import asyncio
import html
import importlib.util
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from google import genai

# =============================================================================
# CONFIGURATION
# =============================================================================

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
MCP_SERVER_SCRIPT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "main.py")
)

# In-memory chat storage
chat_sessions: dict[str, list[dict]] = {}


# =============================================================================
# MCP TOOLS CONFIGURATION
# =============================================================================

MCP_TOOLS = [
    {
        "name": "list_tables",
        "description": "Lists all available tables in the dataset.",
        "parameters": {}
    },
    {
        "name": "describe_table",
        "description": "Gets the schema and metadata for a specific table.",
        "parameters": {"table_name": "Name of the table (e.g., 'attendance' or 'schools')"}
    },
    {
        "name": "execute_query",
        "description": "Executes a SQL query on BigQuery and returns results. Use this for any custom data retrieval.",
        "parameters": {"sql_query": "The SQL query to execute (BigQuery SQL dialect)"}
    },
    {
        "name": "get_sample_data",
        "description": "Retrieves sample rows from a specified table.",
        "parameters": {"table_name": "Name of the table", "limit": "Number of rows (default: 10, max: 100)"}
    },
    {
        "name": "get_table_stats",
        "description": "Gets statistical information about a table.",
        "parameters": {"table_name": "Name of the table"}
    },
    {
        "name": "join_attendance_schools",
        "description": "Performs a JOIN between attendance and schools tables. Use this for queries involving both tables.",
        "parameters": {
            "select_fields": "Comma-separated fields to select (default: \"*\")",
            "where_clause": "Optional WHERE clause without the WHERE keyword",
            "order_by": "Optional ORDER BY clause without ORDER BY keyword",
            "limit": "Maximum rows to return (default: 100)"
        }
    }
]


# =============================================================================
# MCP CLIENT
# =============================================================================

class MCPClient:
    """Direct client that imports and calls MCP tools from main.py."""
    
    def __init__(self):
        self._module = None
        
    def _load_module(self):
        """Lazily load the MCP server module."""
        if self._module is None:
            spec = importlib.util.spec_from_file_location("mcp_server", MCP_SERVER_SCRIPT)
            self._module = importlib.util.module_from_spec(spec)
            sys.modules["mcp_server"] = self._module
            spec.loader.exec_module(self._module)
        return self._module
    
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Call an MCP tool and return the result."""
        try:
            module = self._load_module()
            
            if not hasattr(module, tool_name):
                return {"status": "error", "message": f"Tool '{tool_name}' not found"}
            
            func = getattr(module, tool_name)
            
            # Handle FastMCP's FunctionTool wrapper - the actual function is in .fn
            if hasattr(func, 'fn') and callable(func.fn):
                func = func.fn
            elif hasattr(func, '_func') and callable(func._func):
                func = func._func
            
            if callable(func):
                return func(**arguments)
            else:
                return {"status": "error", "message": f"Tool '{tool_name}' is not callable"}
                
        except Exception as e:
            return {"status": "error", "message": f"Tool execution failed: {str(e)}"}
    
    def get_tools_description(self) -> str:
        """Get a formatted description of available tools for the LLM."""
        lines = []
        for tool in MCP_TOOLS:
            params = ""
            if tool["parameters"]:
                params = "\n" + "\n".join(
                    f"    - {name}: {desc}" 
                    for name, desc in tool["parameters"].items()
                )
            lines.append(f"- **{tool['name']}**: {tool['description']}{params}")
        return "\n".join(lines)


# Global client instance
mcp_client = MCPClient()


# =============================================================================
# GEMINI CONFIGURATION
# =============================================================================

def configure_gemini():
    """Configure and return Gemini client."""
    if not GEMINI_API_KEY:
        print("⚠️ Warning: GEMINI_API_KEY not set.")
        return None
    return genai.Client(api_key=GEMINI_API_KEY)


gemini_client = configure_gemini()
MODEL_ID = "gemini-3-flash-preview"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a helpful BigQuery data assistant. You help users query and analyze school attendance data.

You have access to these MCP tools to interact with BigQuery:

{tools_description}

**IMPORTANT GUIDELINES:**

1. When a user asks about DATA (attendance records, counts, dates, schools, students), you MUST use either:
   - `execute_query` with a proper SQL query, OR
   - `join_attendance_schools` for queries involving both tables

2. Use `list_tables` or `describe_table` ONLY when user specifically asks about table structure or schema.

3. The dataset is `practice-project-481414.school_data` with tables:
   - `attendance` - Contains student attendance records with fields like date, school_id, status, etc.
   - `schools` - Contains school information with school_id, school_name, etc.

4. When writing SQL, use full table paths like: `practice-project-481414.school_data.attendance`

5. To call a tool, respond with a JSON block like this:
```tool_call
{{"tool": "tool_name", "arguments": {{"arg1": "value1"}}}}
```

6. After receiving tool results, provide a clear explanation to the user.

**EXAMPLES:**

User: "Show me attendance for January 2nd"
→ Use execute_query with: SELECT * FROM `practice-project-481414.school_data.attendance` WHERE DATE(date) = '2026-01-02'

User: "What tables are available?"  
→ Use list_tables

User: "Attendance by school for today"
→ Use join_attendance_schools with appropriate where_clause
"""


# =============================================================================
# CORE PROCESSING FUNCTIONS
# =============================================================================

def extract_response_text(response) -> Optional[str]:
    """Safely extract text from a Gemini response."""
    # Try direct text accessor
    try:
        if hasattr(response, 'text'):
            return response.text
    except (ValueError, AttributeError):
        pass
    
    # Try candidates
    try:
        if hasattr(response, 'candidates') and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                parts = candidate.content.parts
                if parts:
                    return ''.join(p.text for p in parts if hasattr(p, 'text') and p.text)
    except (ValueError, AttributeError, IndexError):
        pass
    
    return None


def extract_tool_call(message: str) -> Optional[dict]:
    """Extract tool call JSON from message. Supports multiple formats."""
    
    patterns = [
        # Format 1: ```tool_call block
        (r'```tool_call\s*\n?(.*?)\n?```', 1),
        # Format 2: ```json block with tool/arguments
        (r'```json\s*\n?(.*?)\n?```', 1),
        # Format 3: Generic code block
        (r'```\s*\n?(.*?)\n?```', 1),
        # Format 4: Plain JSON object
        (r'(\{[^{}]*"tool"[^{}]*"arguments"[^{}]*\{[^{}]*\}[^{}]*\})', 1),
    ]
    
    for pattern, group in patterns:
        match = re.search(pattern, message, re.DOTALL)
        if match:
            candidate = match.group(group).strip()
            if '"tool"' in candidate and '"arguments"' in candidate:
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
    
    return None


def execute_tool_call(message: str) -> Optional[dict]:
    """Extract and execute tool call from message."""
    tool_call = extract_tool_call(message)
    if tool_call:
        tool_name = tool_call.get("tool")
        arguments = tool_call.get("arguments", {})
        if tool_name:
            return mcp_client.call_tool(tool_name, arguments)
    return None


def process_with_gemini(user_message: str, session_id: str) -> str:
    """Process user message with Gemini and MCP tools."""
    if not gemini_client:
        return "❌ Gemini API key not configured. Please set GEMINI_API_KEY environment variable."
    
    # Get chat history
    history = chat_sessions.get(session_id, [])
    
    # Build prompt
    tools_description = mcp_client.get_tools_description()
    system = SYSTEM_PROMPT.format(tools_description=tools_description)
    
    # Add recent history
    history_text = ""
    for msg in history[-4:]:
        history_text += f"\n{msg['role'].title()}: {msg['content']}"
    
    full_prompt = system
    if history_text:
        full_prompt += f"\n\nPrevious conversation:{history_text}"
    full_prompt += f"\n\nUser: {user_message}"
    
    try:
        # Get initial Gemini response
        response = gemini_client.models.generate_content(
            model=MODEL_ID,
            contents=full_prompt
        )
        assistant_message = extract_response_text(response)
        
        if not assistant_message:
            return "I apologize, but I couldn't process that request. Please try rephrasing."
        
        # Check for tool call
        tool_result = execute_tool_call(assistant_message)
        
        if tool_result:
            # Send tool result back to Gemini for interpretation
            follow_up = f"""The tool returned this result:
```json
{json.dumps(tool_result, indent=2, default=str)}
```

Please interpret this data and provide a clear, helpful response to: "{user_message}"

Format numbers nicely and use markdown tables if appropriate. Be concise but informative.

For data that can be visualized, include a VISUALIZATION BLOCK at the START:

```visualization
{{
    "type": "bar|line|pie|kpi|table|none",
    "title": "Chart Title",
    "data": {{
        "labels": ["Label1", "Label2"],
        "values": [100, 200],
        "colors": ["#4F46E5", "#10B981"]
    }},
    "kpis": [
        {{"label": "Metric Name", "value": "123", "change": "+5%", "trend": "up|down|neutral"}}
    ]
}}
```

Choose visualization type based on data:
- "bar": For comparing categories
- "line": For time series data
- "pie": For proportions/percentages
- "kpi": For single key metrics
- "table": For detailed multi-column data
- "none": For simple text responses

Always provide BOTH the visualization AND a clear text explanation."""

            follow_up_response = gemini_client.models.generate_content(
                model=MODEL_ID,
                contents=follow_up
            )
            follow_up_text = extract_response_text(follow_up_response)
            
            if follow_up_text:
                assistant_message = follow_up_text
            else:
                # Fallback: format the tool result ourselves
                assistant_message = f"I executed the query and received:\n\n```json\n{json.dumps(tool_result, indent=2, default=str)}\n```"
        
        # Store in history
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        chat_sessions[session_id].append({"role": "user", "content": user_message})
        chat_sessions[session_id].append({"role": "assistant", "content": assistant_message})
        
        return assistant_message
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error processing request: {str(e)}"


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def extract_visualization(text: str) -> tuple[dict | None, str]:
    """Extract visualization JSON block from AI response."""
    pattern = r'```visualization\s*\n(.*?)\n```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        try:
            viz_data = json.loads(match.group(1).strip())
            remaining_text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
            return viz_data, remaining_text
        except json.JSONDecodeError:
            pass
    
    return None, text


def generate_chart_html(viz_data: dict, chart_id: str) -> str:
    """Generate HTML for a Chart.js chart."""
    chart_type = viz_data.get("type", "bar")
    title = viz_data.get("title", "")
    data = viz_data.get("data", {})
    
    labels = json.dumps(data.get("labels", []))
    values = json.dumps(data.get("values", []))
    colors = data.get("colors", ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16"])
    background_colors = json.dumps(colors[:len(data.get("values", []))])
    
    if chart_type == "pie":
        chart_config = f'''{{
            type: 'doughnut',
            data: {{
                labels: {labels},
                datasets: [{{
                    data: {values},
                    backgroundColor: {background_colors},
                    borderWidth: 0
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ position: 'bottom', labels: {{ font: {{ size: 11 }}, padding: 16, usePointStyle: true }} }},
                    title: {{ display: {json.dumps(bool(title))}, text: {json.dumps(title)}, font: {{ size: 14, weight: '600' }} }}
                }}
            }}
        }}'''
    elif chart_type == "line":
        chart_config = f'''{{
            type: 'line',
            data: {{
                labels: {labels},
                datasets: [{{
                    label: {json.dumps(title)},
                    data: {values},
                    borderColor: '#4F46E5',
                    backgroundColor: 'rgba(79, 70, 229, 0.1)',
                    fill: true,
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{ display: {json.dumps(bool(title))}, text: {json.dumps(title)}, font: {{ size: 14, weight: '600' }} }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, grid: {{ color: 'rgba(0,0,0,0.05)' }} }},
                    x: {{ grid: {{ display: false }} }}
                }}
            }}
        }}'''
    else:  # bar (default)
        chart_config = f'''{{
            type: 'bar',
            data: {{
                labels: {labels},
                datasets: [{{
                    data: {values},
                    backgroundColor: {background_colors},
                    borderRadius: 6
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }},
                    title: {{ display: {json.dumps(bool(title))}, text: {json.dumps(title)}, font: {{ size: 14, weight: '600' }} }}
                }},
                scales: {{
                    y: {{ beginAtZero: true, grid: {{ color: 'rgba(0,0,0,0.05)' }} }},
                    x: {{ grid: {{ display: false }} }}
                }}
            }}
        }}'''
    
    return f'''
    <div class="chart-container" style="position: relative; height: 280px; width: 100%; margin: 1rem 0; padding: 1rem; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 12px; border: 1px solid #e2e8f0;">
        <canvas id="{chart_id}"></canvas>
    </div>
    <script>
        (function() {{
            const ctx = document.getElementById('{chart_id}');
            if (ctx) {{ new Chart(ctx, {chart_config}); }}
        }})();
    </script>
    '''


def generate_kpi_html(viz_data: dict) -> str:
    """Generate HTML for KPI cards."""
    kpis = viz_data.get("kpis", [])
    if not kpis:
        return ""
    
    cards = []
    for kpi in kpis:
        label = kpi.get("label", "")
        value = kpi.get("value", "")
        change = kpi.get("change", "")
        trend = kpi.get("trend", "neutral")
        
        trend_icon = "↑" if trend == "up" else ("↓" if trend == "down" else "→")
        trend_color = "#10b981" if trend == "up" else ("#ef4444" if trend == "down" else "#6b7280")
        
        change_html = f'<div style="font-size: 0.8rem; color: {trend_color};">{trend_icon} {change}</div>' if change else ''
        
        cards.append(f'''
        <div class="kpi-card" style="flex: 1; min-width: 140px; padding: 1rem; background: white; border-radius: 12px; border: 1px solid #e2e8f0; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <div style="font-size: 0.75rem; color: #64748b; font-weight: 500; text-transform: uppercase; margin-bottom: 0.5rem;">{label}</div>
            <div style="font-size: 1.75rem; font-weight: 700; color: #1e293b;">{value}</div>
            {change_html}
        </div>
        ''')
    
    return f'''<div class="kpi-grid" style="display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0;">{"".join(cards)}</div>'''


# =============================================================================
# MARKDOWN TO HTML CONVERSION
# =============================================================================

def convert_table_to_html(table_lines: list) -> str:
    """Convert markdown table to HTML."""
    if len(table_lines) < 2:
        return '\n'.join(table_lines)
    
    html_parts = ['<div class="prose"><table>']
    
    for i, line in enumerate(table_lines):
        cells = [c.strip() for c in line.split('|')[1:-1]]
        
        if i == 0:
            html_parts.append('<thead><tr>')
            html_parts.extend(f'<th>{cell}</th>' for cell in cells)
            html_parts.append('</tr></thead><tbody>')
        elif i == 1 and all(c.replace('-', '').replace(':', '') == '' for c in cells):
            continue  # Skip separator row
        else:
            html_parts.append('<tr>')
            html_parts.extend(f'<td>{cell}</td>' for cell in cells)
            html_parts.append('</tr>')
    
    html_parts.append('</tbody></table></div>')
    return ''.join(html_parts)


def convert_markdown_to_html(text: str) -> str:
    """Convert markdown to HTML with visualization support."""
    
    # Extract visualization blocks
    viz_data, text = extract_visualization(text)
    
    # Generate visualization HTML
    viz_html = ""
    if viz_data:
        chart_type = viz_data.get("type", "none")
        if chart_type in ["bar", "line", "pie"]:
            chart_id = f"chart_{datetime.now().timestamp()}".replace(".", "_")
            viz_html = generate_chart_html(viz_data, chart_id)
        elif chart_type == "kpi":
            viz_html = generate_kpi_html(viz_data)
    
    # Escape HTML
    text = html.escape(text)
    
    # Code blocks
    def replace_code_block(match):
        lang = match.group(1) or ''
        code = match.group(2)
        return f'<pre><code class="language-{lang}">{code}</code></pre>'
    
    text = re.sub(r'```(\w+)?\n(.*?)```', replace_code_block, text, flags=re.DOTALL)
    
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    
    # Bold and italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'\*([^*]+)\*', r'<em>\1</em>', text)
    
    # Headers
    text = re.sub(r'^### (.+)$', r'<h3>\1</h3>', text, flags=re.MULTILINE)
    text = re.sub(r'^## (.+)$', r'<h2>\1</h2>', text, flags=re.MULTILINE)
    text = re.sub(r'^# (.+)$', r'<h1>\1</h1>', text, flags=re.MULTILINE)
    
    # Lists
    text = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    text = re.sub(r'^- (.+)$', r'<li>\1</li>', text, flags=re.MULTILINE)
    
    # Process tables and lists
    lines = text.split('\n')
    in_table = False
    in_list = False
    table_lines = []
    result_lines = []
    
    for line in lines:
        if '|' in line and line.strip().startswith('|'):
            if in_list:
                result_lines.append('</ul>')
                in_list = False
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
        elif line.strip().startswith('<li>'):
            if in_table:
                result_lines.append(convert_table_to_html(table_lines))
                in_table = False
            if not in_list:
                result_lines.append('<ul class="styled-list">')
                in_list = True
            result_lines.append(line)
        else:
            if in_table:
                result_lines.append(convert_table_to_html(table_lines))
                in_table = False
            if in_list:
                result_lines.append('</ul>')
                in_list = False
            result_lines.append(line)
    
    if in_table:
        result_lines.append(convert_table_to_html(table_lines))
    if in_list:
        result_lines.append('</ul>')
    
    text = '\n'.join(result_lines)
    
    # Line breaks
    text = text.replace('\n\n', '</p><p>')
    text = text.replace('\n', '<br>')
    
    return f'{viz_html}<div class="prose-content"><p>{text}</p></div>'


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================

app = FastAPI(
    title="BigQuery MCP Chat Client",
    description="Chat with your BigQuery data using natural language"
)

# Templates and static files
templates_dir = os.path.join(os.path.dirname(__file__), "templates")
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(templates_dir, exist_ok=True)
os.makedirs(static_dir, exist_ok=True)

templates = Jinja2Templates(directory=templates_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, session_id: str = None):
    """Render the main chat page."""
    # Get session_id from query param, cookie, or create new
    if session_id is None:
        session_id = request.cookies.get("session_id", str(datetime.now().timestamp()))
    
    # Build recent chats list
    recent_chats = []
    for sid, messages in sorted(chat_sessions.items(), key=lambda x: x[0], reverse=True):
        preview = "New Session"
        if messages:
            for m in reversed(messages):
                if m["role"] == "user":
                    preview = m["content"][:30] + ("..." if len(m["content"]) > 30 else "")
                    break
        
        try:
            date_str = datetime.fromtimestamp(float(sid)).strftime("%b %d, %H:%M")
        except (ValueError, TypeError):
            date_str = sid[:8]
            
        recent_chats.append({
            "id": sid,
            "preview": preview,
            "date": date_str,
            "active": sid == session_id
        })
    
    # Get chat history for current session
    chat_history = chat_sessions.get(session_id, [])

    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "session_id": session_id,
            "recent_chats": recent_chats,
            "chat_history": chat_history,
            "resources": []
        }
    )
    
    # Set the session cookie
    response.set_cookie(key="session_id", value=session_id, max_age=86400*30)  # 30 days
    
    return response


@app.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, message: str = Form(...)):
    """Handle chat messages via HTMX."""
    session_id = request.cookies.get("session_id", "default")
    
    if not message.strip():
        return HTMLResponse("")
    
    # Process message in thread pool
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, process_with_gemini, message, session_id)
    
    # Convert markdown to HTML
    response_html = convert_markdown_to_html(response)
    
    # Return message pair as HTML
    html_content = f'''
    <div class="message-wrapper user" id="msg-{datetime.now().timestamp()}">
        <div class="message-bubble">
            <p>{html.escape(message)}</p>
        </div>
    </div>
    <div class="message-wrapper ai">
        <div class="message-avatar">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
            </svg>
        </div>
        <div class="message-bubble">
            <div class="prose-content">{response_html}</div>
        </div>
    </div>
    '''
    return HTMLResponse(html_content)


@app.get("/tools", response_class=HTMLResponse)
async def get_tools(request: Request):
    """Return available tools as HTML."""
    tools_html = ""
    for tool in MCP_TOOLS:
        tools_html += f'''
        <div class="tool-card">
            <h3 class="tool-title">{tool["name"]}</h3>
            <p class="tool-desc">{tool["description"]}</p>
        </div>
        '''
    return HTMLResponse(tools_html)


@app.post("/delete-session/{session_id}", response_class=HTMLResponse)
async def delete_session(request: Request, session_id: str):
    """Delete a chat session."""
    current_session = request.cookies.get("session_id")
    
    if session_id in chat_sessions:
        del chat_sessions[session_id]
    
    # If we deleted the current session, redirect to home (which will create a new session)
    if session_id == current_session:
        return HTMLResponse(status_code=200, content="", headers={"HX-Redirect": "/"})
    else:
        # Just reload the page to update the sidebar
        return HTMLResponse(status_code=200, content="", headers={"HX-Redirect": "/"})


@app.post("/clear-chat", response_class=HTMLResponse)
async def clear_chat(request: Request):
    """Clear messages in the current session and create a new one."""
    # Create a new session ID
    new_session_id = str(datetime.now().timestamp())
    
    # Redirect to home with new session
    return HTMLResponse(status_code=200, headers={"HX-Redirect": f"/?session_id={new_session_id}"})


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
