"""HTMX chat UI mounted on the multi-tenant SaaS API."""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from bqsaas.auth.dependencies import authenticate_api_key
from bqsaas.ai.gemini import process_with_gemini
from bqsaas.errors import BqSaaSError
from bqsaas.mcp.bigquery_client import ClientError, get_client_from_connection
from bqsaas.services.chat_service import ChatService
from bqsaas.services.connection_service import ConnectionService
from bqsaas.storage.memory import MemoryStore

router = APIRouter(tags=["web"])

# Resolve client package paths (repo root / client)
_CLIENT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "client")
)
_TEMPLATES_DIR = os.path.join(_CLIENT_DIR, "templates")
_STATIC_DIR = os.path.join(_CLIENT_DIR, "static")

templates = Jinja2Templates(directory=_TEMPLATES_DIR) if os.path.isdir(_TEMPLATES_DIR) else None


def mount_static(app) -> None:
    """Mount client static assets if present."""
    if os.path.isdir(_STATIC_DIR):
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


def _store(request: Request) -> MemoryStore:
    return request.app.state.store


def _demo_key(request: Request) -> Optional[str]:
    demo = getattr(request.app.state, "demo", None) or {}
    return demo.get("raw_api_key") or demo.get("api_key")


def _resolve_auth(request: Request):
    """Resolve AuthContext from cookie, header, or demo bootstrap key."""
    store = _store(request)
    raw = (
        request.headers.get("x-api-key")
        or request.cookies.get("api_key")
        or ""
    )
    auth_header = request.headers.get("authorization") or ""
    if auth_header.lower().startswith("bearer "):
        raw = auth_header[7:].strip() or raw
    if not raw:
        raw = _demo_key(request) or ""
    if not raw:
        return None
    try:
        return authenticate_api_key(store, raw)
    except BqSaaSError:
        return None


def _escape(text: str) -> str:
    return html.escape(text or "")


def _coerce_text(value: Any) -> str:
    """Normalize Gemini/tool payloads to a plain string for markdown rendering."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        content = value.get("content")
        if isinstance(content, str):
            return content
        if content is not None:
            return str(content)
        return json.dumps(value, indent=2, default=str)
    return str(value)


def extract_visualization(text: str) -> tuple[dict | None, str]:
    text = _coerce_text(text)
    pattern = r"```visualization\s*\n(.*?)\n```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        try:
            viz_data = json.loads(match.group(1).strip())
            remaining = re.sub(pattern, "", text, flags=re.DOTALL).strip()
            return viz_data, remaining
        except json.JSONDecodeError:
            pass
    return None, text


def generate_chart_html(viz_data: dict, chart_id: str) -> str:
    chart_type = viz_data.get("type", "bar")
    title = viz_data.get("title", "")
    data = viz_data.get("data", {})
    labels = json.dumps(data.get("labels", []))
    values = json.dumps(data.get("values", []))
    colors = data.get(
        "colors",
        ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#EC4899"],
    )
    bg = json.dumps(colors[: max(len(data.get("values", [])), 1)])

    if chart_type == "pie":
        chart_config = f"""{{
            type: 'doughnut',
            data: {{ labels: {labels}, datasets: [{{ data: {values}, backgroundColor: {bg}, borderWidth: 0 }}] }},
            options: {{ responsive: true, maintainAspectRatio: false,
                plugins: {{ legend: {{ position: 'bottom' }},
                    title: {{ display: {json.dumps(bool(title))}, text: {json.dumps(title)} }} }} }}
        }}"""
    elif chart_type == "line":
        chart_config = f"""{{
            type: 'line',
            data: {{ labels: {labels}, datasets: [{{ label: {json.dumps(title)}, data: {values},
                borderColor: '#4F46E5', backgroundColor: 'rgba(79,70,229,0.1)', fill: true, tension: 0.4 }}] }},
            options: {{ responsive: true, maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }},
                    title: {{ display: {json.dumps(bool(title))}, text: {json.dumps(title)} }} }},
                scales: {{ y: {{ beginAtZero: true }}, x: {{ grid: {{ display: false }} }} }} }}
        }}"""
    else:
        chart_config = f"""{{
            type: 'bar',
            data: {{ labels: {labels}, datasets: [{{ data: {values}, backgroundColor: {bg}, borderRadius: 6 }}] }},
            options: {{ responsive: true, maintainAspectRatio: false,
                plugins: {{ legend: {{ display: false }},
                    title: {{ display: {json.dumps(bool(title))}, text: {json.dumps(title)} }} }},
                scales: {{ y: {{ beginAtZero: true }}, x: {{ grid: {{ display: false }} }} }} }}
        }}"""

    return f"""
    <div class="chart-container" style="position:relative;height:280px;width:100%;margin:1rem 0;padding:1rem;background:linear-gradient(135deg,#f8fafc 0%,#f1f5f9 100%);border-radius:12px;border:1px solid #e2e8f0;">
        <canvas id="{chart_id}"></canvas>
    </div>
    <script>(function(){{const c=document.getElementById('{chart_id}');if(c){{new Chart(c,{chart_config});}}}})();</script>
    """


def generate_kpi_html(viz_data: dict) -> str:
    kpis = viz_data.get("kpis", [])
    if not kpis:
        return ""
    cards = []
    for kpi in kpis:
        label = _escape(str(kpi.get("label", "")))
        value = _escape(str(kpi.get("value", "")))
        change = kpi.get("change", "")
        trend = kpi.get("trend", "neutral")
        icon = "↑" if trend == "up" else ("↓" if trend == "down" else "→")
        color = "#10b981" if trend == "up" else ("#ef4444" if trend == "down" else "#6b7280")
        change_html = (
            f'<div style="font-size:0.8rem;color:{color};">{icon} {_escape(str(change))}</div>'
            if change
            else ""
        )
        cards.append(
            f"""<div class="kpi-card" style="flex:1;min-width:140px;padding:1rem;background:white;border-radius:12px;border:1px solid #e2e8f0;">
            <div style="font-size:0.75rem;color:#64748b;font-weight:500;text-transform:uppercase;margin-bottom:0.5rem;">{label}</div>
            <div style="font-size:1.75rem;font-weight:700;color:#1e293b;">{value}</div>
            {change_html}</div>"""
        )
    return f'<div class="kpi-grid" style="display:flex;flex-wrap:wrap;gap:1rem;margin:1rem 0;">{"".join(cards)}</div>'


def convert_table_to_html(table_lines: list[str]) -> str:
    if len(table_lines) < 2:
        return "\n".join(table_lines)
    parts = ['<div class="prose"><table>']
    for i, line in enumerate(table_lines):
        cells = [c.strip() for c in line.split("|")[1:-1]]
        if i == 0:
            parts.append("<thead><tr>")
            parts.extend(f"<th>{c}</th>" for c in cells)
            parts.append("</tr></thead><tbody>")
        elif i == 1 and all(c.replace("-", "").replace(":", "") == "" for c in cells):
            continue
        else:
            parts.append("<tr>")
            parts.extend(f"<td>{c}</td>" for c in cells)
            parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def convert_markdown_to_html(text: str) -> str:
    text = _coerce_text(text)
    viz_data, text = extract_visualization(text)
    viz_html = ""
    if viz_data:
        chart_type = viz_data.get("type", "none")
        if chart_type in ("bar", "line", "pie"):
            chart_id = f"chart_{datetime.now(timezone.utc).timestamp()}".replace(".", "_")
            viz_html = generate_chart_html(viz_data, chart_id)
        elif chart_type == "kpi":
            viz_html = generate_kpi_html(viz_data)

    text = html.escape(text)

    def replace_code_block(match: re.Match) -> str:
        lang = match.group(1) or ""
        code = match.group(2)
        return f'<pre><code class="language-{lang}">{code}</code></pre>'

    text = re.sub(r"```(\w+)?\n(.*?)```", replace_code_block, text, flags=re.DOTALL)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", text)
    text = re.sub(r"^### (.+)$", r"<h3>\1</h3>", text, flags=re.MULTILINE)
    text = re.sub(r"^## (.+)$", r"<h2>\1</h2>", text, flags=re.MULTILINE)
    text = re.sub(r"^# (.+)$", r"<h1>\1</h1>", text, flags=re.MULTILINE)
    text = re.sub(r"^\d+\. (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)
    text = re.sub(r"^- (.+)$", r"<li>\1</li>", text, flags=re.MULTILINE)

    lines = text.split("\n")
    in_table = False
    in_list = False
    table_lines: list[str] = []
    result_lines: list[str] = []

    for line in lines:
        if "|" in line and line.strip().startswith("|"):
            if in_list:
                result_lines.append("</ul>")
                in_list = False
            if not in_table:
                in_table = True
                table_lines = []
            table_lines.append(line)
        elif line.strip().startswith("<li>"):
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
                result_lines.append("</ul>")
                in_list = False
            result_lines.append(line)

    if in_table:
        result_lines.append(convert_table_to_html(table_lines))
    if in_list:
        result_lines.append("</ul>")

    text = "\n".join(result_lines)
    text = text.replace("\n\n", "</p><p>").replace("\n", "<br>")
    return f'{viz_html}<div class="prose-content"><p>{text}</p></div>'


@router.get("/", response_class=HTMLResponse)
async def home(request: Request, session_id: str | None = None):
    if templates is None:
        return HTMLResponse(
            "<h1>BigQuery SaaS</h1><p>UI templates not found. Use the REST API at "
            "<a href='/docs'>/docs</a>.</p>",
            status_code=200,
        )

    ctx = _resolve_auth(request)
    store = _store(request)
    chat_svc = ChatService(store)

    recent_chats: list[dict[str, Any]] = []
    chat_history: list[dict[str, str]] = []
    tenant_label = "Unauthenticated"
    plan_label = "—"

    if ctx is not None:
        tenant_label = f"{ctx.tenant.name} ({ctx.tenant.slug})"
        plan_label = ctx.tenant.plan_id
        sessions = chat_svc.list_sessions(ctx.tenant_id, user_id=ctx.user_id)
        if session_id is None and sessions:
            session_id = sessions[0].id
        if session_id is None:
            session = chat_svc.create_session(
                ctx.tenant_id, ctx.user_id, title="New Session"
            )
            session_id = session.id
            sessions = [session] + sessions

        for s in sessions:
            preview = s.title or "New Session"
            try:
                msgs = chat_svc.list_messages(ctx.tenant_id, s.id)
                for m in reversed(msgs):
                    if getattr(m, "role", None) and str(m.role.value if hasattr(m.role, "value") else m.role) == "user":
                        preview = (m.content or "")[:30] + (
                            "..." if len(m.content or "") > 30 else ""
                        )
                        break
            except BqSaaSError:
                pass
            date_str = ""
            if getattr(s, "updated_at", None):
                date_str = s.updated_at.strftime("%b %d, %H:%M")
            recent_chats.append(
                {
                    "id": s.id,
                    "preview": preview,
                    "date": date_str,
                    "active": s.id == session_id,
                }
            )

        if session_id:
            try:
                for m in chat_svc.list_messages(ctx.tenant_id, session_id):
                    role = m.role.value if hasattr(m.role, "value") else str(m.role)
                    chat_history.append({"role": role, "content": m.content})
            except BqSaaSError:
                pass
    else:
        session_id = session_id or "anonymous"

    response = templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "session_id": session_id,
            "recent_chats": recent_chats,
            "chat_history": chat_history,
            "resources": [],
            "tenant_label": tenant_label,
            "plan_label": plan_label,
        },
    )
    demo_key = _demo_key(request)
    if demo_key and not request.cookies.get("api_key"):
        response.set_cookie("api_key", demo_key, max_age=86400 * 30, httponly=True)
    if session_id:
        response.set_cookie("session_id", session_id, max_age=86400 * 30)
    return response


@router.post("/chat", response_class=HTMLResponse)
async def chat(request: Request, message: str = Form(...)):
    if not message.strip():
        return HTMLResponse("")

    ctx = _resolve_auth(request)
    if ctx is None:
        return HTMLResponse(
            """<div class="message-wrapper ai"><div class="message-bubble">
            <p>Authentication required. Set the <code>api_key</code> cookie or
            <code>X-API-Key</code> header with a valid SaaS API key.</p>
            </div></div>"""
        )

    store = _store(request)
    chat_svc = ChatService(store)
    conn_svc = ConnectionService(store)
    session_id = request.cookies.get("session_id")

    if not session_id:
        session = chat_svc.create_session(ctx.tenant_id, ctx.user_id, title=message[:40])
        session_id = session.id
    else:
        try:
            chat_svc.get_session(ctx.tenant_id, session_id)
        except BqSaaSError:
            session = chat_svc.create_session(
                ctx.tenant_id, ctx.user_id, title=message[:40]
            )
            session_id = session.id

    chat_svc.append_message(ctx.tenant_id, session_id, "user", message.strip())

    history = [
        {
            "role": m.role.value if hasattr(m.role, "value") else str(m.role),
            "content": m.content,
        }
        for m in chat_svc.list_messages(ctx.tenant_id, session_id)
    ]

    bq_client = None
    project_id = ""
    dataset_id = ""
    try:
        conn = conn_svc.get_connection_for_workspace(
            ctx.tenant_id,
            chat_svc.get_session(ctx.tenant_id, session_id).workspace_id,
        )
        if conn is None:
            conns = conn_svc.list_connections(ctx.tenant_id, active_only=True)
            conn = conns[0] if conns else None
        if conn is not None:
            project_id = conn.project_id
            dataset_id = conn.dataset_id
            try:
                bq_client = get_client_from_connection(conn)
            except ClientError:
                bq_client = None
    except BqSaaSError:
        pass

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: process_with_gemini(
            message.strip(),
            history=history[:-1],
            client=bq_client,
            project_id=project_id,
            dataset_id=dataset_id,
        ),
    )

    # process_with_gemini returns {"status", "content", "tool_call", "tool_result"}
    assistant_text = _coerce_text(
        result.get("content", "") if isinstance(result, dict) else result
    )
    if not assistant_text:
        assistant_text = "I couldn't generate a response. Please try again."

    chat_svc.append_message(
        ctx.tenant_id,
        session_id,
        "assistant",
        assistant_text,
        metadata={
            "ai_status": (result or {}).get("status") if isinstance(result, dict) else None,
        },
    )
    response_html = convert_markdown_to_html(assistant_text)

    html_content = f"""
    <div class="message-wrapper user" id="msg-{datetime.now(timezone.utc).timestamp()}">
        <div class="message-bubble"><p>{_escape(message)}</p></div>
    </div>
    <div class="message-wrapper ai">
        <div class="message-avatar">
            <svg width="20" height="20" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
            </svg>
        </div>
        <div class="message-bubble"><div class="prose-content">{response_html}</div></div>
    </div>
    """
    resp = HTMLResponse(html_content)
    resp.set_cookie("session_id", session_id, max_age=86400 * 30)
    return resp


@router.post("/clear-chat", response_class=HTMLResponse)
async def clear_chat(request: Request):
    ctx = _resolve_auth(request)
    if ctx is None:
        return HTMLResponse(status_code=200, headers={"HX-Redirect": "/"})
    store = _store(request)
    session = ChatService(store).create_session(
        ctx.tenant_id, ctx.user_id, title="New Session"
    )
    return HTMLResponse(
        status_code=200,
        headers={"HX-Redirect": f"/?session_id={session.id}"},
    )


@router.post("/delete-session/{session_id}", response_class=HTMLResponse)
async def delete_session(request: Request, session_id: str):
    ctx = _resolve_auth(request)
    if ctx is not None:
        try:
            ChatService(_store(request)).delete_session(ctx.tenant_id, session_id)
        except (BqSaaSError, AttributeError):
            # soft-delete via store if service method differs
            try:
                from bqsaas.kinds import Kind

                _store(request).soft_delete(
                    Kind.CHAT_SESSION, session_id, tenant_id=ctx.tenant_id
                )
            except Exception:
                pass
    return HTMLResponse(status_code=200, headers={"HX-Redirect": "/"})


@router.get("/tools", response_class=HTMLResponse)
async def get_tools():
    from bqsaas.mcp.tools import TOOL_SPECS

    tools_html = ""
    for tool in TOOL_SPECS:
        name = tool.get("name", "")
        desc = tool.get("description", "")
        tools_html += f"""
        <div class="tool-card">
            <h3 class="tool-title">{_escape(name)}</h3>
            <p class="tool-desc">{_escape(desc)}</p>
        </div>
        """
    return HTMLResponse(tools_html)
