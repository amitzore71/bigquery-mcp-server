"""Chat sessions, messages, and AI pipeline."""

from __future__ import annotations

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field

from bqsaas.ai.gemini import process_with_gemini
from bqsaas.auth.dependencies import AuthContext
from bqsaas.auth.deps import domain_error_to_http, get_auth_context, get_store
from bqsaas.billing.meter import EVENT_QUERY, UsageMeter
from bqsaas.errors import BqSaaSError
from bqsaas.mcp.bigquery_client import ClientError, get_client_from_connection
from bqsaas.services.chat_service import ChatService
from bqsaas.services.connection_service import ConnectionService
from bqsaas.storage.memory import MemoryStore

router = APIRouter(prefix="/v1", tags=["chat"])


class CreateSessionRequest(BaseModel):
    title: str = "New chat"
    workspace_id: Optional[str] = None


class PostMessageRequest(BaseModel):
    content: str = Field(..., min_length=1)
    connection_id: Optional[str] = None


@router.get("/sessions")
async def list_sessions(
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> list[dict]:
    svc = ChatService(store)
    return [
        s.model_dump(mode="json")
        for s in svc.list_sessions(ctx.tenant_id, user_id=ctx.user_id)
    ]


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(
    body: CreateSessionRequest,
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> dict:
    svc = ChatService(store)
    try:
        session = svc.create_session(
            ctx.tenant_id,
            ctx.user_id,
            workspace_id=body.workspace_id,
            title=body.title,
        )
        return session.model_dump(mode="json")
    except BqSaaSError as e:
        raise domain_error_to_http(e) from e


@router.get("/sessions/{session_id}/messages")
async def list_messages(
    session_id: str,
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> list[dict]:
    svc = ChatService(store)
    try:
        return [
            m.model_dump(mode="json")
            for m in svc.list_messages(ctx.tenant_id, session_id)
        ]
    except BqSaaSError as e:
        raise domain_error_to_http(e) from e


@router.post("/sessions/{session_id}/messages")
async def post_message(
    session_id: str,
    body: PostMessageRequest,
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> dict:
    """Post a user message, run the AI pipeline, store and return the assistant reply."""
    chat_svc = ChatService(store)
    conn_svc = ConnectionService(store)
    meter = UsageMeter(store)

    try:
        user_msg = chat_svc.append_message(
            ctx.tenant_id, session_id, "user", body.content.strip()
        )
    except BqSaaSError as e:
        raise domain_error_to_http(e) from e

    session = chat_svc.get_session(ctx.tenant_id, session_id)
    bq_client = None
    project_id = ""
    dataset_id = ""

    try:
        if body.connection_id:
            conn = conn_svc.get_connection(ctx.tenant_id, body.connection_id)
        else:
            conn = conn_svc.get_connection_for_workspace(
                ctx.tenant_id, session.workspace_id
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

    history = [
        {"role": m.role.value if hasattr(m.role, "value") else str(m.role), "content": m.content}
        for m in chat_svc.list_messages(ctx.tenant_id, session_id)
        if (m.role.value if hasattr(m.role, "value") else str(m.role))
        in ("user", "assistant")
    ][:-1]

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        lambda: process_with_gemini(
            body.content.strip(),
            history=history,
            client=bq_client,
            project_id=project_id,
            dataset_id=dataset_id,
        ),
    )

    if result.get("tool_result") is not None:
        try:
            meter.check_and_consume(
                ctx.tenant_id,
                EVENT_QUERY,
                1,
                user_id=ctx.user_id,
                workspace_id=session.workspace_id,
            )
        except BqSaaSError as e:
            raise domain_error_to_http(e) from e

    assistant = chat_svc.append_message(
        ctx.tenant_id,
        session_id,
        "assistant",
        result.get("content", ""),
        metadata={
            "ai_status": result.get("status"),
            "tool_status": str(
                (result.get("tool_result") or {}).get("status", "")
            ),
        },
    )

    return {
        "user_message": user_msg.model_dump(mode="json"),
        "assistant_message": assistant.model_dump(mode="json"),
        "tool_call": result.get("tool_call"),
        "tool_result": result.get("tool_result"),
        "status": result.get("status"),
    }
