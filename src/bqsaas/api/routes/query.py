"""Direct SQL execution with quota + tenant connection."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from bqsaas.auth.dependencies import AuthContext
from bqsaas.auth.deps import domain_error_to_http, get_auth_context, get_store
from bqsaas.billing.meter import EVENT_QUERY, UsageMeter
from bqsaas.errors import BqSaaSError, ValidationError
from bqsaas.mcp.bigquery_client import ClientError, get_client_from_connection
from bqsaas.mcp.tools import execute_query
from bqsaas.services.connection_service import ConnectionService
from bqsaas.storage.memory import MemoryStore

router = APIRouter(prefix="/v1", tags=["query"])


class QueryRequest(BaseModel):
    sql: str = Field(..., min_length=1)
    connection_id: Optional[str] = None
    max_rows: Optional[int] = None


@router.post("/query")
async def run_query(
    body: QueryRequest,
    ctx: AuthContext = Depends(get_auth_context),
    store: MemoryStore = Depends(get_store),
) -> dict:
    """
    Execute SQL against the tenant's BigQuery connection.

    Body: ``{ "sql": "...", "connection_id": optional, "max_rows": optional }``
    """
    sql = (body.sql or "").strip()
    if not sql:
        raise domain_error_to_http(
            ValidationError("sql must not be empty", field="sql")
        )

    meter = UsageMeter(store)
    conn_svc = ConnectionService(store)

    try:
        meter.check_quota(ctx.tenant_id, EVENT_QUERY, 1)
    except BqSaaSError as e:
        raise domain_error_to_http(e) from e

    try:
        if body.connection_id:
            conn = conn_svc.get_connection(ctx.tenant_id, body.connection_id)
        else:
            conns = conn_svc.list_connections(ctx.tenant_id, active_only=True)
            if not conns:
                raise ValidationError(
                    "No data connection configured for tenant",
                    field="connection_id",
                )
            conn = conns[0]
    except BqSaaSError as e:
        raise domain_error_to_http(e) from e

    try:
        client = get_client_from_connection(conn)
    except ClientError as e:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=503, detail=f"BigQuery client unavailable: {e.message}"
        ) from e

    plan = meter.get_plan_for_tenant(ctx.tenant_id)
    # Use plan daily limit as soft signal; byte cap from settings
    from bqsaas.config import get_settings

    settings = get_settings()
    result = execute_query(
        client,
        conn.project_id,
        conn.dataset_id,
        sql,
        max_bytes_billed=settings.max_query_bytes_billed,
        max_rows=body.max_rows,
    )

    if result.get("status") == "success":
        try:
            meter.check_and_consume(
                ctx.tenant_id,
                EVENT_QUERY,
                1,
                user_id=ctx.user_id,
            )
        except BqSaaSError as e:
            raise domain_error_to_http(e) from e

    usage = meter.usage_snapshot(ctx.tenant_id)
    return {
        **result,
        "connection_id": conn.id,
        "quota": {
            "used": usage.get("queries_used_today"),
            "limit": usage.get("daily_query_limit"),
            "plan": plan.id,
        },
    }
