"""Multi-tenant BigQuery MCP tools and server.

Server imports are lazy so tools can be used/tested without FastMCP installed.
"""

from __future__ import annotations

from typing import Any

__all__ = ["create_mcp", "mcp", "run", "tools"]


def __getattr__(name: str) -> Any:
    if name in {"create_mcp", "mcp", "run"}:
        from bqsaas.mcp import server as _server

        return getattr(_server, name)
    if name == "tools":
        from bqsaas.mcp import tools as _tools

        return _tools
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
