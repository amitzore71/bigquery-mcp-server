"""Multi-tenant BigQuery MCP tools and server.

Server imports are lazy so tools can be used/tested without FastMCP installed.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["create_mcp", "mcp", "run", "tools"]


def __getattr__(name: str) -> Any:
    # Use import_module (not `from bqsaas.mcp import ...`) so submodule loads
    # do not re-enter this __getattr__ and recurse.
    if name in {"create_mcp", "mcp", "run"}:
        _server = import_module(".server", __name__)
        return getattr(_server, name)
    if name == "tools":
        return import_module(".tools", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
