"""
Chat UI entrypoint — thin wrapper around the multi-tenant BigQuery SaaS API.

Prefer running the full platform:

    uv run uvicorn bqsaas.api.app:app --app-dir src --reload --host 0.0.0.0 --port 8000

This module remains for backward compatibility with:

    uv run uvicorn client.app:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import os
import sys

# Ensure src/ is importable when launched from client/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from bqsaas.api.app import app, create_app  # noqa: E402

__all__ = ["app", "create_app", "main"]


def main() -> None:
    import uvicorn

    uvicorn.run(
        "client.app:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        reload=os.environ.get("APP_ENV", "development") != "production",
    )


if __name__ == "__main__":
    main()
