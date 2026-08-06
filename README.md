# BigQuery SaaS — Multi-tenant MCP Platform

Professional multi-tenant SaaS for natural-language analytics over Google BigQuery. Built around a typed **Kind system** (Kubernetes/GCP-style resource taxonomy), API-key auth, plan-based usage metering, an MCP tool server, and an HTMX chat UI powered by Gemini.

## Architecture

```
┌──────────────────┐     API key      ┌─────────────────────────────────────┐
│  Chat UI (HTMX)  │ ───────────────► │  FastAPI SaaS API  (/v1/*)          │
│  OpenAPI /docs   │                  │  Tenants · Workspaces · Sessions    │
└──────────────────┘                  │  Query · Kinds · Usage metering     │
                                      └──────────────┬──────────────────────┘
                                                     │
                      ┌──────────────────────────────┼──────────────────────┐
                      ▼                              ▼                      ▼
               Kind Registry                  Chat + Gemini           MCP Tools
               Tenant / User                  tool-calling            list_tables
               Workspace / ApiKey             BigQuery client         execute_query
               Connection / Session           per DataConnection      describe_table
               Subscription / Usage                                   …
```

### Kind system

Every platform object is a **Resource** with:

| Field | Description |
|-------|-------------|
| `kind` | Typed enum (`tenant`, `user`, `workspace`, `data_connection`, `api_key`, `chat_session`, `message`, `subscription`, `usage_event`, `query_job`) |
| `id` | `{kind}_{ulid\|uuid}` e.g. `tenant_01HXYZ…` |
| `tenant_id` | Isolation boundary (null only for `tenant` itself) |
| `status` | `active` · `suspended` · `deleted` |
| `labels` / `annotations` | Free-form metadata |
| `created_at` / `updated_at` | UTC timestamps |

`KindRegistry` maps each kind to its Pydantic model and schema metadata.

### SaaS features

- **Multi-tenancy** — strict tenant isolation in `MemoryStore`
- **API keys** — `bqs_{env}_{secret}`; only SHA-256 hash stored
- **Scopes** — `chat:read|write`, `query:execute`, `admin:read|write`, `*`
- **Plans** — Free / Pro / Enterprise with daily query + resource limits
- **Usage metering** — daily counters with automatic period reset
- **Demo bootstrap** — local dev tenant + API key printed on startup

## Quick start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- Gemini API key ([Google AI Studio](https://aistudio.google.com/app/apikey))
- Optional: BigQuery service account JSON for live queries

### Install

```bash
uv sync --all-groups
```

### Configure

```powershell
$env:GEMINI_API_KEY = "your-key"
$env:GCP_PROJECT_ID = "your-gcp-project"   # optional
$env:DATASET_ID = "school_data"            # optional
```

Place `service-account.json` in the repo root (gitignored) for BigQuery access.

### Run the SaaS API + chat UI

```bash
uv run uvicorn bqsaas.api.app:app --app-dir src --reload --host 0.0.0.0 --port 8000
```

On startup the demo tenant is bootstrapped and a **one-time API key** is printed:

```
============================================================
  BigQuery SaaS demo tenant bootstrapped
  API key: bqs_dev_…
  Use: Authorization: Bearer <api_key>
  Docs: http://localhost:8000/docs
============================================================
```

| URL | Purpose |
|-----|---------|
| http://localhost:8000/ | Chat UI (auto-uses demo API key cookie) |
| http://localhost:8000/docs | OpenAPI |
| http://localhost:8000/health | Liveness |
| http://localhost:8000/v1/kinds | Kind registry |

### Run the MCP server (stdio)

```bash
uv run python main.py
# or
uv run python -m bqsaas
```

## REST API (selected)

Auth: `Authorization: Bearer bqs_…` or `X-API-Key: bqs_…`

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/me` | Current user, tenant, plan, usage |
| `GET` | `/v1/kinds` | Registered resource kinds |
| `GET/POST` | `/v1/workspaces` | Workspace CRUD |
| `GET` | `/v1/connections` | BigQuery connections |
| `GET/POST` | `/v1/sessions` | Chat sessions |
| `POST` | `/v1/sessions/{id}/messages` | AI chat turn |
| `POST` | `/v1/query` | Execute SQL (quota-enforced) |
| `POST` | `/v1/auth/api-keys` | Mint API key (admin) |

## Project layout

```
bigquery-mcp-server/
├── main.py                 # MCP stdio entrypoint
├── pyproject.toml
├── client/                 # HTMX templates + static assets
│   ├── templates/
│   ├── static/
│   └── app.py              # thin re-export of SaaS app
├── src/bqsaas/
│   ├── kinds/              # Kind enum, Resource, KindRegistry
│   ├── models/             # Tenant, User, Workspace, …
│   ├── storage/            # MemoryStore (tenant-isolated)
│   ├── auth/               # API keys, scopes, AuthContext
│   ├── billing/            # Plans + UsageMeter
│   ├── services/           # Bootstrap, chat, tenant, connection
│   ├── mcp/                # Pure BigQuery tools + FastMCP server
│   ├── ai/                 # Gemini tool-calling pipeline
│   └── api/                # FastAPI app + routes + web UI
└── tests/                  # pytest suite (mocked BQ/Gemini)
```

## Tests

```bash
uv run pytest tests/ -q
```

All external services (BigQuery, Gemini) are mocked. Tests cover kinds, storage isolation, auth, billing quotas, MCP tool validation, services, and HTTP API smoke paths.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Required for AI chat |
| `GCP_PROJECT_ID` | `practice-project-481414` | Default BQ project |
| `DATASET_ID` | `school_data` | Default dataset |
| `SERVICE_ACCOUNT_PATH` | `service-account.json` | BQ credentials path |
| `APP_ENV` | `development` | `production` tightens CORS |
| `SECRET_KEY` | dev default | Token signing secret |

## Plans

| Plan | Daily queries | Workspaces | Connections | Users |
|------|---------------|------------|-------------|-------|
| Free | 100 | 1 | 1 | 3 |
| Pro | 10,000 | 10 | 10 | 50 |
| Enterprise | 1,000,000 | high | high | high |

## Security notes

- Never commit `service-account.json` or raw API keys
- API keys are hashed at rest (SHA-256); raw value shown only once
- Table identifiers are validated before SQL construction
- Query jobs enforce max bytes billed and row caps

## License

Proprietary / private — adjust as needed for your organization.
