# Kind System

The platform uses a **typed resource taxonomy** (similar to Kubernetes `kind` or GCP resource types). Every persisted object is a `Resource` with a known `Kind`.

## Built-in kinds

| Kind | Purpose | Tenant-scoped? |
|------|---------|----------------|
| `tenant` | Organization root | No (is the root) |
| `user` | Member of a tenant | Yes |
| `workspace` | Logical project area | Yes |
| `data_connection` | BigQuery project/dataset + credentials ref | Yes |
| `api_key` | Hashed API credential | Yes |
| `chat_session` | Conversation container | Yes |
| `message` | Chat message | Yes |
| `subscription` | Plan + usage counters | Yes |
| `usage_event` | Metered event | Yes |
| `query_job` | Reserved for async query jobs | Yes |

## ID format

```
{kind}_{raw}
```

Examples:

- `tenant_01J8ABC…` (ULID when `python-ulid` is installed)
- `user_a1b2c3d4e5f6…` (UUID4 hex fallback)

Helpers:

```python
from bqsaas import Kind, generate_resource_id, ResourceRef

rid = generate_resource_id(Kind.WORKSPACE)  # workspace_…
ref = ResourceRef.parse(rid)
```

## Registry

```python
from bqsaas import default_registry, Kind

meta = default_registry.get(Kind.TENANT)
print(meta.model_class, meta.description)
for item in default_registry.list_kinds():
    print(item["kind"], item["description"])
```

HTTP: `GET /v1/kinds` and `GET /v1/kinds/{kind}`.

## Isolation rules

- Child resources **must** set `tenant_id` to the owning tenant’s id.
- `MemoryStore.list()` requires `tenant_id` unless `platform_admin=True` or the kind is `tenant`.
- Soft-delete sets `status=deleted` and hides the resource from default reads.

## Extending

1. Add a value to `Kind`.
2. Define a Pydantic model subclassing `Resource` with fixed `kind=Kind.YOUR_KIND`.
3. Register it in `register_builtin_kinds()`.
4. Add repository helpers if needed.
