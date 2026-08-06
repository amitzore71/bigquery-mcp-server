"""Chat session and message storage (no LLM/Gemini — persistence only)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from bqsaas.billing.meter import UsageMeter
from bqsaas.errors import ForbiddenError, NotFoundError, ValidationError
from bqsaas.kinds import Kind, generate_id
from bqsaas.models import ChatSession, Message, MessageRole

if TYPE_CHECKING:
    from bqsaas.storage import MemoryStore

VALID_ROLES = frozenset({"user", "assistant", "system", "tool"})
MessageRoleStr = Literal["user", "assistant", "system", "tool"]

# Alias
ChatMessage = Message


class ChatService:
    """Session CRUD and message append for multi-tenant chat history."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._meter = UsageMeter(store)

    def create_session(
        self,
        tenant_id: str,
        user_id: str,
        *,
        workspace_id: str | None = None,
        title: str = "New chat",
        metadata: dict[str, Any] | None = None,
    ) -> ChatSession:
        """Create a new chat session owned by ``tenant_id`` / ``user_id``."""
        self._assert_tenant_exists(tenant_id)
        self._assert_user_in_tenant(tenant_id, user_id)

        if not workspace_id:
            # pick first workspace for tenant
            workspaces = self._store.list_by_tenant(Kind.WORKSPACE, tenant_id)
            if not workspaces:
                raise ValidationError(
                    "workspace_id is required (tenant has no workspaces)",
                    field="workspace_id",
                )
            workspace_id = workspaces[0].id
        else:
            self._assert_workspace_in_tenant(tenant_id, workspace_id)

        all_sessions = self.list_sessions(tenant_id)
        self._meter.check_resource_limit(tenant_id, "chat_sessions", len(all_sessions))

        session = ChatSession(
            id=generate_id(Kind.CHAT_SESSION),
            tenant_id=tenant_id,
            user_id=user_id,
            workspace_id=workspace_id,
            title=(title or "New chat").strip() or "New chat",
            message_count=0,
        )
        if metadata:
            session.annotations = {k: str(v) for k, v in metadata.items()}
        self._store.save(session)
        return session

    def get_session(self, tenant_id: str, session_id: str) -> ChatSession:
        session = self._store.get(Kind.CHAT_SESSION, session_id)
        if session is None:
            raise NotFoundError(
                "Chat session not found",
                resource="chat_session",
                resource_id=session_id,
            )
        if getattr(session, "tenant_id", None) != tenant_id:
            raise ForbiddenError("Chat session does not belong to tenant")
        return session  # type: ignore[return-value]

    def list_sessions(
        self,
        tenant_id: str,
        *,
        user_id: str | None = None,
        workspace_id: str | None = None,
        include_archived: bool = False,
    ) -> list[ChatSession]:
        self._assert_tenant_exists(tenant_id)
        sessions = self._store.list_by_tenant(Kind.CHAT_SESSION, tenant_id)
        result: list[ChatSession] = []
        for s in sessions:
            if user_id is not None and getattr(s, "user_id", None) != user_id:
                continue
            if workspace_id is not None and getattr(s, "workspace_id", None) != workspace_id:
                continue
            result.append(s)  # type: ignore[arg-type]
        result.sort(
            key=lambda s: getattr(s, "updated_at", None)
            or getattr(s, "created_at", None)
            or 0,
            reverse=True,
        )
        return result

    def update_session(
        self,
        tenant_id: str,
        session_id: str,
        *,
        title: str | None = None,
        is_archived: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatSession:
        session = self.get_session(tenant_id, session_id)
        if title is not None:
            cleaned = title.strip()
            if not cleaned:
                raise ValidationError("Title cannot be empty", field="title")
            session.title = cleaned
        if is_archived is not None:
            from bqsaas.kinds.base import ResourceStatus

            session.status = (
                ResourceStatus.SUSPENDED if is_archived else ResourceStatus.ACTIVE
            )
        if metadata is not None:
            session.annotations.update({k: str(v) for k, v in metadata.items()})
        session.touch()
        self._store.save(session)
        return session

    def delete_session(self, tenant_id: str, session_id: str) -> None:
        """Delete a session and its messages."""
        session = self.get_session(tenant_id, session_id)
        for msg in self.list_messages(tenant_id, session_id):
            self._store.delete(Kind.MESSAGE, msg.id)
        self._store.delete(Kind.CHAT_SESSION, session.id)

    def append_message(
        self,
        tenant_id: str,
        session_id: str,
        role: str,
        content: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Append a message to a session. Does not call Gemini."""
        session = self.get_session(tenant_id, session_id)

        if role not in VALID_ROLES:
            raise ValidationError(
                f"Invalid role '{role}'",
                field="role",
                details={"allowed": sorted(VALID_ROLES)},
            )
        if content is None or (isinstance(content, str) and content == ""):
            raise ValidationError("Message content is required", field="content")

        message = Message(
            id=generate_id(Kind.MESSAGE),
            tenant_id=tenant_id,
            session_id=session.id,
            role=MessageRole(role),
            content=content,
        )
        if metadata:
            message.annotations = {k: str(v) for k, v in metadata.items()}
        self._store.save(message)

        session.message_count = int(getattr(session, "message_count", 0) or 0) + 1
        if (
            role == "user"
            and getattr(session, "title", "New chat") in ("New chat", "", None)
            and isinstance(content, str)
        ):
            session.title = content.strip()[:80] or "New chat"
        session.touch()
        self._store.save(session)
        return message

    def list_messages(
        self,
        tenant_id: str,
        session_id: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Message]:
        self.get_session(tenant_id, session_id)
        messages = self._store.list_messages_for_session(session_id)
        filtered: list[Message] = []
        for m in messages:
            msg_tenant = getattr(m, "tenant_id", None)
            if msg_tenant is not None and msg_tenant != tenant_id:
                continue
            filtered.append(m)  # type: ignore[arg-type]

        filtered.sort(key=lambda m: getattr(m, "created_at", None) or 0)
        if offset:
            filtered = filtered[offset:]
        if limit is not None:
            if limit < 0:
                raise ValidationError("limit must be >= 0", field="limit")
            filtered = filtered[:limit]
        return filtered

    def get_message(self, tenant_id: str, message_id: str) -> Message:
        message = self._store.get(Kind.MESSAGE, message_id)
        if message is None:
            raise NotFoundError(
                "Chat message not found",
                resource="chat_message",
                resource_id=message_id,
            )
        msg_tenant = getattr(message, "tenant_id", None)
        if msg_tenant is not None and msg_tenant != tenant_id:
            raise ForbiddenError("Chat message does not belong to tenant")
        if msg_tenant is None:
            self.get_session(tenant_id, message.session_id)  # type: ignore[attr-defined]
        return message  # type: ignore[return-value]

    def _assert_tenant_exists(self, tenant_id: str) -> None:
        tenant = self._store.get(Kind.TENANT, tenant_id)
        if tenant is None:
            raise NotFoundError(
                "Tenant not found", resource="tenant", resource_id=tenant_id
            )

    def _assert_user_in_tenant(self, tenant_id: str, user_id: str) -> None:
        user = self._store.get(Kind.USER, user_id)
        if user is None:
            raise NotFoundError("User not found", resource="user", resource_id=user_id)
        if getattr(user, "tenant_id", None) != tenant_id:
            raise ForbiddenError("User does not belong to tenant")

    def _assert_workspace_in_tenant(self, tenant_id: str, workspace_id: str) -> None:
        workspace = self._store.get(Kind.WORKSPACE, workspace_id)
        if workspace is None:
            raise NotFoundError(
                "Workspace not found",
                resource="workspace",
                resource_id=workspace_id,
            )
        if getattr(workspace, "tenant_id", None) != tenant_id:
            raise ForbiddenError("Workspace does not belong to tenant")
