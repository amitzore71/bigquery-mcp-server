"""Chat session and message domain models."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from bqsaas.kinds.base import Kind, Resource


class MessageRole(str, Enum):
    """Role of a message author in a chat session."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class ToolCall(BaseModel):
    """Structured tool invocation recorded on an assistant/tool message."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str | None = None


class ChatSession(Resource):
    """AI chat session scoped to a tenant workspace and user."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: Kind = Kind.CHAT_SESSION

    workspace_id: str = Field(..., min_length=1)
    user_id: str = Field(..., min_length=1)
    title: str = Field(default="New chat", min_length=1, max_length=500)
    message_count: int = Field(default=0, ge=0)

    @field_validator("workspace_id", "user_id")
    @classmethod
    def _strip_ids(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("field must not be empty")
        return stripped

    @field_validator("title")
    @classmethod
    def _strip_title(cls, value: str) -> str:
        title = value.strip()
        return title if title else "New chat"

    def increment_message_count(self, by: int = 1) -> ChatSession:
        """Increase message_count and bump updated_at."""
        if by < 0:
            raise ValueError("by must be non-negative")
        self.message_count += by
        return self.touch()


class Message(Resource):
    """Single message belonging to a chat session."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    kind: Kind = Kind.MESSAGE

    session_id: str = Field(..., min_length=1)
    role: MessageRole
    content: str = Field(default="", max_length=1_000_000)
    tool_calls: list[ToolCall] | None = Field(
        default=None,
        description="Optional tool invocations associated with this message",
    )
    parent_message_id: str | None = Field(
        default=None,
        description="Optional parent message for threading/tool replies",
    )
    token_count: int | None = Field(default=None, ge=0)

    @field_validator("session_id")
    @classmethod
    def _strip_session_id(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("session_id must not be empty")
        return stripped

    @field_validator("role", mode="before")
    @classmethod
    def _coerce_role(cls, value: object) -> object:
        if isinstance(value, str):
            return value.lower()
        return value

    @field_validator("content", mode="before")
    @classmethod
    def _coerce_content(cls, value: object) -> str:
        if value is None:
            return ""
        return str(value)
