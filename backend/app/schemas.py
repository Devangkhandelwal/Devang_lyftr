"""Request and response models.

Pydantic does the validation, so a malformed request is rejected with a 422
before any database or provider work happens.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

MAX_MESSAGE_LENGTH = 4000
MAX_SESSION_ID_LENGTH = 128


class ChatRequest(BaseModel):
    """Body of POST /chat."""

    session_id: str
    message: str
    system_prompt: str | None = None

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, value: str) -> str:
        session_id = value.strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        if len(session_id) > MAX_SESSION_ID_LENGTH:
            raise ValueError(
                f"session_id must be at most {MAX_SESSION_ID_LENGTH} characters"
            )
        return session_id

    @field_validator("message")
    @classmethod
    def _validate_message(cls, value: str) -> str:
        # The length limits apply to the trimmed message, so a whitespace
        # only message is rejected just like an empty one.
        message = value.strip()
        if not message:
            raise ValueError("message must not be empty")
        if len(message) > MAX_MESSAGE_LENGTH:
            raise ValueError(
                f"message must be at most {MAX_MESSAGE_LENGTH} characters"
            )
        return message

    @field_validator("system_prompt")
    @classmethod
    def _validate_system_prompt(cls, value: str | None) -> str | None:
        if value is None:
            return None
        system_prompt = value.strip()
        # A blank system prompt is treated as "not provided" so it cannot
        # silently replace the configured default with an empty string.
        return system_prompt or None


class ResponseMetadata(BaseModel):
    """The metadata block returned with every assistant reply."""

    resolution_source: Literal["LIVE_INFERENCE", "LLM_CACHE"]
    model: str
    latency_ms: int
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int


class ChatResponse(BaseModel):
    """Successful response of POST /chat."""

    session_id: str
    reply: str
    metadata: ResponseMetadata


class HistoryMessage(BaseModel):
    """One stored message in the history response."""

    role: str
    content: str
    metadata: dict[str, Any] | None = None


class HistoryResponse(BaseModel):
    """Successful response of GET /sessions/{session_id}/messages."""

    session_id: str
    messages: list[HistoryMessage]


class HealthResponse(BaseModel):
    """Successful response of GET /health."""

    status: str = Field(default="ok")
    keys_configured: int
    cache_enabled: bool
