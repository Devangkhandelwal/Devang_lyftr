"""FastAPI application: health, chat and session history endpoints."""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import litellm
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import Settings, get_settings
from app.db import Database
from app.llm import (
    AllProvidersUnavailable,
    configure_cache,
    generate,
    parse_simulation_header,
)
from app.schemas import ChatRequest, ChatResponse, HealthResponse, HistoryResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LiteLLM is chatty by default; keep its debug output off so no request
# payload (and therefore no API key) can end up in the logs.
litellm.suppress_debug_info = True


def create_app() -> FastAPI:
    """Build the FastAPI application."""
    settings = get_settings()
    database = Database(settings.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Create the SQLite file, its parent directory and the schema, then
        # install the LiteLLM response cache.
        database.init_schema()
        disk_cache_dir = str(database.path.parent / "llm_cache")
        configure_cache(settings, disk_cache_dir)
        logger.info(
            "Startup complete: %s keys configured, database at %s",
            len(settings.api_keys),
            database.path,
        )
        yield

    app = FastAPI(
        title="LLM Chatbot with Key Failover and Response Caching",
        version="1.0.0",
        lifespan=lifespan,
    )

    allowed_origins = settings.origins
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials="*" not in allowed_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.db = database

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exception: Exception
    ) -> JSONResponse:
        """Last resort handler, so a client never sees a stack trace."""
        logger.exception("Unhandled error on %s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    # ------------------------------- health -------------------------------

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        """Report readiness: the database answers and keys are loaded."""
        current_settings = get_settings()
        try:
            database.ping()
        except Exception:
            logger.exception("Health check failed: database unreachable")
            raise HTTPException(status_code=503, detail="database unavailable")

        if not current_settings.api_keys:
            raise HTTPException(status_code=503, detail="no api keys configured")

        return HealthResponse(
            status="ok",
            keys_configured=len(current_settings.api_keys),
            cache_enabled=litellm.cache is not None,
        )

    # -------------------------------- chat --------------------------------

    @app.post("/chat", response_model=ChatResponse)
    async def chat(
        payload: ChatRequest,
        x_simulate_failure: str | None = Header(default=None),
    ) -> ChatResponse:
        """Answer a chat message and append the exchange to the history."""
        current_settings: Settings = get_settings()
        started_at = time.perf_counter()

        # The simulation header is honoured only when the feature is enabled.
        simulation: dict[str, int] = {}
        if current_settings.enable_failure_simulation:
            simulation = parse_simulation_header(x_simulate_failure)

        # Load the session system prompt and the stored history. An unknown
        # session id is simply a new conversation.
        stored_prompt = database.get_system_prompt(payload.session_id)
        effective_prompt = (
            payload.system_prompt
            or stored_prompt
            or current_settings.default_system_prompt
        )
        history = database.get_messages(payload.session_id)

        # [system] + history + [new user message]
        messages: list[dict[str, str]] = [
            {"role": "system", "content": effective_prompt}
        ]
        for item in history:
            messages.append({"role": item["role"], "content": item["content"]})
        messages.append({"role": "user", "content": payload.message})

        try:
            result = await generate(messages, simulation, current_settings)
        except AllProvidersUnavailable:
            # Every key failed. Nothing is written to the history.
            logger.warning("All providers unavailable for session %s", payload.session_id)
            raise HTTPException(status_code=503, detail="all providers unavailable")
        except HTTPException:
            raise
        except Exception:
            logger.exception("Unexpected generation failure")
            raise HTTPException(status_code=500, detail="internal error")

        latency_ms = int((time.perf_counter() - started_at) * 1000)

        metadata: dict[str, Any] = {
            "resolution_source": result.resolution_source,
            "model": result.model,
            "latency_ms": latency_ms,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "key_id": result.key_id,
            "fallback_attempts": result.fallback_attempts,
        }

        # Only on success: store the user row and the assistant row together.
        database.save_exchange(
            session_id=payload.session_id,
            user_content=payload.message,
            assistant_content=result.reply,
            assistant_metadata=metadata,
            system_prompt=payload.system_prompt,
        )

        return ChatResponse(
            session_id=payload.session_id,
            reply=result.reply,
            metadata=metadata,
        )

    # ------------------------------- history ------------------------------

    @app.get(
        "/sessions/{session_id}/messages",
        response_model=HistoryResponse,
        response_model_exclude_none=True,
    )
    async def session_messages(session_id: str) -> HistoryResponse:
        """Return the stored conversation, oldest first, without the prompt."""
        if not database.session_exists(session_id):
            raise HTTPException(status_code=404, detail="session not found")

        messages = database.get_messages(session_id)
        return HistoryResponse(session_id=session_id, messages=messages)

    return app


app = create_app()
