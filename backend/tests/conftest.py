"""Shared test fixtures.

The environment is configured before the application is imported, so the
app under test runs against a throwaway SQLite file and an in memory cache.
No test performs a real network call: litellm.acompletion is always mocked.
"""

import asyncio
import os
import tempfile

# The settings object is built when app.main is imported, so the environment
# has to be in place first.
_TEST_DB_DIR = tempfile.mkdtemp(prefix="llm-chat-tests-")
os.environ.update(
    {
        "LLM_API_KEYS": "test-key-one,test-key-two",
        "LLM_MODEL": "gemini/gemini-3.1-flash-lite",
        "DATABASE_URL": "sqlite:///" + _TEST_DB_DIR.replace("\\", "/") + "/test.db",
        "ALLOWED_ORIGINS": "http://localhost:3000",
        "ENABLE_FAILURE_SIMULATION": "true",
        "DEFAULT_SYSTEM_PROMPT": "You are a helpful, concise assistant.",
        "CACHE_TYPE": "local",
    }
)

import litellm  # noqa: E402
import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from litellm.types.utils import Choices, Message, ModelResponse, Usage  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.main import app  # noqa: E402

TEST_MODEL = "gemini/gemini-3.1-flash-lite"

# A live provider call is slower than a cache lookup. The mock sleeps for
# this long so the "cache hit is faster" assertion measures something real.
LIVE_CALL_DELAY_SECONDS = 0.05


def make_response(
    text: str = "The capital of France is Paris.",
    prompt_tokens: int = 42,
    completion_tokens: int = 9,
) -> ModelResponse:
    """Build a completion response shaped like a real provider reply."""
    return ModelResponse(
        id="test-response",
        model=TEST_MODEL,
        object="chat.completion",
        created=1,
        choices=[
            Choices(
                finish_reason="stop",
                index=0,
                message=Message(content=text, role="assistant"),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


class FakeCompletion:
    """Stand in for litellm.acompletion.

    Records the kwargs of every call and returns the queued responses, or
    raises them when a queued item is an exception.
    """

    def __init__(self, *results: object, delay: float = LIVE_CALL_DELAY_SECONDS) -> None:
        self.results = list(results) or [make_response()]
        self.delay = delay
        self.calls: list[dict] = []

    async def __call__(self, **kwargs: object) -> ModelResponse:
        self.calls.append(kwargs)
        # Simulate the latency of a real network round trip.
        if self.delay:
            await asyncio.sleep(self.delay)
        index = min(len(self.calls) - 1, len(self.results) - 1)
        result = self.results[index]
        if isinstance(result, BaseException):
            raise result
        return result  # type: ignore[return-value]

    @property
    def call_count(self) -> int:
        return len(self.calls)

    def messages_of_call(self, index: int) -> list[dict]:
        """The message list passed to the n-th call."""
        return self.calls[index]["messages"]

    def api_key_of_call(self, index: int) -> str:
        return self.calls[index]["api_key"]


@pytest.fixture
def fake_completion(monkeypatch):
    """Return a factory that installs a FakeCompletion over litellm."""

    def install(*results: object, delay: float = LIVE_CALL_DELAY_SECONDS) -> FakeCompletion:
        fake = FakeCompletion(*results, delay=delay)
        monkeypatch.setattr(litellm, "acompletion", fake)
        return fake

    return install


@pytest.fixture(scope="session")
def client():
    """A TestClient that runs the application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(autouse=True)
def clean_state(client):
    """Give every test an empty database and an empty response cache."""
    database = app.state.db
    with database.connect() as connection:
        connection.execute("DELETE FROM messages")
        connection.execute("DELETE FROM sessions")
        connection.commit()
    litellm.cache = litellm.Cache(type="local", max_messages=None)
    yield


@pytest.fixture
def settings():
    """The live settings object used by the application."""
    return get_settings()
