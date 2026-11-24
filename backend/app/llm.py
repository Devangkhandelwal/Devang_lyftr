"""LLM gateway: response caching, key pool failover and failure simulation.

All provider traffic goes through generate(). The function first consults the
LiteLLM cache and, on a miss, walks the configured API keys in order until
one of them answers.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm
from litellm.caching.base_cache import BaseCache

from app.config import Settings

logger = logging.getLogger(__name__)

# Statuses the X-Simulate-Failure header is allowed to request.
ALLOWED_SIMULATION_STATUSES = {429, 500, 502, 503}

# Wildcard key id meaning "every key in the pool".
WILDCARD_KEY = "*"

RESOLUTION_LIVE = "LIVE_INFERENCE"
RESOLUTION_CACHE = "LLM_CACHE"

# Provider errors that should move the request on to the next API key.
# A rate limit, an exhausted quota, a provider side 5xx, a dropped
# connection or a timeout are all transient for that particular key.
RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    litellm.RateLimitError,
    litellm.InternalServerError,
    litellm.ServiceUnavailableError,
    litellm.APIConnectionError,
    litellm.Timeout,
    # A key that is revoked or lacking permission is dead for this request
    # but the next key may still work, so these fail over as well.
    litellm.AuthenticationError,
    litellm.PermissionDeniedError,
    asyncio.TimeoutError,
)


class AllProvidersUnavailable(Exception):
    """Raised when every configured API key failed for this request."""


class EmptyProviderReply(Exception):
    """Raised when a provider answered successfully but with no text.

    Treated like a transient provider failure so the next key gets a turn,
    because the API contract promises a non empty reply.
    """


@dataclass
class GenerationResult:
    """Everything the /chat endpoint needs to build its response."""

    reply: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int
    resolution_source: str


# --------------------------------------------------------------------------
# Cache setup
# --------------------------------------------------------------------------


class DiskCacheBackend(BaseCache):
    """A disk backed LiteLLM cache backend.

    LiteLLM removed its built in "disk" cache type, so CACHE_TYPE=disk is
    served by this small backend instead. It is plugged into the normal
    litellm.Cache object, which keeps LiteLLM in charge of cache key
    generation and of serialising responses. The underlying diskcache
    library stores everything in a local SQLite database, so this adds no
    external service.
    """

    def __init__(self, cache_dir: str, default_ttl: int = 60 * 60 * 24) -> None:
        super().__init__(default_ttl=default_ttl)
        import diskcache

        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        self.disk_cache = diskcache.Cache(cache_dir)

    def set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        ttl = kwargs.get("ttl")
        self.disk_cache.set(key, value, expire=ttl)

    def get_cache(self, key: str, **kwargs: Any) -> Any:
        return self.disk_cache.get(key)

    async def async_set_cache(self, key: str, value: Any, **kwargs: Any) -> None:
        self.set_cache(key, value, **kwargs)

    async def async_get_cache(self, key: str, **kwargs: Any) -> Any:
        return self.get_cache(key, **kwargs)

    async def async_set_cache_pipeline(self, cache_list: Any, **kwargs: Any) -> None:
        for key, value in cache_list:
            self.set_cache(key, value, **kwargs)

    async def batch_cache_write(self, key: str, value: Any, **kwargs: Any) -> None:
        self.set_cache(key, value, **kwargs)

    def flush_cache(self) -> None:
        self.disk_cache.clear()

    async def disconnect(self) -> None:
        self.disk_cache.close()


def configure_cache(settings: Settings, disk_cache_dir: str) -> None:
    """Install the LiteLLM response cache for this process.

    max_messages=None is important: current LiteLLM refuses to cache
    requests with more than four messages by default, which would silently
    disable caching as soon as a conversation grows.
    """
    if settings.cache_type == "disk":
        litellm.cache = litellm.Cache(
            type=None,
            _backend=DiskCacheBackend(disk_cache_dir),
            max_messages=None,
        )
    else:
        litellm.cache = litellm.Cache(type="local", max_messages=None)
    logger.info("LLM response cache enabled (type=%s)", settings.cache_type)


def build_cache_key(model: str, messages: list[dict[str, str]]) -> str:
    """Build the cache key from the model and the full message list.

    The key deliberately covers only the model and the messages. The API key
    is not part of it, otherwise a reply produced by key_1 would not be
    found when the same question is later served through key_2.
    """
    if litellm.cache is None:
        raise RuntimeError("LLM cache is not configured")
    return litellm.cache.get_cache_key(model=model, messages=messages)


# --------------------------------------------------------------------------
# Failure simulation
# --------------------------------------------------------------------------


def parse_simulation_header(raw_header: str | None) -> dict[str, int]:
    """Parse an X-Simulate-Failure header into {key_id: status}.

    The format is a comma separated list of <key_id>=<status> pairs, where
    key_id may be "*" to mean every key. Invalid parts are ignored rather
    than rejected, so a typo cannot turn into a 500.
    """
    if not raw_header:
        return {}

    simulation: dict[str, int] = {}
    for part in raw_header.split(","):
        if "=" not in part:
            continue
        raw_key, raw_status = part.split("=", 1)
        key_id = raw_key.strip()
        status_text = raw_status.strip()
        if not key_id or not status_text.isdigit():
            continue
        status = int(status_text)
        if status not in ALLOWED_SIMULATION_STATUSES:
            continue
        simulation[key_id] = status
    return simulation


def _simulated_exception(status: int, key_id: str, model: str) -> Exception:
    """Build the LiteLLM exception that a real provider failure would raise."""
    message = f"simulated {status} failure for {key_id}"
    provider = model.split("/")[0] if "/" in model else model
    if status == 429:
        return litellm.RateLimitError(
            message=message, llm_provider=provider, model=model
        )
    if status == 503:
        return litellm.ServiceUnavailableError(
            message=message, llm_provider=provider, model=model
        )
    if status == 502:
        return litellm.APIError(
            status_code=502, message=message, llm_provider=provider, model=model
        )
    return litellm.InternalServerError(
        message=message, llm_provider=provider, model=model
    )


def _status_for_key(simulation: dict[str, int], key_id: str) -> int | None:
    """Return the simulated status for a key, honouring the wildcard."""
    if key_id in simulation:
        return simulation[key_id]
    return simulation.get(WILDCARD_KEY)


# --------------------------------------------------------------------------
# Response helpers
# --------------------------------------------------------------------------


def _qualified_model(raw: Any, configured: str) -> str:
    """Return the model name with its LiteLLM provider prefix.

    Providers report the bare model name in the response ("gemini-3.1-flash-lite"),
    while the API contract shows the full LiteLLM identifier
    ("gemini/gemini-3.1-flash-lite"). The provider prefix is taken from the
    configured model so the reported name still identifies the model that
    actually answered.
    """
    if not raw:
        return configured
    name = str(raw)
    if "/" in name:
        return name
    prefix, separator, _ = configured.partition("/")
    return f"{prefix}/{name}" if separator else name


def _field(container: Any, name: str, default: Any = None) -> Any:
    """Read a field from either an object or a dict.

    Live calls return a LiteLLM model object while the cache returns a
    plain dict, and both shapes are read the same way here.
    """
    if container is None:
        return default
    if isinstance(container, dict):
        return container.get(name, default)
    return getattr(container, name, default)


def _extract_reply(response: Any) -> str:
    """Pull the assistant text out of a completion response."""
    choices = _field(response, "choices") or []
    if not choices:
        return ""
    message = _field(choices[0], "message")
    content = _field(message, "content")
    if content is None:
        return ""
    return str(content)


def _extract_usage(response: Any) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) for this response only.

    Taken straight from the usage block of the single response at hand, so
    the numbers are never a running total across the conversation.
    """
    usage = _field(response, "usage")
    input_tokens = _field(usage, "prompt_tokens", 0) or 0
    output_tokens = _field(usage, "completion_tokens", 0) or 0
    try:
        return int(input_tokens), int(output_tokens)
    except (TypeError, ValueError):
        return 0, 0


def _is_retryable(exception: Exception) -> bool:
    """True when the next API key should be given a chance."""
    if isinstance(exception, RETRYABLE_EXCEPTIONS):
        return True
    if isinstance(exception, EmptyProviderReply):
        return True

    # Some providers surface quota problems as a generic APIError, so fall
    # back to the HTTP status and then to the error text.
    status_code = getattr(exception, "status_code", None)
    if isinstance(status_code, int) and (status_code == 429 or status_code >= 500):
        return True

    text = str(exception).lower()
    quota_markers = ("resource_exhausted", "resource exhausted", "quota", "rate limit")
    return any(marker in text for marker in quota_markers)


# --------------------------------------------------------------------------
# The gateway
# --------------------------------------------------------------------------


async def _call_with_failover(
    messages: list[dict[str, str]],
    simulation: dict[str, int],
    settings: Settings,
) -> tuple[Any, str, str, int]:
    """Try every API key in order until one answers.

    Returns (response, reply, key_id, fallback_attempts).
    Raises AllProvidersUnavailable when no key could serve the request.
    """
    keyed_api_keys = settings.keyed_api_keys
    if not keyed_api_keys:
        raise AllProvidersUnavailable("no API keys are configured")

    fallback_attempts = 0
    last_error: Exception | None = None

    for key_id, api_key in keyed_api_keys:
        try:
            # Simulated failures are raised here, before the network call and
            # inside this same loop, so they travel the exact code path a
            # real provider failure would take.
            simulated_status = _status_for_key(simulation, key_id)
            if simulated_status is not None:
                raise _simulated_exception(
                    simulated_status, key_id, settings.llm_model
                )

            response = await litellm.acompletion(
                model=settings.llm_model,
                messages=messages,
                api_key=api_key,
                num_retries=0,
                timeout=30,
            )

            reply = _extract_reply(response)
            if not reply.strip():
                raise EmptyProviderReply(f"{key_id} returned an empty reply")

            return response, reply, key_id, fallback_attempts

        except Exception as exception:  # noqa: BLE001 - classified below
            if not _is_retryable(exception):
                # A bad request or a similar permanent error would fail the
                # same way on every key, so there is nothing to fail over to.
                raise
            fallback_attempts += 1
            last_error = exception
            # Only the public key id is logged, never the key itself.
            logger.warning(
                "API %s failed with %s, trying the next key",
                key_id,
                type(exception).__name__,
            )

    raise AllProvidersUnavailable(
        f"all {len(keyed_api_keys)} keys failed, last error: {last_error}"
    )


async def generate(
    messages: list[dict[str, str]],
    simulate: dict[str, int] | None,
    settings: Settings,
) -> GenerationResult:
    """Produce an assistant reply, from cache when possible.

    On a cache hit the stored reply and the stored token usage are returned
    unchanged, with no key id and no fallback attempts, because no provider
    was contacted.
    """
    simulation = simulate or {}
    cache_key = build_cache_key(settings.llm_model, messages)

    # A simulated failure request skips the cache lookup on purpose: a cache
    # hit would return before the failover loop ran, so the simulation would
    # never be exercised.
    if not simulation:
        cached = litellm.cache.get_cache(cache_key=cache_key)
        if cached is not None:
            reply = _extract_reply(cached)
            if reply.strip():
                input_tokens, output_tokens = _extract_usage(cached)
                return GenerationResult(
                    reply=reply,
                    model=_qualified_model(_field(cached, "model"), settings.llm_model),
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=None,
                    fallback_attempts=0,
                    resolution_source=RESOLUTION_CACHE,
                )

    response, reply, key_id, fallback_attempts = await _call_with_failover(
        messages, simulation, settings
    )

    # Store the fresh response so the identical message list is served from
    # the cache next time, whichever key answered this time.
    litellm.cache.add_cache(response, cache_key=cache_key)

    input_tokens, output_tokens = _extract_usage(response)
    return GenerationResult(
        reply=reply,
        model=_qualified_model(_field(response, "model"), settings.llm_model),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        key_id=key_id,
        fallback_attempts=fallback_attempts,
        resolution_source=RESOLUTION_LIVE,
    )
