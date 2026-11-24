"""Application configuration.

Every setting is read from the environment via pydantic-settings.
Comma separated variables are kept as raw strings and split by helper
properties, because pydantic would otherwise try to parse them as JSON.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration for the backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Comma separated list of provider API keys. At least two are expected so
    # that failover can actually be demonstrated.
    llm_api_keys: str = Field(default="")

    # The LiteLLM model identifier, e.g. "gemini/gemini-3.1-flash-lite".
    llm_model: str = Field(default="gemini/gemini-3.1-flash-lite")

    # SQLite only. Four slashes means an absolute path: sqlite:////data/app.db
    database_url: str = Field(default="sqlite:///./data/app.db")

    # Comma separated list of origins allowed by the CORS middleware.
    allowed_origins: str = Field(default="http://localhost:3000,http://localhost:5173")

    # When false the X-Simulate-Failure header is ignored completely.
    enable_failure_simulation: bool = Field(default=False)

    # Used when a session does not define its own system prompt.
    default_system_prompt: str = Field(
        default="You are a helpful, concise assistant."
    )

    # "local" keeps the cache in memory, "disk" persists it on disk.
    cache_type: str = Field(default="local")

    @field_validator("cache_type")
    @classmethod
    def _validate_cache_type(cls, value: str) -> str:
        normalised = value.strip().lower()
        if normalised not in {"local", "disk"}:
            raise ValueError("CACHE_TYPE must be either 'local' or 'disk'")
        return normalised

    @property
    def api_keys(self) -> list[str]:
        """The provider keys, in the order they were configured.

        The position in this list defines the key id: the first key is
        "key_1", the second "key_2", and so on.
        """
        return [key.strip() for key in self.llm_api_keys.split(",") if key.strip()]

    @property
    def key_ids(self) -> list[str]:
        """Public identifiers for the configured keys ("key_1", "key_2", ...)."""
        return [f"key_{index}" for index in range(1, len(self.api_keys) + 1)]

    @property
    def keyed_api_keys(self) -> list[tuple[str, str]]:
        """Pairs of (key_id, real key) in failover order.

        The real key value must never be logged or returned to a client.
        """
        return list(zip(self.key_ids, self.api_keys))

    @property
    def origins(self) -> list[str]:
        """Allowed CORS origins."""
        return [origin.strip() for origin in self.allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Return the settings singleton.

    Cached so that the environment is read once per process, but still a
    function call so tests can clear the cache when they need to.
    """
    return Settings()
