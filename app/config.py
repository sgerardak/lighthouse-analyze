"""Application settings loaded from the environment and the .env file."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    LLM_PROVIDER: str = "anthropic"
    ANTHROPIC_API_KEY: str
    MODEL_NAME: str = "claude-haiku-4-5-20251001"
    LLM_TIMEOUT_SECONDS: int = 20
    REQUEST_TIMEOUT_SECONDS: int = 45
    MAX_RETRIES: int = 3
    MAX_QUERY_LENGTH: int = 2000
    MAX_OUTPUT_TOKENS: int = 1024
    POLICY_PATH: str = "app/data/expense_policy.md"


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings instance."""
    return Settings()
