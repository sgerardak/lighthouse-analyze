"""LLM provider package: exposes the provider-agnostic client interface."""

from __future__ import annotations

from functools import lru_cache

from app.config import get_settings
from app.llm.base import LLMClient, LLMResult

__all__ = ["LLMClient", "LLMResult", "get_llm_client"]


@lru_cache
def get_llm_client() -> LLMClient:
    """Return the client for the configured provider, built once."""
    provider = get_settings().LLM_PROVIDER.strip().lower()

    if provider == "anthropic":
        # Imported lazily so this package stays free of provider SDKs, and so a
        # new provider is a new module plus a branch here.
        from app.llm.anthropic_client import AnthropicClient

        return AnthropicClient()

    raise ValueError(
        f"Unsupported LLM_PROVIDER {provider!r}. Supported providers: 'anthropic'."
    )
