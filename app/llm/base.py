"""Abstract LLMClient interface that every provider implementation must satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class LLMResult:
    """A structured answer from a provider, plus what the call cost us."""

    data: dict
    model: str
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    stop_reason: str | None
    latency_ms: int


class LLMClient(ABC):
    """Provider-agnostic interface for producing schema-constrained output.

    Implementations live in sibling modules and are the only place allowed to
    import a provider SDK; the rest of the app depends on this interface alone.
    """

    @abstractmethod
    async def get_structured_output(
        self,
        system_prompt: str,
        user_message: str,
        schema: dict,
        schema_name: str,
        schema_description: str,
    ) -> LLMResult:
        """Return output conforming to schema, or raise an app.errors LLM error."""
        raise NotImplementedError
