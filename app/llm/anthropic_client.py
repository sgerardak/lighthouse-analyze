"""Anthropic-backed implementation of the LLMClient interface."""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

import anthropic
import jiter

from app.config import get_settings
from app.errors import (
    AppError,
    InternalError,
    LLMAuthError,
    LLMInvalidOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
)
from app.llm.base import LLMClient, LLMResult, LLMStreamEvent

logger = logging.getLogger("app.llm.anthropic")


def _parse_partial(raw: str) -> dict | None:
    """Parse half-written tool JSON, including the string being typed.

    The SDK's own snapshot omits an unterminated string, which would hide the
    answer until it was complete and defeat the point of streaming.
    'trailing-strings' keeps it, so the prose can be forwarded as it arrives.
    """
    try:
        parsed = jiter.from_json(raw.encode("utf-8"), partial_mode="trailing-strings")
    except ValueError:
        # Mid-token states that parse as nothing at all are simply skipped.
        return None
    return parsed if isinstance(parsed, dict) else None


def _strict(schema: dict) -> dict:
    """Return the schema in the shape strict tool use requires.

    Anthropic rejects strict tools whose object schemas allow extra properties,
    so that is set here rather than on the shared model, which must stay
    provider-agnostic.
    """
    return {**schema, "additionalProperties": False}


class AnthropicClient(LLMClient):
    """Gets structured output from Claude by forcing a single tool call."""

    def __init__(self) -> None:
        settings = get_settings()
        self._model = settings.MODEL_NAME
        self._max_output_tokens = settings.MAX_OUTPUT_TOKENS
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.ANTHROPIC_API_KEY,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            # The SDK retries 429/5xx itself by default. We disable that so the
            # service owns the retry policy: one place to reason about total
            # latency against REQUEST_TIMEOUT_SECONDS, and one place that
            # decides which of our errors are retryable.
            max_retries=0,
        )

    async def get_structured_output(
        self,
        system_prompt: str,
        user_message: str,
        schema: dict,
        schema_name: str,
        schema_description: str,
    ) -> LLMResult:
        """Call Claude with a forced tool call and return the tool input."""
        started = time.perf_counter()
        try:
            response = await self._client.messages.create(
                **self._request(system_prompt, user_message, schema, schema_name,
                                schema_description)
            )
        except Exception as exc:
            raise self._as_app_error(exc) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        return self._to_result(response, schema_name, latency_ms)

    async def stream_structured_output(
        self,
        system_prompt: str,
        user_message: str,
        schema: dict,
        schema_name: str,
        schema_description: str,
    ) -> AsyncIterator[LLMStreamEvent]:
        """Yield the tool input as Claude produces it, then the final result."""
        started = time.perf_counter()
        request = self._request(
            system_prompt, user_message, schema, schema_name, schema_description
        )
        # Server-side buffering would deliver the whole tool input at once, so
        # there would be nothing to stream. The cost is that the API stops
        # validating the input, which is why the snapshots below are untrusted
        # and the caller revalidates the final result.
        request["tools"][0]["eager_input_streaming"] = True

        raw = ""
        try:
            async with self._client.messages.stream(**request) as stream:
                async for event in stream:
                    if event.type != "input_json":
                        continue
                    raw += event.partial_json or ""
                    snapshot = _parse_partial(raw)
                    if snapshot is not None:
                        yield LLMStreamEvent(snapshot=snapshot)
                response = await stream.get_final_message()
        except Exception as exc:
            raise self._as_app_error(exc) from exc

        latency_ms = int((time.perf_counter() - started) * 1000)
        yield LLMStreamEvent(result=self._to_result(response, schema_name, latency_ms))

    def _request(
        self,
        system_prompt: str,
        user_message: str,
        schema: dict,
        schema_name: str,
        schema_description: str,
    ) -> dict:
        """Build the request both paths send, so they cannot drift apart."""
        return {
            "model": self._model,
            "max_tokens": self._max_output_tokens,
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    # The policy dominates the prompt and never changes between
                    # requests, so cache it.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            "messages": [{"role": "user", "content": user_message}],
            "tools": [
                {
                    "name": schema_name,
                    "description": schema_description,
                    "input_schema": _strict(schema),
                    # Without this the API treats 'required' as advice: the model
                    # can omit a required field and still return a well-formed
                    # tool call. Measured at ~19% of calls on some questions;
                    # strict mode removed it entirely.
                    "strict": True,
                }
            ],
            "tool_choice": {"type": "tool", "name": schema_name},
        }

    @staticmethod
    def _as_app_error(exc: Exception) -> Exception:
        """Translate an SDK exception into ours, leaving our own errors alone."""
        if isinstance(exc, AppError):
            return exc
        # APITimeoutError subclasses APIConnectionError, so it must come first.
        if isinstance(exc, anthropic.APITimeoutError):
            return LLMTimeoutError(
                f"Claude did not respond within {get_settings().LLM_TIMEOUT_SECONDS}s."
            )
        if isinstance(exc, anthropic.APIConnectionError):
            return LLMUnavailableError("Could not reach the Claude API.")
        if isinstance(exc, anthropic.APIStatusError):
            return AnthropicClient._map_status_error(exc)
        return exc

    @staticmethod
    def _map_status_error(exc: anthropic.APIStatusError) -> Exception:
        """Translate an HTTP status from the provider into one of our errors."""
        status = exc.status_code
        logger.warning("anthropic api error status=%s", status)

        if status in (401, 403):
            return LLMAuthError(
                f"Claude rejected our credentials (HTTP {status})."
            )
        if status == 429:
            return LLMUnavailableError("Claude rate limit reached (HTTP 429).")
        if status >= 500:
            # Includes 529 "overloaded", which Anthropic returns under load.
            return LLMUnavailableError(f"Claude returned HTTP {status}.")
        # Any other 4xx means we built a bad request: a caller retrying cannot
        # help, and it is our bug, not the provider's.
        return InternalError(f"The request to Claude was rejected (HTTP {status}).")

    @staticmethod
    def _to_result(response, schema_name: str, latency_ms: int) -> LLMResult:
        """Pull the forced tool call and usage figures out of the response."""
        if response.stop_reason == "max_tokens":
            raise LLMInvalidOutputError(
                "Claude hit the output token limit before completing the answer."
            )

        tool_use = next(
            (
                block
                for block in response.content
                if block.type == "tool_use" and block.name == schema_name
            ),
            None,
        )
        if tool_use is None:
            raise LLMInvalidOutputError(
                f"Claude returned no '{schema_name}' tool call "
                f"(stop_reason={response.stop_reason})."
            )
        if not isinstance(tool_use.input, dict):
            raise LLMInvalidOutputError("Claude returned a non-object tool input.")

        usage = response.usage
        return LLMResult(
            data=tool_use.input,
            model=response.model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            stop_reason=response.stop_reason,
            latency_ms=latency_ms,
        )
