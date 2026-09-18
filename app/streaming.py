"""Server-sent events helpers for streaming LLM responses to clients.

The event contract:

    meta    {"request_id": ..., "model": ...}   once, first
    delta   {"text": ...}                       zero or more, in order
    result  the complete AnalyzeResponse        once, on success
    error   the same ErrorResponse as the       instead of result, on failure
            non-streaming path

Deltas are provisional. The answer is only checked once it is complete, so a
client must treat the text it has rendered as unconfirmed until 'result'
arrives, and be ready for 'error' to arrive after visible output.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.errors import AppError, to_error_response
from app.schemas import AnalyzeResponse


@dataclass(frozen=True)
class ServiceEvent:
    """One event to send, named by the contract above."""

    name: str
    data: dict


def meta_event(request_id: str, model: str) -> ServiceEvent:
    """Announce the request before any content exists."""
    return ServiceEvent("meta", {"request_id": request_id, "model": model})


def delta_event(text: str) -> ServiceEvent:
    """Send the next piece of the answer."""
    return ServiceEvent("delta", {"text": text})


def result_event(response: AnalyzeResponse) -> ServiceEvent:
    """Send the complete, fully validated answer."""
    return ServiceEvent("result", response.model_dump())


def error_event(exc: AppError, request_id: str) -> ServiceEvent:
    """Report a failure in the same shape the non-streaming path uses."""
    return ServiceEvent("error", to_error_response(exc, request_id).model_dump())


async def to_sse(
    events: AsyncIterator[ServiceEvent],
) -> AsyncIterator[ServerSentEvent]:
    """Render service events as SSE frames."""
    async for event in events:
        yield ServerSentEvent(event=event.name, data=json.dumps(event.data))


def sse_response(
    events: AsyncIterator[ServiceEvent], request_id: str
) -> EventSourceResponse:
    """Wrap a stream of service events in an SSE response."""
    return EventSourceResponse(
        to_sse(events),
        headers={"X-Request-ID": request_id},
    )
