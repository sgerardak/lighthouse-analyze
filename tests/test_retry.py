"""Tests for the retry policy and the overall request deadline.

Every test drives a fake client, so none of them calls the provider.
"""

import asyncio

import pytest

from app import service
from app.errors import (
    LLMAuthError,
    LLMInvalidOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
    RequestTimeoutError,
)
from app.llm.base import LLMResult

GOOD_DATA = {
    "answer": "Team meals are allowed up to 40 EUR per person.",
    "sources": ["4.2"],
    "in_scope": True,
    "escalate_to_finance": False,
    "confidence": "high",
    "amount_eur": 300.0,
    "headcount": 6,
    "per_person": 50.0,
    "limit_applied": 40.0,
    "verdict": "above_threshold",
}


class FakeClient:
    """Raises the queued failures, then returns a valid answer."""

    def __init__(self, failures=(), delay=0.0, data=None):
        self.failures = list(failures)
        self.delay = delay
        self.data = data or GOOD_DATA
        self.calls = 0

    async def get_structured_output(self, **kwargs) -> LLMResult:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.failures:
            raise self.failures.pop(0)
        return LLMResult(self.data, "fake-model", 100, 50, 0, "tool_use", 10)


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    """Keep the retry waits out of the test runtime."""
    monkeypatch.setattr(service, "RETRY_INITIAL_WAIT", 0.0)
    monkeypatch.setattr(service, "RETRY_MAX_WAIT", 0.0)


@pytest.fixture
def use_client(monkeypatch):
    def install(client):
        monkeypatch.setattr(service, "get_llm_client", lambda: client)
        return client

    return install


@pytest.mark.asyncio
async def test_succeeds_without_retrying_when_the_first_call_works(use_client):
    client = use_client(FakeClient())

    response = await service.analyze_query("Team dinner for 6, 300 EUR?", "r1")

    assert client.calls == 1
    assert response.result.verdict == "above_threshold"


@pytest.mark.asyncio
async def test_retries_transient_failures_then_succeeds(use_client):
    client = use_client(
        FakeClient(failures=[LLMUnavailableError(), LLMTimeoutError()])
    )

    response = await service.analyze_query("Team dinner for 6, 300 EUR?", "r2")

    assert client.calls == 3
    assert response.result.sources == ["4.2"]


@pytest.mark.asyncio
async def test_gives_up_after_max_retries_and_reraises_the_last_error(use_client):
    from app.config import get_settings

    attempts = 1 + get_settings().MAX_RETRIES
    client = use_client(FakeClient(failures=[LLMUnavailableError()] * (attempts + 2)))

    with pytest.raises(LLMUnavailableError):
        await service.analyze_query("Team dinner for 6, 300 EUR?", "r3")

    assert client.calls == attempts


@pytest.mark.asyncio
async def test_does_not_retry_a_malformed_answer(use_client):
    """A completed call is billable; retrying it is a decision, not a default."""
    client = use_client(FakeClient(data={**GOOD_DATA, "confidence": "certain"}))

    with pytest.raises(LLMInvalidOutputError):
        await service.analyze_query("Team dinner for 6, 300 EUR?", "r4")

    assert client.calls == 1


@pytest.mark.asyncio
async def test_does_not_retry_an_auth_error(use_client):
    """A bad key fails identically every time, so retrying only wastes the deadline."""
    client = use_client(FakeClient(failures=[LLMAuthError()]))

    with pytest.raises(LLMAuthError):
        await service.analyze_query("Team dinner for 6, 300 EUR?", "r5")

    assert client.calls == 1


@pytest.mark.asyncio
async def test_request_deadline_cuts_off_slow_retries(use_client, monkeypatch):
    """Retrying must not let one request run for an unbounded time."""
    from app.config import Settings, get_settings

    settings = get_settings()
    slow = Settings(**{**settings.model_dump(), "REQUEST_TIMEOUT_SECONDS": 1})
    monkeypatch.setattr(service, "get_settings", lambda: slow)
    # Each attempt outlasts half the deadline, so the retries cannot all run.
    client = use_client(FakeClient(failures=[LLMUnavailableError()] * 50, delay=0.6))

    with pytest.raises(RequestTimeoutError) as caught:
        await service.analyze_query("Team dinner for 6, 300 EUR?", "r6")

    assert caught.value.status_code == 504
    assert caught.value.retryable is True
    assert client.calls < 50
