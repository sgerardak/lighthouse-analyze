"""Tests for the SSE streaming path.

Every test drives a fake client, so none of them calls the provider.
"""

import pytest

from app import service
from app.errors import LLMUnavailableError
from app.llm.base import LLMResult, LLMStreamEvent

FINAL = {
    "amount_eur": 300.0,
    "headcount": 6,
    "per_person": 50.0,
    "limit_applied": 40.0,
    "verdict": "above_threshold",
    "sources": ["4.2"],
    "in_scope": True,
    "escalate_to_finance": False,
    "confidence": "high",
    "answer": "Team meals are allowed up to 40 EUR per person.",
}

SENTENCE = "50 EUR per person is above the 40 EUR per person limit (section 4.2)."


def snapshots(final=None):
    """The snapshots a provider would emit, numbers first and prose last."""
    final = final or FINAL
    numbers = {k: final[k] for k in list(final)[:9]}
    growing = [dict(numbers, answer=final["answer"][:n]) for n in (0, 10, 30)]
    return [{k: final[k] for k in list(final)[:5]}, numbers, *growing, dict(final)]


class FakeStreamClient:
    """Replays snapshots, optionally failing the first N attempts."""

    def __init__(self, final=None, failures=0, fail_after_snapshots=None):
        self.final = final or FINAL
        self.failures = failures
        self.fail_after_snapshots = fail_after_snapshots
        self.attempts = 0

    async def stream_structured_output(self, **kwargs):
        self.attempts += 1
        if self.failures > 0:
            self.failures -= 1
            raise LLMUnavailableError()

        for index, snapshot in enumerate(snapshots(self.final)):
            if self.fail_after_snapshots is not None and index >= self.fail_after_snapshots:
                raise LLMUnavailableError()
            yield LLMStreamEvent(snapshot=snapshot)

        yield LLMStreamEvent(
            result=LLMResult(self.final, "fake-model", 100, 50, 0, "tool_use", 10)
        )


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(service, "RETRY_INITIAL_WAIT", 0.0)
    monkeypatch.setattr(service, "RETRY_MAX_WAIT", 0.0)


@pytest.fixture
def use_client(monkeypatch):
    def install(client):
        monkeypatch.setattr(service, "get_llm_client", lambda: client)
        return client

    return install


async def collect(query="Team dinner for 6, 300 EUR?", request_id="s1"):
    return [event async for event in service.stream_analyze_query(query, request_id)]


@pytest.mark.asyncio
async def test_event_order_is_meta_then_deltas_then_result(use_client):
    use_client(FakeStreamClient())

    events = await collect()

    assert events[0].name == "meta"
    assert events[-1].name == "result"
    assert {e.name for e in events[1:-1]} == {"delta"}


@pytest.mark.asyncio
async def test_the_computed_sentence_is_the_first_delta(use_client):
    """The decisive claim reaches the client before any model prose."""
    use_client(FakeStreamClient())

    events = await collect()
    deltas = [e for e in events if e.name == "delta"]

    assert deltas[0].data["text"] == f"{SENTENCE} "


@pytest.mark.asyncio
async def test_deltas_reassemble_into_the_final_answer(use_client):
    use_client(FakeStreamClient())

    events = await collect()
    streamed = "".join(e.data["text"] for e in events if e.name == "delta")
    final = next(e for e in events if e.name == "result")

    assert streamed == final.data["result"]["answer"]
    assert streamed.startswith(SENTENCE)


@pytest.mark.asyncio
async def test_meta_carries_the_request_id(use_client):
    use_client(FakeStreamClient())

    events = await collect(request_id="abc123")

    assert events[0].data["request_id"] == "abc123"
    assert events[-1].data["request_id"] == "abc123"


@pytest.mark.asyncio
async def test_a_failed_check_ends_the_stream_with_an_error(use_client):
    """Validation still applies to a streamed answer, late though it is."""
    use_client(FakeStreamClient(final={**FINAL, "verdict": "at_or_below_threshold"}))

    events = await collect()

    assert events[-1].name == "error"
    assert events[-1].data["error"]["type"] == "llm_invalid_output"
    assert not any(e.name == "result" for e in events)


@pytest.mark.asyncio
async def test_an_invented_citation_ends_the_stream_with_an_error(use_client):
    use_client(FakeStreamClient(final={**FINAL, "sources": ["9.9"]}))

    events = await collect()

    assert events[-1].name == "error"
    assert "9.9" in events[-1].data["error"]["message"]


@pytest.mark.asyncio
async def test_transient_failure_before_any_delta_is_retried(use_client):
    client = use_client(FakeStreamClient(failures=2))

    events = await collect()

    assert client.attempts == 3
    assert events[-1].name == "result"
    # The retries happened before anything was sent, so the client cannot tell.
    assert [e.name for e in events].count("meta") == 1


@pytest.mark.asyncio
async def test_transient_failure_after_a_delta_is_not_retried(use_client):
    """Once text is out it cannot be unsent, so the stream fails instead."""
    client = use_client(FakeStreamClient(fail_after_snapshots=4))

    events = await collect()

    assert client.attempts == 1
    assert any(e.name == "delta" for e in events)
    assert events[-1].name == "error"
    assert events[-1].data["error"]["type"] == "llm_unavailable"


@pytest.mark.asyncio
async def test_a_stream_that_ends_without_a_result_is_an_error(use_client):
    class Truncated:
        async def stream_structured_output(self, **kwargs):
            yield LLMStreamEvent(snapshot=snapshots()[0])

    use_client(Truncated())

    events = await collect()

    assert events[-1].name == "error"
    assert events[-1].data["error"]["type"] == "llm_invalid_output"


def test_endpoint_returns_an_event_stream(use_client):
    """The SSE frames reach the wire with the right content type and headers."""
    from fastapi.testclient import TestClient

    from app.main import app

    use_client(FakeStreamClient())
    with TestClient(app) as client:
        response = client.post("/analyze?stream=true", json={"query": "Team dinner?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["X-Request-ID"]
    assert "event: meta" in response.text
    assert "event: delta" in response.text
    assert "event: result" in response.text


def test_endpoint_rejects_a_bad_body_without_streaming(use_client):
    """Validation happens before the stream starts, so it is a normal 422."""
    from fastapi.testclient import TestClient

    from app.main import app

    use_client(FakeStreamClient())
    with TestClient(app) as client:
        response = client.post("/analyze?stream=true", json={})

    assert response.status_code == 422
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["error"]["type"] == "invalid_request"
