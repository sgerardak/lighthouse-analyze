"""Tests for the error contract, including failures the framework raises."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def assert_error_shape(response, expected_type: str, status: int):
    """Every failure must carry the same five fields and a matching header."""
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    error = body["error"]
    assert set(error) == {"type", "layer", "message", "retryable", "request_id"}
    assert error["type"] == expected_type
    assert error["request_id"] == response.headers["X-Request-ID"]
    return error


def test_unknown_path_uses_our_error_shape(client):
    """A 404 used to answer {'detail': 'Not Found'}, breaking the contract."""
    error = assert_error_shape(client.get("/nope"), "not_found", 404)

    assert error["layer"] == "http"
    assert error["retryable"] is False


def test_wrong_method_uses_our_error_shape_and_keeps_allow(client):
    response = client.get("/analyze")

    error = assert_error_shape(response, "method_not_allowed", 405)
    assert error["layer"] == "http"
    # The Allow header is what makes a 405 actionable, so it must survive.
    assert "POST" in response.headers.get("allow", "")


def test_validation_failure_uses_our_error_shape(client):
    assert_error_shape(client.post("/analyze", json={}), "invalid_request", 422)


def test_query_too_long_uses_our_error_shape(client):
    from app.config import get_settings

    long_query = "x" * (get_settings().MAX_QUERY_LENGTH + 1)
    error = assert_error_shape(
        client.post("/analyze", json={"query": long_query}), "query_too_long", 413
    )

    assert str(get_settings().MAX_QUERY_LENGTH) in error["message"]


def test_every_response_carries_a_unique_request_id(client):
    ids = {client.get("/nope").headers["X-Request-ID"] for _ in range(3)}

    assert len(ids) == 3
