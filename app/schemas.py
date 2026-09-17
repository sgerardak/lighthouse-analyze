"""Pydantic request and response models for the public API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.config import get_settings
from app.errors import QueryTooLongError


class AnalyzeRequest(BaseModel):
    """Body of a POST /analyze request."""

    query: str = Field(description="The free-text policy question to analyse.")

    @field_validator("query")
    @classmethod
    def _normalise_query(cls, value: str) -> str:
        """Trim the query, then reject empty or over-long input."""
        query = value.strip()
        if not query:
            raise ValueError("query must not be empty")

        max_length = get_settings().MAX_QUERY_LENGTH
        if len(query) > max_length:
            # Raised instead of ValueError so the caller gets 413 rather than a
            # generic 422. Pydantic lets non-ValueError exceptions propagate.
            raise QueryTooLongError(
                f"Query is {len(query)} characters; the maximum is {max_length}."
            )
        return query


class PolicyAnswer(BaseModel):
    """Structured answer produced by the LLM.

    This model's JSON schema is sent to the provider as the tool definition, so
    every field carries a description the model can read.
    """

    answer: str = Field(
        min_length=1,
        max_length=1500,
        description="Plain-language answer to the question, grounded in the policy.",
    )
    sources: list[str] = Field(
        description=(
            "Ids of the policy sections supporting the answer, e.g. ['3.2']. "
            "Empty when the question is out of scope."
        ),
    )
    in_scope: bool = Field(
        description="True if the question is covered by the expense policy.",
    )
    escalate_to_finance: bool = Field(
        description=(
            "True if a human in finance must review this case. Always true when "
            "the question is out of scope."
        ),
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="How well the cited policy sections settle the question.",
    )

    @model_validator(mode="after")
    def _check_out_of_scope(self) -> PolicyAnswer:
        """Out-of-scope answers cite nothing and must be escalated."""
        if not self.in_scope:
            if self.sources:
                raise ValueError("sources must be empty when in_scope is false")
            if not self.escalate_to_finance:
                raise ValueError(
                    "escalate_to_finance must be true when in_scope is false"
                )
        return self


class AnalyzeResponse(BaseModel):
    """Successful response of POST /analyze."""

    request_id: str = Field(description="Correlation id, echoed in X-Request-ID.")
    model: str = Field(description="Identifier of the model that produced the answer.")
    result: PolicyAnswer = Field(description="The structured answer.")


class ErrorDetail(BaseModel):
    """Machine-readable description of a single failure."""

    type: str = Field(description="Stable error code, e.g. 'llm_timeout'.")
    layer: Literal["http", "llm", "service"] = Field(
        description="Where the failure originated.",
    )
    message: str = Field(description="Human-readable explanation.")
    retryable: bool = Field(description="True if an identical retry may succeed.")
    request_id: str = Field(description="Correlation id, echoed in X-Request-ID.")


class ErrorResponse(BaseModel):
    """Error body returned for every failed request."""

    error: ErrorDetail
