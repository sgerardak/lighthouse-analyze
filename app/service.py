"""Orchestration of a single analysis request."""

from __future__ import annotations

import asyncio
import logging

from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import get_settings
from app.errors import (
    LLMInvalidOutputError,
    LLMTimeoutError,
    LLMUnavailableError,
    RequestTimeoutError,
)
from app.limits import verdict_sentence, verify_arithmetic
from app.llm import get_llm_client
from app.llm.base import LLMResult
from app.policy import get_policy, validate_sources
from app.prompts import build_system_prompt, build_user_message
from app.schemas import AnalyzeResponse, PolicyAnswer

logger = logging.getLogger("app.service")

TOOL_NAME = "submit_policy_answer"
TOOL_DESCRIPTION = (
    "Submit the structured answer to the employee's policy question. "
    "This is the only way to reply."
)

# Failures a retry can plausibly fix: the provider never produced output, so
# the retry is also free. LLMInvalidOutputError is deliberately absent.
RETRYABLE_ERRORS = (LLMUnavailableError, LLMTimeoutError)

# Backoff between attempts; tests set these to zero.
RETRY_INITIAL_WAIT = 0.5
RETRY_MAX_WAIT = 8.0


def _summarise(exc: ValidationError, limit: int = 3) -> str:
    """Condense a ValidationError into one short line for the error message."""
    problems = [
        # A model-level validator has no field path; label it as the whole answer.
        f"{'.'.join(str(part) for part in error['loc']) or 'answer object'}: "
        f"{error['msg']}"
        for error in exc.errors()[:limit]
    ]
    return "; ".join(problems)


async def analyze_query(query: str, request_id: str) -> AnalyzeResponse:
    """Answer one policy question under the overall request deadline."""
    timeout = get_settings().REQUEST_TIMEOUT_SECONDS
    try:
        async with asyncio.timeout(timeout):
            return await _analyze(query, request_id)
    except TimeoutError as exc:
        # Reached when the retries below outlast the deadline; the caller gets
        # one bounded wait instead of however long the provider takes.
        logger.warning("request_id=%s request timed out after %ss", request_id, timeout)
        raise RequestTimeoutError(
            f"The request did not complete within {timeout}s."
        ) from exc


async def _call_llm(client, request_id: str, **kwargs) -> LLMResult:
    """Call the provider, retrying failures that a retry can plausibly fix.

    Only transient errors are retried. A malformed answer is not: that call
    already produced billable output, so retrying it pays twice, and it is a
    deliberate decision rather than a default.
    """
    attempts = 1 + max(0, get_settings().MAX_RETRIES)

    last_error: Exception | None = None

    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential_jitter(initial=RETRY_INITIAL_WAIT, max=RETRY_MAX_WAIT),
        retry=retry_if_exception_type(RETRYABLE_ERRORS),
        reraise=True,
    ):
        with attempt:
            number = attempt.retry_state.attempt_number
            if number > 1:
                logger.warning(
                    "request_id=%s retrying llm call attempt=%d/%d after %s",
                    request_id,
                    number,
                    attempts,
                    type(last_error).__name__,
                )
            try:
                return await client.get_structured_output(**kwargs)
            except Exception as exc:
                last_error = exc
                raise

    raise AssertionError("unreachable: AsyncRetrying always returns or raises")


async def _analyze(query: str, request_id: str) -> AnalyzeResponse:
    """Answer one policy question, validating everything the model returned."""
    policy = get_policy()
    client = get_llm_client()

    result = await _call_llm(
        client,
        request_id,
        system_prompt=build_system_prompt(policy.text),
        user_message=build_user_message(query),
        schema=PolicyAnswer.model_json_schema(),
        schema_name=TOOL_NAME,
        schema_description=TOOL_DESCRIPTION,
    )

    logger.info(
        "request_id=%s model=%s latency_ms=%d input_tokens=%d output_tokens=%d "
        "cache_read_tokens=%d stop_reason=%s",
        request_id,
        result.model,
        result.latency_ms,
        result.input_tokens,
        result.output_tokens,
        result.cache_read_tokens,
        result.stop_reason,
    )

    # The tool schema constrains the shape, but not our cross-field rules, so
    # the answer is re-validated here before anyone sees it.
    try:
        answer = PolicyAnswer.model_validate(result.data)
    except ValidationError as exc:
        raise LLMInvalidOutputError(
            f"Claude returned an answer that does not fit the schema: {_summarise(exc)}"
        ) from exc

    # Nothing stops a model from citing a section that does not exist, which is
    # what a fabricated answer looks like, so the ids are checked too.
    unknown = validate_sources(answer.sources, policy)
    if unknown:
        raise LLMInvalidOutputError(
            "Claude cited policy sections that do not exist: " + ", ".join(unknown)
        )

    # The model is unreliable at comparing an amount against a threshold, so
    # every comparison it claims is recomputed here.
    problems = verify_arithmetic(answer)
    if problems:
        logger.warning("request_id=%s arithmetic rejected: %s", request_id, problems)
        raise LLMInvalidOutputError(
            "Claude's arithmetic does not check out: " + "; ".join(problems)
        )

    # The comparison is stated by us, not by the model, so the sentence the
    # employee reads cannot contradict the arithmetic above.
    sentence = verdict_sentence(answer)
    if sentence:
        answer = answer.model_copy(update={"answer": f"{sentence} {answer.answer}"})

    return AnalyzeResponse(
        request_id=request_id,
        model=result.model,
        result=answer,
    )
