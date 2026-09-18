"""Orchestration of a single analysis request."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from types import SimpleNamespace

from pydantic import ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.config import get_settings
from app.errors import (
    AppError,
    InternalError,
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
from app.streaming import (
    ServiceEvent,
    delta_event,
    error_event,
    meta_event,
    result_event,
)

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


def _backoff(attempt: int) -> float:
    """Exponential backoff for the streaming path, which cannot use tenacity."""
    return min(RETRY_MAX_WAIT, RETRY_INITIAL_WAIT * (2 ** (attempt - 1)))


def _partial(snapshot: dict) -> SimpleNamespace:
    """Adapt a partial tool input to what verdict_sentence reads.

    The snapshot is mid-generation, so any field may still be absent; it is
    never validated here, only used to compose a sentence that the final
    validation must still agree with.
    """
    return SimpleNamespace(
        verdict=snapshot.get("verdict"),
        limit_applied=snapshot.get("limit_applied"),
        per_person=snapshot.get("per_person"),
        amount_eur=snapshot.get("amount_eur"),
        sources=snapshot.get("sources") or [],
    )


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

    _log_usage(result, request_id)
    return _build_response(result, policy, request_id)


def _log_usage(result: LLMResult, request_id: str) -> None:
    """Record what the call cost, whether or not its output survives checks."""
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


def _build_response(result: LLMResult, policy, request_id: str) -> AnalyzeResponse:
    """Run every check, then compose the answer the caller receives.

    Shared by both paths, so a streamed answer is held to exactly the same
    standard as a buffered one.
    """
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


async def stream_analyze_query(
    query: str, request_id: str
) -> AsyncIterator[ServiceEvent]:
    """Answer one policy question as a stream of events.

    Errors become an event rather than an exception: once the response has
    started, its HTTP status is already sent and cannot be changed.
    """
    settings = get_settings()
    deadline = time.monotonic() + settings.REQUEST_TIMEOUT_SECONDS

    try:
        async for event in _stream(query, request_id, deadline):
            yield event
    except AppError as exc:
        logger.warning(
            "request_id=%s stream failed error_type=%s", request_id, exc.error_type
        )
        yield error_event(exc, request_id)
    except Exception:
        logger.exception("request_id=%s unhandled exception while streaming", request_id)
        yield error_event(InternalError(), request_id)


async def _stream(
    query: str, request_id: str, deadline: float
) -> AsyncIterator[ServiceEvent]:
    """Drive the provider stream and decide what the client is told."""
    settings = get_settings()

    # Sent before the client is built, so the caller gets a first byte straight
    # away rather than waiting on a cold provider client.
    yield meta_event(request_id, settings.MODEL_NAME)

    policy = get_policy()
    client = get_llm_client()
    call = {
        "system_prompt": build_system_prompt(policy.text),
        "user_message": build_user_message(query),
        "schema": PolicyAnswer.model_json_schema(),
        "schema_name": TOOL_NAME,
        "schema_description": TOOL_DESCRIPTION,
    }

    attempts = 1 + max(0, settings.MAX_RETRIES)
    for attempt in range(1, attempts + 1):
        # sent is everything the client has already been given, so the next
        # delta is always the part of the composed answer beyond it.
        sent = ""
        prefix = ""
        sentence_done = False

        try:
            async for event in client.stream_structured_output(**call):
                # The deadline is checked between events rather than with
                # asyncio.timeout: a timeout scope cannot safely span the yields
                # of an async generator, whose consumer runs in another context.
                if time.monotonic() > deadline:
                    raise RequestTimeoutError(
                        f"The request did not complete within "
                        f"{settings.REQUEST_TIMEOUT_SECONDS}s."
                    )

                if event.result is not None:
                    _log_usage(event.result, request_id)
                    yield result_event(_build_response(event.result, policy, request_id))
                    return

                snapshot = event.snapshot or {}

                # Field order makes this reliable: 'answer' is generated last, so
                # once it appears every number the sentence needs is final.
                if "answer" in snapshot and not sentence_done:
                    sentence_done = True
                    sentence = verdict_sentence(_partial(snapshot))
                    if sentence:
                        prefix = f"{sentence} "
                        sent = prefix
                        yield delta_event(prefix)

                if not sentence_done:
                    continue

                text = snapshot.get("answer")
                composed = prefix + (text if isinstance(text, str) else "")
                if len(composed) > len(sent):
                    addition = composed[len(sent) :]
                    sent = composed
                    yield delta_event(addition)

            # The provider ended the stream without a final result.
            raise LLMInvalidOutputError("The model stream ended without an answer.")

        except RETRYABLE_ERRORS as exc:
            # A retry can only be transparent while nothing has reached the
            # client. Once a delta is out, the stream is committed.
            if sent or attempt == attempts:
                raise
            logger.warning(
                "request_id=%s retrying stream attempt=%d/%d after %s",
                request_id,
                attempt + 1,
                attempts,
                type(exc).__name__,
            )
            await asyncio.sleep(_backoff(attempt))
