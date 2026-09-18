"""FastAPI application entrypoint: app creation, middleware and route wiring."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.errors import (
    AppError,
    InternalError,
    InvalidRequestError,
    QueryTooLongError,
    to_error_response,
)
from app.policy import get_policy
from app.schemas import AnalyzeRequest, AnalyzeResponse
from app.service import analyze_query, stream_analyze_query
from app.streaming import sse_response

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("app")

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the policy before serving traffic, so a broken file fails fast."""
    policy = get_policy()
    logger.info(
        "policy loaded path=%s sections=%d",
        settings.POLICY_PATH,
        len(policy.sections),
    )
    yield


app = FastAPI(
    title="Lighthouse Analyze Service",
    version="0.1.0",
    lifespan=lifespan,
)


def _request_id(request: Request) -> str:
    """Return the id assigned by the middleware, or a fresh one as a fallback."""
    return getattr(request.state, "request_id", None) or uuid4().hex


def _error_json(exc: AppError, request_id: str) -> JSONResponse:
    """Render an AppError as the standard error response."""
    body = to_error_response(exc, request_id)
    return JSONResponse(
        status_code=exc.status_code,
        content=body.model_dump(),
        headers={"X-Request-ID": request_id},
    )


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tag every request with an id and log how it finished."""
    request_id = uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        # The exception handlers below build the response; log the timing here
        # so failed requests are not missing from the access log.
        duration_ms = (time.perf_counter() - started) * 1000
        logger.error(
            "request_id=%s method=%s path=%s status=500 duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1000
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%d duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        duration_ms,
    )
    return response


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    """Return the structured form of any error the service raised itself."""
    request_id = _request_id(request)
    logger.warning(
        "request_id=%s error_type=%s layer=%s message=%s",
        request_id,
        exc.error_type,
        exc.layer,
        exc.message,
    )
    return _error_json(exc, request_id)


_PARAM_SOURCES = {"body", "query", "path", "header", "cookie"}


def _find_query_too_long(exc: RequestValidationError) -> QueryTooLongError | None:
    """Return the QueryTooLongError behind a validation error, if there is one."""
    for error in exc.errors():
        cause = (error.get("ctx") or {}).get("error")
        if isinstance(cause, QueryTooLongError):
            return cause
    return None


@app.exception_handler(RequestValidationError)
async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Map FastAPI validation failures onto our error model."""
    too_long = _find_query_too_long(exc)
    if too_long is not None:
        return await handle_app_error(request, too_long)

    fields = []
    for error in exc.errors():
        # loc starts with the parameter source ("body", "query", ...); drop it
        # so the message names the field the caller actually sent.
        parts = list(error["loc"])
        if parts and parts[0] in _PARAM_SOURCES:
            parts = parts[1:]
        location = ".".join(str(part) for part in parts)
        fields.append(f"{location or 'body'}: {error['msg']}")
    return await handle_app_error(
        request, InvalidRequestError("Invalid request. " + "; ".join(fields))
    )


@app.exception_handler(Exception)
async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Return a generic 500 and keep the traceback in the logs, not the response."""
    request_id = _request_id(request)
    logger.exception("request_id=%s unhandled exception", request_id)
    return _error_json(InternalError(), request_id)


@app.get("/health")
async def health() -> dict:
    """Report service liveness and the configured LLM provider."""
    return {
        "status": "ok",
        "provider": settings.LLM_PROVIDER,
        "model": settings.MODEL_NAME,
        "policy_sections": len(get_policy().sections),
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    request: Request, payload: AnalyzeRequest, stream: bool = False
):
    """Answer a policy question, grounded in the expense policy.

    With stream=true the answer arrives as server-sent events; the response is
    then always 200, and failures arrive as an 'error' event, because the status
    line is already sent by the time most things can go wrong.
    """
    request_id = _request_id(request)
    if stream:
        return sse_response(
            stream_analyze_query(payload.query, request_id), request_id
        )
    return await analyze_query(payload.query, request_id)
