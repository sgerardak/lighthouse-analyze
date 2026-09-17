"""Domain exceptions and the handlers that map them to HTTP error responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # pragma: no cover - import cycle broken at runtime
    from app.schemas import ErrorResponse

ErrorLayer = Literal["http", "llm", "service"]


class AppError(Exception):
    """Base class for every error the service reports in a structured form.

    Subclasses describe the error through class attributes; the human-readable
    message is per instance.
    """

    error_type: str = "app_error"
    layer: ErrorLayer = "service"
    status_code: int = 500
    retryable: bool = False
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


# --- HTTP layer: the caller sent something we cannot accept. ---


class InvalidRequestError(AppError):
    """The request body or parameters failed validation."""

    error_type = "invalid_request"
    layer = "http"
    status_code = 422
    retryable = False
    default_message = "The request payload is invalid."


class QueryTooLongError(AppError):
    """The query exceeds the configured maximum length."""

    error_type = "query_too_long"
    layer = "http"
    status_code = 413
    retryable = False
    default_message = "The query is too long."


# --- LLM layer: the upstream model provider failed us. ---


class LLMUnavailableError(AppError):
    """The provider is unreachable, overloaded or rate limiting us."""

    error_type = "llm_unavailable"
    layer = "llm"
    status_code = 503
    retryable = True
    default_message = "The language model is temporarily unavailable."


class LLMTimeoutError(AppError):
    """A single call to the provider exceeded LLM_TIMEOUT_SECONDS."""

    error_type = "llm_timeout"
    layer = "llm"
    status_code = 504
    retryable = True
    default_message = "The language model did not respond in time."


class LLMInvalidOutputError(AppError):
    """The provider returned output that does not satisfy the tool schema."""

    error_type = "llm_invalid_output"
    layer = "llm"
    status_code = 502
    retryable = True
    default_message = "The language model returned malformed output."


class LLMAuthError(AppError):
    """The provider rejected our credentials; retrying cannot help."""

    error_type = "llm_auth_error"
    layer = "llm"
    status_code = 500
    retryable = False
    default_message = "The language model provider rejected our credentials."


# --- Service layer: our own orchestration failed. ---


class RequestTimeoutError(AppError):
    """The whole request exceeded REQUEST_TIMEOUT_SECONDS."""

    error_type = "request_timeout"
    layer = "service"
    status_code = 504
    retryable = True
    default_message = "The request took too long to complete."


class InternalError(AppError):
    """An unexpected failure inside the service."""

    error_type = "internal_error"
    layer = "service"
    status_code = 500
    retryable = False
    default_message = "An internal error occurred."


class NotImplementedYetError(AppError):
    """A requested feature exists in the API surface but is not built yet."""

    error_type = "not_implemented"
    layer = "service"
    status_code = 501
    retryable = False
    default_message = "This feature is not implemented yet."


def to_error_response(exc: AppError, request_id: str) -> ErrorResponse:
    """Build the wire-format error body for an AppError."""
    # Imported here because app.schemas imports QueryTooLongError from this
    # module; a module-level import would be circular.
    from app.schemas import ErrorDetail, ErrorResponse

    return ErrorResponse(
        error=ErrorDetail(
            type=exc.error_type,
            layer=exc.layer,
            message=exc.message,
            retryable=exc.retryable,
            request_id=request_id,
        )
    )
