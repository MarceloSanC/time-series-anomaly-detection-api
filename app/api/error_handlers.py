from __future__ import annotations

from datetime import UTC, datetime
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.domain.exceptions import (
    ConstantSeriesError,
    DuplicateTimestampsError,
    InsufficientDataError,
    InvalidValuesError,
    SeriesNotFoundError,
    ValidationServiceError,
    VersionNotFoundError,
    UnorderedTimestampsError,
)

logger = logging.getLogger(__name__)

VALIDATION_ERROR_CODE_MAP: dict[type[ValidationServiceError], str] = {
    InsufficientDataError: "INSUFFICIENT_DATA",
    ConstantSeriesError: "CONSTANT_SERIES",
    DuplicateTimestampsError: "DUPLICATE_TIMESTAMPS",
    UnorderedTimestampsError: "UNORDERED_TIMESTAMPS",
    InvalidValuesError: "INVALID_VALUES",
}


def _error_payload(error: str, message: str, detail: str | None = None) -> dict[str, str | None]:
    """Build standardized API error payload with UTC timestamp."""
    return {
        "error": error,
        "message": message,
        "detail": detail,
        "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }

def register_error_handlers(app: FastAPI) -> None:
    """Register API-level exception handlers with normalized error responses."""

    @app.exception_handler(SeriesNotFoundError)
    async def handle_series_not_found(_: Request, exc: SeriesNotFoundError) -> JSONResponse:
        """Translate missing series domain error into HTTP 404 response."""
        logger.warning("Series not found", extra={"error": str(exc)})
        return JSONResponse(
            status_code=404,
            content=_error_payload(error="SERIES_NOT_FOUND", message=str(exc)),
        )

    @app.exception_handler(VersionNotFoundError)
    async def handle_version_not_found(_: Request, exc: VersionNotFoundError) -> JSONResponse:
        """Translate missing version domain error into HTTP 404 response."""
        logger.warning("Version not found", extra={"error": str(exc)})
        return JSONResponse(
            status_code=404,
            content=_error_payload(error="VERSION_NOT_FOUND", message=str(exc)),
        )

    @app.exception_handler(ValidationServiceError)
    async def handle_validation_service_error(_: Request, exc: ValidationServiceError) -> JSONResponse:
        """Translate business-rule validation rejections into HTTP 400 responses."""
        error_code = VALIDATION_ERROR_CODE_MAP.get(type(exc), "VALIDATION_ERROR")
        logger.warning("Validation rejected", extra={"error_code": error_code, "detail": str(exc)})
        return JSONResponse(
            status_code=400,
            content=_error_payload(error=error_code, message=str(exc)),
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        """Translate FastAPI request validation errors into normalized 422 payload."""
        logger.warning("Request validation error", extra={"error": str(exc)})
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                error="VALIDATION_ERROR",
                message="Request payload validation failed",
                detail=str(exc.errors()),
            ),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_: Request, _exc: Exception) -> JSONResponse:
        """Catch unhandled exceptions and return generic HTTP 500 payload."""
        logger.exception("Unhandled internal error")
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                error="INTERNAL_ERROR",
                message="An unexpected internal error occurred",
            ),
        )
