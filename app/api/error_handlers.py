from __future__ import annotations

from datetime import UTC, datetime
import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.domain.exceptions import (
    ConstantSeriesError,
    DuplicateTimestampsError,
    FlatLineDetectedError,
    InvalidSeriesIdError,
    InsufficientDataError,
    InvalidValuesError,
    MetadataIncompleteError,
    PlotDataUnavailableError,
    SeriesNotFoundError,
    TemporalGapDetectedError,
    UnsupportedDetectorError,
    ValidationServiceError,
    VersionNotFoundError,
    VersionNotFoundForDetectorError,
    UnorderedTimestampsError,
)

logger = logging.getLogger(__name__)

VALIDATION_ERROR_CODE_MAP: dict[type[ValidationServiceError], str] = {
    InsufficientDataError: "INSUFFICIENT_DATA",
    ConstantSeriesError: "CONSTANT_SERIES",
    DuplicateTimestampsError: "DUPLICATE_TIMESTAMPS",
    UnorderedTimestampsError: "UNORDERED_TIMESTAMPS",
    InvalidValuesError: "INVALID_VALUES",
    FlatLineDetectedError: "FLAT_LINE_DETECTED",
    TemporalGapDetectedError: "TEMPORAL_GAP_DETECTED",
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

    @app.exception_handler(InvalidSeriesIdError)
    async def handle_invalid_series_id(_: Request, exc: InvalidSeriesIdError) -> JSONResponse:
        """Translate invalid path series identifier into HTTP 400 response."""
        logger.warning("Invalid series_id", extra={"error": str(exc)})
        return JSONResponse(
            status_code=400,
            content=_error_payload(error="INVALID_SERIES_ID", message=str(exc)),
        )

    @app.exception_handler(UnsupportedDetectorError)
    async def handle_unsupported_detector(_: Request, exc: UnsupportedDetectorError) -> JSONResponse:
        """Translate unsupported detector type into HTTP 422 response."""
        logger.warning("Unsupported detector", extra={"error": str(exc)})
        return JSONResponse(
            status_code=422,
            content=_error_payload(error="UNSUPPORTED_DETECTOR", message=str(exc)),
        )

    @app.exception_handler(VersionNotFoundForDetectorError)
    async def handle_version_not_found_for_detector(_: Request, exc: VersionNotFoundForDetectorError) -> JSONResponse:
        """Translate missing version for a detector into HTTP 404 response."""
        logger.warning("Version not found for detector", extra={"error": str(exc)})
        return JSONResponse(
            status_code=404,
            content=_error_payload(error="VERSION_NOT_FOUND_FOR_DETECTOR", message=str(exc)),
        )

    @app.exception_handler(PlotDataUnavailableError)
    async def handle_plot_data_unavailable(_: Request, exc: PlotDataUnavailableError) -> JSONResponse:
        """Translate missing plot-ready metadata into HTTP 422 response."""
        logger.warning("Plot data unavailable", extra={"error": str(exc)})
        return JSONResponse(
            status_code=422,
            content=_error_payload(error="PLOT_DATA_UNAVAILABLE", message=str(exc)),
        )

    @app.exception_handler(MetadataIncompleteError)
    async def handle_metadata_incomplete(_: Request, exc: MetadataIncompleteError) -> JSONResponse:
        """Translate incomplete metadata in introspection flow into HTTP 422 response."""
        logger.warning("Incomplete model metadata", extra={"error": str(exc)})
        return JSONResponse(
            status_code=422,
            content=_error_payload(error="INCOMPLETE_MODEL_METADATA", message=str(exc)),
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
