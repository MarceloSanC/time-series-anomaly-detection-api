from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query

from app.dependencies import get_model_service
from app.domain.schemas import ErrorResponse, ModelDetail, ModelSummary, ModelVersionMetadata
from app.services.model_service import ModelService

router = APIRouter(tags=["Model Introspection"])
logger = logging.getLogger(__name__)


@router.get(
    "/models",
    response_model=list[ModelSummary],
    summary="List tracked model series",
    description="Lists series summaries for latest version metadata. Can run in tolerant or strict mode.",
    responses={
        422: {
            "model": ErrorResponse,
            "description": "Incomplete metadata found when `strict=true` or unsupported detector value.",
            "content": {
                "application/json": {
                    "examples": {
                        "incomplete_metadata": {
                            "summary": "Missing latest metadata (strict mode)",
                            "value": {
                                "error": "INCOMPLETE_MODEL_METADATA",
                                "message": "Metadata missing for series 'sensor_XYZ' at latest version 'v1'",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        },
                        "unsupported_detector": {
                            "summary": "Unsupported detector query value",
                            "value": {
                                "error": "UNSUPPORTED_DETECTOR",
                                "message": "Detector 'random_forest' is not supported",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        },
                    }
                }
            },
        }
    },
)
def list_models(
    strict: bool = Query(
        default=False,
        description=(
            "When true, fail-fast if any series has incomplete latest metadata. "
            "When false, ignore incomplete metadata and return all available summaries."
        ),
    ),
    detector: str | None = Query(
        default=None,
        description=(
            "Filter results to a specific detector type. Supported values: gaussian, isolation_forest. "
            "Returns all detectors when omitted or sent empty."
        ),
        openapi_examples={
            "gaussian": {"summary": "Gaussian only", "value": "gaussian"},
            "isolation_forest": {"summary": "Isolation Forest only", "value": "isolation_forest"},
            "all_detectors": {
                "summary": "All detectors",
                "description": "Omit `detector` or send empty value (`?detector=`) to include all detector namespaces.",
            },
        },
    ),
    model_service: ModelService = Depends(get_model_service),
) -> list[ModelSummary]:
    """List all tracked series with latest version and summary metadata."""
    normalized_detector = detector.strip() if detector is not None else None
    detector_filter = normalized_detector or None

    logger.info("Models list request received", extra={"strict": strict, "detector": detector_filter})
    summaries = model_service.list_model_summaries(strict=strict, detector=detector_filter)
    logger.info("Models list request completed", extra={"series_count": len(summaries), "strict": strict})
    return summaries


@router.get(
    "/models/{series_id}",
    response_model=ModelDetail,
    summary="Get model detail for one series",
    description="Returns lineage metadata and derived data-quality indicators for the latest model version.",
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Series not found.",
            "content": {
                "application/json": {
                    "examples": {
                        "series_not_found": {
                            "summary": "Series does not exist",
                            "value": {
                                "error": "SERIES_NOT_FOUND",
                                "message": "Series 'sensor_XYZ' not found",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        }
                    }
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "Unsupported detector value.",
            "content": {
                "application/json": {
                    "examples": {
                        "unsupported_detector": {
                            "summary": "Unsupported detector query value",
                            "value": {
                                "error": "UNSUPPORTED_DETECTOR",
                                "message": "Detector 'random_forest' is not supported",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        }
                    }
                }
            },
        },
    },
)
def get_model_detail(
    series_id: str,
    detector: str = Query(
        default="gaussian",
        description="Detector type used to scope series lookup. Supported values: gaussian, isolation_forest.",
        openapi_examples={
            "gaussian": {"summary": "Default detector", "value": "gaussian"},
            "isolation_forest": {"summary": "Isolation Forest detector", "value": "isolation_forest"},
        },
    ),
    model_service: ModelService = Depends(get_model_service),
) -> ModelDetail:
    """Return model lineage metadata for one `series_id`."""
    logger.info("Model detail request received", extra={"series_id": series_id, "detector": detector})
    info = model_service.get_model_detail(series_id=series_id, detector=detector)
    logger.info(
        "Model detail request completed",
        extra={"series_id": series_id, "detector": detector, "latest_version": info.latest_version},
    )
    return info


@router.get(
    "/models/{series_id}/versions/{version}",
    response_model=ModelVersionMetadata,
    response_model_exclude_none=True,
    summary="Get metadata for one model version",
    description="Returns persisted metadata for a concrete model version. Training data is optional via `include_data`.",  # noqa: E501
    responses={
        404: {
            "model": ErrorResponse,
            "description": "Series or version not found.",
            "content": {
                "application/json": {
                    "examples": {
                        "series_not_found": {
                            "summary": "Series does not exist",
                            "value": {
                                "error": "SERIES_NOT_FOUND",
                                "message": "Series 'sensor_XYZ' not found",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        },
                        "version_not_found_for_detector": {
                            "summary": "Version missing in selected detector namespace",
                            "value": {
                                "error": "VERSION_NOT_FOUND_FOR_DETECTOR",
                                "message": "Version 'v999' not found for series 'sensor_XYZ' detector 'gaussian'",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        },
                    }
                }
            },
        },
        422: {
            "model": ErrorResponse,
            "description": "Unsupported detector value.",
            "content": {
                "application/json": {
                    "examples": {
                        "unsupported_detector": {
                            "summary": "Unsupported detector query value",
                            "value": {
                                "error": "UNSUPPORTED_DETECTOR",
                                "message": "Detector 'random_forest' is not supported",
                                "detail": None,
                                "timestamp": "2026-04-15T18:00:00Z",
                            },
                        }
                    }
                }
            },
        },
    },
)
def get_model_version_metadata(
    series_id: str,
    version: str,
    detector: str = Query(
        default="gaussian",
        description="Detector type used to scope version lookup. Supported values: gaussian, isolation_forest.",
        openapi_examples={
            "gaussian": {"summary": "Default detector", "value": "gaussian"},
            "isolation_forest": {"summary": "Isolation Forest detector", "value": "isolation_forest"},
        },
    ),
    model_service: ModelService = Depends(get_model_service),
    include_data: bool = Query(
        default=False,
        description="When true, include `training_data` points in the metadata payload.",
    ),
) -> ModelVersionMetadata:
    """Return metadata for one concrete `series_id` model version."""
    logger.info(
        "Model version metadata request received",
        extra={"series_id": series_id, "version": version, "detector": detector, "include_data": include_data},
    )
    payload = model_service.get_version_metadata(
        series_id=series_id,
        version=version,
        detector=detector,
        include_data=include_data,
    )

    logger.info(
        "Model version metadata request completed",
        extra={"series_id": series_id, "version": version, "detector": detector, "include_data": include_data},
    )
    return payload
