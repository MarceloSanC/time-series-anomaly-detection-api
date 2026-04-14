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
    responses={422: {"model": ErrorResponse, "description": "Incomplete metadata found when `strict=true`."}},
)
def list_models(
    strict: bool = Query(
        default=False,
        description="When true, fail-fast if any series has incomplete latest metadata.",
    ),
    model_service: ModelService = Depends(get_model_service),
) -> list[ModelSummary]:
    """List all tracked series with latest version and summary metadata."""
    logger.info("Models list request received", extra={"strict": strict})
    summaries = model_service.list_model_summaries(strict=strict)
    logger.info("Models list request completed", extra={"series_count": len(summaries), "strict": strict})
    return summaries


@router.get(
    "/models/{series_id}",
    response_model=ModelDetail,
    summary="Get model detail for one series",
    description="Returns lineage metadata and derived data-quality indicators for the latest model version.",
    responses={404: {"model": ErrorResponse, "description": "Series not found."}},
)
def get_model_detail(
    series_id: str,
    model_service: ModelService = Depends(get_model_service),
) -> ModelDetail:
    """Return model lineage metadata for one `series_id`."""
    logger.info("Model detail request received", extra={"series_id": series_id})
    info = model_service.get_model_detail(series_id=series_id)
    logger.info(
        "Model detail request completed",
        extra={"series_id": series_id, "latest_version": info.latest_version},
    )
    return info


@router.get(
    "/models/{series_id}/versions/{version}",
    response_model=ModelVersionMetadata,
    response_model_exclude_none=True,
    summary="Get metadata for one model version",
    description="Returns persisted metadata for a concrete model version. Training data is optional via `include_data`.",
    responses={404: {"model": ErrorResponse, "description": "Series or version not found."}},
)
def get_model_version_metadata(
    series_id: str,
    version: str,
    model_service: ModelService = Depends(get_model_service),
    include_data: bool = Query(
        default=False,
        description="When true, include `training_data` points in the metadata payload.",
    ),
) -> ModelVersionMetadata:
    """Return metadata for one concrete `series_id` model version."""
    logger.info(
        "Model version metadata request received",
        extra={"series_id": series_id, "version": version, "include_data": include_data},
    )
    payload = model_service.get_version_metadata(
        series_id=series_id,
        version=version,
        include_data=include_data,
    )

    logger.info(
        "Model version metadata request completed",
        extra={"series_id": series_id, "version": version, "include_data": include_data},
    )
    return payload
