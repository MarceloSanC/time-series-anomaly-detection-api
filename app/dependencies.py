from __future__ import annotations

from fastapi import Request

from app.services.metrics_service import MetricsService
from app.services.model_service import ModelService
from app.utils.concurrency import LockManager


def get_lock_manager(request: Request) -> LockManager:
    """Return the shared LockManager instance from app state."""
    return request.app.state.lock_manager


def get_metrics_service(request: Request) -> MetricsService:
    """Return the shared MetricsService instance from app state."""
    return request.app.state.metrics_service


def get_model_service(request: Request) -> ModelService:
    """Build ModelService with request-scoped access to shared infra."""
    lock_manager = get_lock_manager(request)
    repository = request.app.state.model_repository
    return ModelService(repository=repository, lock_manager=lock_manager)
