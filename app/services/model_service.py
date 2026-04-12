from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
import logging
from time import perf_counter
from typing import Any, ContextManager, Protocol

from app.domain.exceptions import PlotDataUnavailableError, SeriesNotFoundError, VersionNotFoundError
from app.domain.models import AnomalyDetectionModel
from app.domain.schemas import DataPoint, ModelInfo, PredictionResponse, TimeSeries, TrainResponse
from app.repository.model_repository import ModelRepository
from app.services.validation_service import ValidationService

logger = logging.getLogger(__name__)


class SupportsSeriesLock(Protocol):
    """Protocol for lock providers keyed by `series_id`."""

    def get_lock(self, series_id: str) -> ContextManager[object]:
        ...


class ModelService:
    """Service layer that orchestrates train/predict/versioning operations."""

    def __init__(
        self,
        repository: ModelRepository,
        lock_manager: SupportsSeriesLock | None = None,
        validation_service: ValidationService | None = None,
    ) -> None:
        """Initialize service with repository and optional lock manager."""
        self.repository = repository
        self.lock_manager = lock_manager
        self.validation_service = ValidationService() if validation_service is None else validation_service

    def train(self, series_id: str, data: TimeSeries) -> TrainResponse:
        """Train and persist a new model version for a series."""
        self.validation_service.validate_training_data(data)
        lock_ctx = self.lock_manager.get_lock(series_id) if self.lock_manager is not None else nullcontext()
        logger.info("Training started", extra={"series_id": series_id})

        with lock_ctx:
            # Measure effective training time only (exclude lock wait time).
            start = perf_counter()
            version = self._next_version(series_id)
            model = AnomalyDetectionModel().fit(data)

            trained_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            values = [d.value for d in data.data]
            timestamps = [d.timestamp for d in data.data]
            duration_ms = (perf_counter() - start) * 1000

            metadata = {
                "version": version,
                "mean": model.mean,
                "std": model.std,
                "n_samples": len(values),
                "trained_at": trained_at,
                "training_duration_ms": duration_ms,
                "data_range": {
                    "min_timestamp": min(timestamps),
                    "max_timestamp": max(timestamps),
                },
                "training_data": [
                    {"timestamp": point.timestamp, "value": point.value}
                    for point in data.data
                ],
            }
            self.repository.save(series_id=series_id, version=version, model=model, metadata=metadata)
            logger.info("Training completed", extra={"series_id": series_id, "version": version})

            return TrainResponse(
                series_id=series_id,
                version=version,
                n_samples=metadata["n_samples"],
                mean=metadata["mean"],
                std=metadata["std"],
                training_duration_ms=metadata["training_duration_ms"],
                trained_at=metadata["trained_at"],
            )

    def predict(self, series_id: str, data_point: DataPoint, version: str | None = None) -> PredictionResponse:
        """Run anomaly prediction using latest or a specific model version."""
        resolved_version = self._resolve_version(series_id=series_id, version=version)
        model, _metadata = self.repository.load(series_id=series_id, version=resolved_version)

        is_anomaly = bool(model.predict(data_point))
        logger.info(
            "Prediction completed",
            extra={
                "series_id": series_id,
                "version": resolved_version,
                "is_anomaly": is_anomaly,
            },
        )
        return PredictionResponse(
            series_id=series_id,
            version=resolved_version,
            is_anomaly=is_anomaly,
            value=data_point.value,
            timestamp=data_point.timestamp,
            mean=float(model.mean),
            upper_bound=float(model.mean + 3 * model.std),
        )

    def list_series(self) -> list[dict[str, Any]]:
        """List all tracked series index entries."""
        return self.repository.list_all()

    def get_series_info(self, series_id: str) -> ModelInfo:
        """Return metadata for the latest trained version of a series."""
        index = self.repository.get_index(series_id)
        if index is None:
            raise SeriesNotFoundError(f"Series '{series_id}' not found")

        latest_version = index["latest_version"]
        _, metadata = self.repository.load(series_id=series_id, version=latest_version)

        return ModelInfo(
            series_id=series_id,
            latest_version=latest_version,
            versions=index["versions"],
            trained_at=metadata["trained_at"],
            n_samples=metadata["n_samples"],
        )

    def get_plot_data(self, series_id: str, version: str | None = None) -> dict[str, Any]:
        """Return metadata fields required to render training data visualization."""
        resolved_version = self._resolve_version(series_id=series_id, version=version)
        try:
            metadata = self.repository.load_metadata(series_id=series_id, version=resolved_version)
        except FileNotFoundError as exc:
            raise PlotDataUnavailableError(
                f"Plot metadata not available for series '{series_id}' version '{resolved_version}'"
            ) from exc
        training_data = metadata.get("training_data")
        if not isinstance(training_data, list) or not training_data:
            raise PlotDataUnavailableError(
                f"Plot data not available for series '{series_id}' version '{resolved_version}'"
            )
        return {
            "series_id": series_id,
            "version": resolved_version,
            "mean": metadata["mean"],
            "std": metadata["std"],
            "training_data": training_data,
        }

    def _next_version(self, series_id: str) -> str:
        """Compute next incremental version label for a series."""
        index = self.repository.get_index(series_id)
        if index is None:
            return "v1"

        latest = str(index.get("latest_version", "v0"))
        try:
            latest_number = int(latest[1:])
        except (TypeError, ValueError):
            latest_number = 0
        return f"v{latest_number + 1}"

    def _resolve_version(self, series_id: str, version: str | None) -> str:
        """Resolve requested version or fallback to latest available version."""
        index = self.repository.get_index(series_id)
        if index is None:
            raise SeriesNotFoundError(f"Series '{series_id}' not found")

        if version is None:
            return str(index["latest_version"])

        if not self.repository.version_exists(series_id=series_id, version=version):
            logger.warning("Version not found", extra={"series_id": series_id, "version": version})
            raise VersionNotFoundError(f"Version '{version}' not found for series '{series_id}'")

        return version
