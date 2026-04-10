from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
from time import perf_counter
from typing import Any, Protocol

from app.domain.models import AnomalyDetectionModel
from app.domain.schemas import DataPoint, ModelInfo, PredictionResponse, TimeSeries, TrainResponse
from app.repository.model_repository import ModelRepository


class SupportsSeriesLock(Protocol):
    def get_lock(self, series_id: str) -> Any:
        ...


class SeriesNotFoundError(Exception):
    pass


class VersionNotFoundError(Exception):
    pass


class ModelService:
    def __init__(self, repository: ModelRepository, lock_manager: SupportsSeriesLock | None = None) -> None:
        self.repository = repository
        self.lock_manager = lock_manager

    def train(self, series_id: str, data: TimeSeries) -> TrainResponse:
        start = perf_counter()
        lock_ctx = self.lock_manager.get_lock(series_id) if self.lock_manager is not None else nullcontext()

        with lock_ctx:
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
            }
            self.repository.save(series_id=series_id, version=version, model=model, metadata=metadata)

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
        resolved_version = self._resolve_version(series_id=series_id, version=version)
        model, _metadata = self.repository.load(series_id=series_id, version=resolved_version)

        is_anomaly = bool(model.predict(data_point))
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
        return self.repository.list_all()

    def get_series_info(self, series_id: str) -> ModelInfo:
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

    def _next_version(self, series_id: str) -> str:
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
        index = self.repository.get_index(series_id)
        if index is None:
            raise SeriesNotFoundError(f"Series '{series_id}' not found")

        if version is None:
            return str(index["latest_version"])

        if not self.repository.version_exists(series_id=series_id, version=version):
            raise VersionNotFoundError(f"Version '{version}' not found for series '{series_id}'")

        return version
