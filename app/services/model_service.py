from __future__ import annotations

from contextlib import nullcontext
from datetime import UTC, datetime
import logging
from time import perf_counter
from typing import Any, ContextManager, Protocol

from app.domain.exceptions import (
    MetadataIncompleteError,
    PlotDataUnavailableError,
    SeriesNotFoundError,
    UnsupportedDetectorError,
    VersionNotFoundForDetectorError,
)
from app.domain.models import AnomalyDetectionModel, IsolationForestDetector
from app.domain.schemas import (
    DataPoint,
    DataQualityReport,
    DetectorType,
    ModelDetail,
    ModelInfo,
    ModelSummary,
    ModelVersionMetadata,
    PredictionResponse,
    TimeSeries,
    TrainResponse,
)
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

    def train(self, series_id: str, data: TimeSeries, detector: DetectorType = "gaussian") -> TrainResponse:
        """Train and persist a new model version for a series."""
        self._validate_detector_type(detector)
        self.validation_service.validate_training_data(data)
        lock_ctx = self.lock_manager.get_lock(series_id) if self.lock_manager is not None else nullcontext()
        logger.info("Training started", extra={"series_id": series_id, "detector": detector})

        with lock_ctx:
            # Measure effective training time only (exclude lock wait time).
            start = perf_counter()
            version = self._next_version(series_id, detector=detector)
            model = self._make_detector(detector).fit(data)

            trained_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            values = [d.value for d in data.data]
            timestamps = [d.timestamp for d in data.data]
            duration_ms = (perf_counter() - start) * 1000

            model_params = self._extract_train_params(model, detector)
            metadata = {
                "version": version,
                "detector": detector,
                "model_params": model_params,
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
            self.repository.save(
                series_id=series_id, version=version, model=model, metadata=metadata, detector=detector
            )
            logger.info("Training completed", extra={"series_id": series_id, "detector": detector, "version": version})

            return TrainResponse(
                series_id=series_id,
                version=version,
                n_samples=metadata["n_samples"],
                detector=detector,
                model_params=model_params,
                training_duration_ms=metadata["training_duration_ms"],
                trained_at=metadata["trained_at"],
            )

    def predict(
        self,
        series_id: str,
        data_point: DataPoint,
        version: str | None = None,
        detector: DetectorType = "gaussian",
    ) -> PredictionResponse:
        """Run anomaly prediction using latest or a specific model version."""
        self._validate_detector_type(detector)
        resolved_version = self._resolve_version(series_id=series_id, version=version, detector=detector)
        model, _metadata = self.repository.load(series_id=series_id, version=resolved_version, detector=detector)

        is_anomaly = bool(model.predict(data_point))
        predict_params = self._extract_predict_params(model, detector)
        logger.info(
            "Prediction completed",
            extra={
                "series_id": series_id,
                "detector": detector,
                "version": resolved_version,
                "value": data_point.value,
                "is_anomaly": is_anomaly,
                **({} if predict_params is None else predict_params),
            },
        )
        return PredictionResponse(
            series_id=series_id,
            version=resolved_version,
            is_anomaly=is_anomaly,
            value=data_point.value,
            timestamp=data_point.timestamp,
            detector=detector,
            model_params=predict_params,
        )

    def list_series(self) -> list[dict[str, Any]]:
        """List all tracked series index entries."""
        return self.repository.list_all()

    def list_model_summaries(self, strict: bool = False) -> list[ModelSummary]:
        """List all series with latest-version summary metadata.

        When `strict=True`, missing metadata raises MetadataIncompleteError.
        """
        summaries: list[ModelSummary] = []
        for index in self.repository.list_all():
            series_id = str(index["series_id"])
            latest_version = str(index["latest_version"])
            try:
                detector = str(index.get("detector", "gaussian"))
                metadata = self.repository.load_metadata(series_id=series_id, version=latest_version, detector=detector)
            except FileNotFoundError:
                if strict:
                    raise MetadataIncompleteError(
                        f"Metadata missing for series '{series_id}' at latest version '{latest_version}'"
                    )
                logger.warning(
                    "Skipping series with missing latest metadata",
                    extra={"series_id": series_id, "version": latest_version},
                )
                continue
            summaries.append(
                ModelSummary(
                    series_id=series_id,
                    latest_version=latest_version,
                    n_samples=int(metadata["n_samples"]),
                    trained_at=str(metadata["trained_at"]),
                )
            )
        return summaries

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

    def get_model_detail(self, series_id: str) -> ModelDetail:
        """Return detail payload for one series including derived data quality."""
        info = self.get_series_info(series_id=series_id)
        metadata = self.repository.load_metadata(series_id=series_id, version=info.latest_version)
        quality = self._build_data_quality(metadata=metadata, n_samples=info.n_samples)
        return ModelDetail(
            series_id=info.series_id,
            latest_version=info.latest_version,
            versions=info.versions,
            trained_at=info.trained_at,
            n_samples=info.n_samples,
            data_quality=quality,
        )

    def get_version_metadata(
        self,
        series_id: str,
        version: str,
        include_data: bool = False,
    ) -> ModelVersionMetadata:
        """Return metadata for one concrete model version of a series."""
        resolved_version = self._resolve_version(series_id=series_id, version=version)
        metadata = self.repository.load_metadata(series_id=series_id, version=resolved_version)
        stored_detector = str(metadata.get("detector", "gaussian"))
        # Read model_params from metadata; fall back to top-level mean/std for legacy format.
        stored_params: dict[str, Any] | None = metadata.get("model_params")
        if stored_params is None and "mean" in metadata:
            stored_params = {"mean": float(metadata["mean"]), "std": float(metadata["std"])}
        payload: dict[str, Any] = {
            "version": resolved_version,
            "detector": stored_detector,
            "model_params": stored_params,
            "n_samples": int(metadata["n_samples"]),
            "trained_at": str(metadata["trained_at"]),
            "training_duration_ms": float(metadata["training_duration_ms"]),
            "data_range": metadata["data_range"],
        }
        if include_data:
            payload["training_data"] = metadata.get("training_data", [])
        return ModelVersionMetadata.model_validate(payload)

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
        stored_params = metadata.get("model_params") or {}
        mean = stored_params.get("mean") if stored_params else metadata.get("mean")
        std = stored_params.get("std") if stored_params else metadata.get("std")
        return {
            "series_id": series_id,
            "version": resolved_version,
            "mean": mean,
            "std": std,
            "training_data": training_data,
        }

    def _next_version(self, series_id: str, detector: str = "gaussian") -> str:
        """Compute next incremental version label for a (series_id, detector) pair."""
        index = self.repository.get_index(series_id, detector=detector)
        if index is None:
            return "v1"

        latest = str(index.get("latest_version", "v0"))
        try:
            latest_number = int(latest[1:])
        except (TypeError, ValueError):
            latest_number = 0
        return f"v{latest_number + 1}"

    def _resolve_version(self, series_id: str, version: str | None, detector: str = "gaussian") -> str:
        """Resolve requested version or fallback to latest for a (series_id, detector) pair."""
        index = self.repository.get_index(series_id, detector=detector)
        if index is None:
            raise SeriesNotFoundError(f"Series '{series_id}' not found")

        if version is None:
            return str(index["latest_version"])

        if not self.repository.version_exists(series_id=series_id, version=version, detector=detector):
            logger.warning(
                "Version not found", extra={"series_id": series_id, "detector": detector, "version": version}
            )
            raise VersionNotFoundForDetectorError(
                f"Version '{version}' not found for series '{series_id}' detector '{detector}'"
            )

        return version

    @staticmethod
    def _build_data_quality(metadata: dict[str, Any], n_samples: int) -> DataQualityReport:
        """Derive deterministic quality indicators from persisted metadata."""
        training_data = metadata.get("training_data", [])
        values = [
            float(point["value"])
            for point in training_data
            if isinstance(point, dict) and "value" in point
        ]
        # Read mean/std from model_params (new format) with fallback to top-level (legacy format).
        stored_params = metadata.get("model_params") or {}
        mean_raw = stored_params.get("mean") if stored_params else metadata.get("mean")
        if mean_raw is None:
            mean_raw = metadata.get("mean")
        std_raw = stored_params.get("std") if stored_params else metadata.get("std")
        if std_raw is None:
            std_raw = metadata.get("std")
        mean = float(mean_raw) if mean_raw is not None else None
        std = float(std_raw) if std_raw is not None else None

        min_value = min(values) if values else (mean if mean is not None else 0.0)
        max_value = max(values) if values else (mean if mean is not None else 0.0)

        data_range = metadata["data_range"]
        min_ts = int(data_range["min_timestamp"])
        max_ts = int(data_range["max_timestamp"])
        time_span_seconds = max(max_ts - min_ts, 0)
        points_per_second = float(n_samples / max(time_span_seconds, 1))

        return DataQualityReport(
            n_samples=n_samples,
            mean=mean,
            std=std,
            min_value=min_value,
            max_value=max_value,
            time_span_seconds=time_span_seconds,
            points_per_second=points_per_second,
        )

    @staticmethod
    def _validate_detector_type(detector: str) -> None:
        """Raise UnsupportedDetectorError before any repository call for unknown detector types."""
        if detector not in ("gaussian", "isolation_forest"):
            raise UnsupportedDetectorError(f"Detector '{detector}' is not supported")

    @staticmethod
    def _make_detector(detector: DetectorType) -> Any:
        """Instantiate the correct detector class for the given detector type."""
        if detector == "gaussian":
            return AnomalyDetectionModel()
        if detector == "isolation_forest":
            return IsolationForestDetector()
        raise UnsupportedDetectorError(f"Detector '{detector}' is not supported")

    @staticmethod
    def _extract_train_params(model: Any, detector: DetectorType) -> dict[str, Any]:
        """Extract detector-specific model parameters after training."""
        if detector == "gaussian":
            return {"mean": float(model.mean), "std": float(model.std)}
        if detector == "isolation_forest":
            return {"n_estimators": 100, "contamination": "auto", "score_threshold": model.score_threshold}
        return {}

    @staticmethod
    def _extract_predict_params(model: Any, detector: DetectorType) -> dict[str, Any] | None:
        """Extract detector-specific parameters for prediction response."""
        if detector == "gaussian":
            upper_bound = float(model.mean + 3 * model.std)
            return {"mean": float(model.mean), "upper_bound": upper_bound}
        if detector == "isolation_forest":
            return {"score_threshold": float(model.score_threshold)}
        return None
