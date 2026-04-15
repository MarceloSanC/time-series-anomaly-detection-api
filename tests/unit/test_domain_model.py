from __future__ import annotations

import pytest

from app.domain.models import AnomalyDetectionModel, DetectorProtocol
from app.domain.schemas import DataPoint


def test_predict_raises_when_model_not_fitted() -> None:
    """Reject predict() calls before fit() initializes mean and std."""
    model = AnomalyDetectionModel()

    with pytest.raises(RuntimeError):
        model.predict(DataPoint(timestamp=1, value=1.0))


def test_anomaly_detection_model_satisfies_detector_protocol() -> None:
    """AnomalyDetectionModel must satisfy DetectorProtocol at runtime."""
    assert isinstance(AnomalyDetectionModel(), DetectorProtocol)
