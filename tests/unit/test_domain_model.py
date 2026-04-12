from __future__ import annotations

import pytest

from app.domain.models import AnomalyDetectionModel
from app.domain.schemas import DataPoint


def test_predict_raises_when_model_not_fitted() -> None:
    """Reject predict() calls before fit() initializes mean and std."""
    model = AnomalyDetectionModel()

    with pytest.raises(RuntimeError):
        model.predict(DataPoint(timestamp=1, value=1.0))
