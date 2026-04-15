from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from sklearn.ensemble import IsolationForest

from app.domain.schemas import DataPoint, TimeSeries


@runtime_checkable
class DetectorProtocol(Protocol):
    """Interface contract for all anomaly detector implementations."""

    def fit(self, data: TimeSeries) -> "DetectorProtocol":
        ...

    def predict(self, data_point: DataPoint) -> bool:
        ...


class AnomalyDetectionModel:
    """Simple anomaly detector based on mean and standard deviation."""

    mean: float
    std: float

    def fit(self, data: TimeSeries) -> "AnomalyDetectionModel":
        """Learn `mean` and `std` from the input time series."""
        values = [d.value for d in data.data]
        self.mean = float(np.mean(values))
        self.std = float(np.std(values))
        return self

    def predict(self, data_point: DataPoint) -> bool:
        """Return True when value is above `mean + 3 * std`."""
        if not hasattr(self, "mean") or not hasattr(self, "std"):
            raise RuntimeError("Model must be fitted before calling predict()")
        return data_point.value > self.mean + 3 * self.std


class IsolationForestDetector:
    """Anomaly detector using Isolation Forest algorithm."""

    def fit(self, data: TimeSeries) -> "IsolationForestDetector":
        """Fit Isolation Forest on training values and derive anomaly threshold."""
        values = [[p.value] for p in data.data]
        self._clf = IsolationForest(
            n_estimators=100,
            contamination="auto",
            random_state=42,
        ).fit(values)
        scores = self._clf.score_samples(values)
        self._threshold = float(np.percentile(scores, 10))
        return self

    def predict(self, data_point: DataPoint) -> bool:
        """Return True when the point score falls below the training threshold."""
        if not hasattr(self, "_clf"):
            raise RuntimeError("Model must be fitted before calling predict()")
        score = float(self._clf.score_samples([[data_point.value]])[0])
        return score < self._threshold

    @property
    def score_threshold(self) -> float:
        """Anomaly threshold derived from the bottom 10th percentile of training scores."""
        return self._threshold
