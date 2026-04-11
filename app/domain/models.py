import numpy as np

from app.domain.schemas import DataPoint, TimeSeries


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
