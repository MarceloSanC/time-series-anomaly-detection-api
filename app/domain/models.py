import numpy as np

from app.domain.schemas import DataPoint, TimeSeries


class AnomalyDetectionModel:
    mean: float
    std: float

    def fit(self, data: TimeSeries) -> "AnomalyDetectionModel":
        values = [d.value for d in data.data]
        self.mean = float(np.mean(values))
        self.std = float(np.std(values))
        return self

    def predict(self, data_point: DataPoint) -> bool:
        return data_point.value > self.mean + 3 * self.std
