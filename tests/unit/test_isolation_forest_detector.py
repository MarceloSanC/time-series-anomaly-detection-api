from __future__ import annotations

import pytest

from app.domain.models import DetectorProtocol, IsolationForestDetector
from app.domain.exceptions import UnsupportedDetectorError, VersionNotFoundForDetectorError
from app.domain.schemas import DataPoint, TimeSeries


def _make_series(values: list[float], start_ts: int = 1700000000) -> TimeSeries:
    return TimeSeries(
        data=[DataPoint(timestamp=start_ts + i, value=v) for i, v in enumerate(values)]
    )


def _normal_series(n: int = 100, mean: float = 10.0, std: float = 0.5, seed: int = 42) -> TimeSeries:
    """Generate a series sampled from a normal distribution for robust IsolationForest training."""
    rng = __import__("random").Random(seed)
    values = [mean + rng.gauss(0, std) for _ in range(n)]
    return _make_series(values)


class TestIsolationForestDetector:
    def test_fit_returns_self(self) -> None:
        """fit() must return the detector instance for method chaining."""
        detector = IsolationForestDetector()
        result = detector.fit(_normal_series())
        assert result is detector

    def test_score_threshold_set_after_fit(self) -> None:
        """score_threshold property must be accessible after fit."""
        detector = IsolationForestDetector().fit(_normal_series())
        assert isinstance(detector.score_threshold, float)

    def test_predict_normal_point(self) -> None:
        """Point within training distribution must not be flagged as anomaly."""
        detector = IsolationForestDetector().fit(_normal_series())
        assert detector.predict(DataPoint(timestamp=1700001000, value=10.1)) is False

    def test_predict_anomalous_point(self) -> None:
        """Point far outside training distribution must be flagged as anomaly.

        IsolationForest requires varied training data (not uniform) to correctly
        score extreme outliers — uniform data causes masking of distant points.
        """
        detector = IsolationForestDetector().fit(_normal_series())
        assert detector.predict(DataPoint(timestamp=1700001000, value=9999.0)) is True

    def test_predict_raises_before_fit(self) -> None:
        """predict() must raise RuntimeError when called before fit()."""
        detector = IsolationForestDetector()
        with pytest.raises(RuntimeError):
            detector.predict(DataPoint(timestamp=1, value=1.0))

    def test_satisfies_detector_protocol(self) -> None:
        """IsolationForestDetector must satisfy DetectorProtocol at runtime."""
        assert isinstance(IsolationForestDetector(), DetectorProtocol)


class TestDetectorExceptions:
    def test_unsupported_detector_error_is_value_error(self) -> None:
        exc = UnsupportedDetectorError("unknown_detector")
        assert isinstance(exc, ValueError)
        assert "unknown_detector" in str(exc)

    def test_version_not_found_for_detector_error(self) -> None:
        exc = VersionNotFoundForDetectorError("v99 not found for isolation_forest")
        assert "v99" in str(exc)
        assert "isolation_forest" in str(exc)
