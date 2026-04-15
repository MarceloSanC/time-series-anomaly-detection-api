from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.dependencies import get_model_service
from app.domain.exceptions import UnsupportedDetectorError, VersionNotFoundForDetectorError
from app.main import create_app


def _stub_service_raising(exc: Exception) -> object:
    """Return a stub model service whose train/predict methods raise the given exception."""

    def _raise(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise exc

    class _StubService:
        train = _raise
        predict = _raise

    return _StubService()


_FIT_PAYLOAD = {
    "timestamps": [1700000000 + i for i in range(30)],
    "values": [10.0 + i * 0.1 for i in range(30)],
}


@pytest.fixture
def client_unsupported(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    exc = UnsupportedDetectorError("unknown_detector is not supported")
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: _stub_service_raising(exc)
    # Context manager triggers lifespan startup so metrics_service is initialized.
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


@pytest.fixture
def client_version_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    exc = VersionNotFoundForDetectorError("v99 not found for isolation_forest/sensor_A")
    app = create_app()
    app.dependency_overrides[get_model_service] = lambda: _stub_service_raising(exc)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


def test_unsupported_detector_returns_422(client_unsupported: TestClient) -> None:
    """UnsupportedDetectorError must map to HTTP 422 with UNSUPPORTED_DETECTOR code."""
    response = client_unsupported.post("/fit/sensor_A", json=_FIT_PAYLOAD)
    assert response.status_code == 422
    body = response.json()
    assert body["error"] == "UNSUPPORTED_DETECTOR"
    assert "unknown_detector" in body["message"]


def test_version_not_found_for_detector_returns_404(client_version_missing: TestClient) -> None:
    """VersionNotFoundForDetectorError must map to HTTP 404 with VERSION_NOT_FOUND_FOR_DETECTOR code."""
    response = client_version_missing.post("/fit/sensor_A", json=_FIT_PAYLOAD)
    assert response.status_code == 404
    body = response.json()
    assert body["error"] == "VERSION_NOT_FOUND_FOR_DETECTOR"
    assert "v99" in body["message"]
