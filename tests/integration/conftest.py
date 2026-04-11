from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import create_app


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Provide an isolated API client with per-test storage directory."""
    # Patch happens before create_app(); repository reads settings.storage_path at runtime
    # inside lifespan, so each test gets its own isolated storage root.
    monkeypatch.setattr(settings, "storage_path", tmp_path)
    app = create_app()
    with TestClient(app) as test_client:
        yield test_client
