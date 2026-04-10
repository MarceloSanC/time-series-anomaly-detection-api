import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)


class ModelRepository:
    def __init__(self, storage_path: Path) -> None:
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save(self, series_id: str, version: str, model: Any, metadata: dict[str, Any]) -> None:
        self._validate_series_id(series_id)
        series_path = self.storage_path / series_id
        version_path = series_path / version
        version_path.mkdir(parents=True, exist_ok=True)

        model_path = version_path / "model.joblib"
        metadata_path = version_path / "metadata.json"
        index_path = series_path / "index.json"

        joblib.dump(model, model_path)
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")

        current_index = self.get_index(series_id)
        if current_index is None:
            versions = [version]
        else:
            versions = list(current_index.get("versions", []))
            if version not in versions:
                versions.append(version)

        new_index = {
            "series_id": series_id,
            "latest_version": version,
            "versions": versions,
        }
        self._write_index_atomically(index_path=index_path, payload=new_index)
        logger.info("Model saved", extra={"series_id": series_id, "version": version})

    def load(self, series_id: str, version: str) -> tuple[Any, dict[str, Any]]:
        self._validate_series_id(series_id)
        version_path = self.storage_path / series_id / version
        model_path = version_path / "model.joblib"
        metadata_path = version_path / "metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            logger.warning("Model not found", extra={"series_id": series_id, "version": version})
            raise FileNotFoundError(f"Model or metadata not found for series_id='{series_id}' version='{version}'")

        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        logger.info("Model loaded", extra={"series_id": series_id, "version": version})
        return model, metadata

    def get_index(self, series_id: str) -> dict[str, Any] | None:
        self._validate_series_id(series_id)
        index_path = self.storage_path / series_id / "index.json"
        if not index_path.exists():
            return None
        return json.loads(index_path.read_text(encoding="utf-8"))

    def list_all(self) -> list[dict[str, Any]]:
        indexes: list[dict[str, Any]] = []
        if not self.storage_path.exists():
            return indexes

        for series_dir in sorted(self.storage_path.iterdir()):
            if not series_dir.is_dir():
                continue
            index_path = series_dir / "index.json"
            if index_path.exists():
                indexes.append(json.loads(index_path.read_text(encoding="utf-8")))
        return indexes

    def version_exists(self, series_id: str, version: str) -> bool:
        self._validate_series_id(series_id)
        version_path = self.storage_path / series_id / version
        return version_path.exists() and version_path.is_dir()

    def _write_index_atomically(self, index_path: Path, payload: dict[str, Any]) -> None:
        tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        os.replace(tmp_path, index_path)

    @staticmethod
    def _validate_series_id(series_id: str) -> None:
        invalid_tokens = ("/", "\\", "..", "\x00")
        if not series_id or any(token in series_id for token in invalid_tokens):
            raise ValueError(f"Invalid series_id: '{series_id}'")
