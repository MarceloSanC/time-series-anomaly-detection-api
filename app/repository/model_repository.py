import json
import logging
import os
from pathlib import Path
from typing import Any

import joblib

from app.domain.exceptions import InvalidSeriesIdError

logger = logging.getLogger(__name__)

_UNSAFE_TOKENS = ("/", "\\", "..", "\x00")
_INDEX_SCHEMA_VERSION = "1"


class ModelRepository:
    """Filesystem-backed repository for model artifacts and version indexes.

    Storage layout: `{storage_path}/{series_id}/{detector}/{version}/`
    Index location: `{storage_path}/{series_id}/{detector}/index.json`
    """

    def __init__(self, storage_path: Path) -> None:
        """Initialize repository storage root and ensure directory exists."""
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        series_id: str,
        version: str,
        model: Any,
        metadata: dict[str, Any],
        detector: str = "gaussian",
    ) -> None:
        """Persist model + metadata and update detector-scoped index atomically."""
        self._validate_series_id(series_id)
        self._validate_detector(detector)
        detector_path = self.storage_path / series_id / detector
        version_path = detector_path / version
        version_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, version_path / "model.joblib")
        (version_path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8"
        )

        index_path = detector_path / "index.json"
        current_index = self._read_index(index_path)
        if current_index is None:
            versions = [version]
        else:
            versions = list(current_index.get("versions", []))
            if version not in versions:
                versions.append(version)

        new_index = {
            "schema_version": _INDEX_SCHEMA_VERSION,
            "series_id": series_id,
            "detector": detector,
            "latest_version": version,
            "versions": versions,
        }
        self._write_index_atomically(index_path=index_path, payload=new_index)
        logger.info("Model saved", extra={"series_id": series_id, "detector": detector, "version": version})

    def load(self, series_id: str, version: str, detector: str = "gaussian") -> tuple[Any, dict[str, Any]]:
        """Load model and metadata for a given (series_id, detector, version) triple."""
        self._validate_series_id(series_id)
        self._validate_detector(detector)
        version_path = self.storage_path / series_id / detector / version
        model_path = version_path / "model.joblib"
        metadata_path = version_path / "metadata.json"

        if not model_path.exists() or not metadata_path.exists():
            logger.warning(
                "Model not found",
                extra={"series_id": series_id, "detector": detector, "version": version},
            )
            raise FileNotFoundError(
                f"Model or metadata not found for series_id='{series_id}' detector='{detector}' version='{version}'"
            )

        model = joblib.load(model_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        logger.info("Model loaded", extra={"series_id": series_id, "detector": detector, "version": version})
        return model, metadata

    def load_metadata(self, series_id: str, version: str, detector: str = "gaussian") -> dict[str, Any]:
        """Load metadata only for a given (series_id, detector, version) triple."""
        self._validate_series_id(series_id)
        self._validate_detector(detector)
        metadata_path = self.storage_path / series_id / detector / version / "metadata.json"
        if not metadata_path.exists():
            logger.warning(
                "Metadata not found",
                extra={"series_id": series_id, "detector": detector, "version": version},
            )
            raise FileNotFoundError(
                f"Metadata not found for series_id='{series_id}' detector='{detector}' version='{version}'"
            )
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def get_index(self, series_id: str, detector: str = "gaussian") -> dict[str, Any] | None:
        """Return detector-scoped index for a series, or None when not found."""
        self._validate_series_id(series_id)
        self._validate_detector(detector)
        index_path = self.storage_path / series_id / detector / "index.json"
        return self._read_index(index_path)

    def list_all(self) -> list[dict[str, Any]]:
        """List all available (series, detector) indexes stored in the repository."""
        indexes: list[dict[str, Any]] = []
        if not self.storage_path.exists():
            return indexes

        for series_dir in sorted(self.storage_path.iterdir()):
            if not series_dir.is_dir():
                continue
            for detector_dir in sorted(series_dir.iterdir()):
                if not detector_dir.is_dir():
                    continue
                index_path = detector_dir / "index.json"
                if index_path.exists():
                    indexes.append(json.loads(index_path.read_text(encoding="utf-8")))
        return indexes

    def version_exists(self, series_id: str, version: str, detector: str = "gaussian") -> bool:
        """Check whether a version directory exists for a (series_id, detector) pair."""
        self._validate_series_id(series_id)
        self._validate_detector(detector)
        version_path = self.storage_path / series_id / detector / version
        return version_path.exists() and version_path.is_dir()

    def _read_index(self, index_path: Path) -> dict[str, Any] | None:
        """Read and parse an index file, returning None if it does not exist."""
        if not index_path.exists():
            return None
        return json.loads(index_path.read_text(encoding="utf-8"))

    def _write_index_atomically(self, index_path: Path, payload: dict[str, Any]) -> None:
        """Write index via temp file and atomic replace to avoid corruption."""
        tmp_path = index_path.with_suffix(index_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
        os.replace(tmp_path, index_path)

    @staticmethod
    def _validate_series_id(series_id: str) -> None:
        """Reject empty or unsafe series identifiers for filesystem paths."""
        if not series_id or any(token in series_id for token in _UNSAFE_TOKENS):
            raise InvalidSeriesIdError(f"Invalid series_id: '{series_id}'")

    @staticmethod
    def _validate_detector(detector: str) -> None:
        """Reject empty or unsafe detector strings to prevent path traversal."""
        if not detector or any(token in detector for token in _UNSAFE_TOKENS):
            raise ValueError(f"Invalid detector path component: '{detector}'")
