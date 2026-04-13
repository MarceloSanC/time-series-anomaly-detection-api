from __future__ import annotations


class ValidationServiceError(ValueError):
    """Base class for business-rule validation errors during training."""


class InsufficientDataError(ValidationServiceError):
    """Raised when a series has fewer points than configured minimum."""


class ConstantSeriesError(ValidationServiceError):
    """Raised when a series standard deviation is effectively zero."""


class DuplicateTimestampsError(ValidationServiceError):
    """Raised when duplicate timestamps are found in training data."""


class UnorderedTimestampsError(ValidationServiceError):
    """Raised when timestamps are not strictly increasing."""


class InvalidValuesError(ValidationServiceError):
    """Raised when training values contain NaN or infinity."""


class SeriesNotFoundError(Exception):
    """Raised when requested series_id does not exist in repository."""


class VersionNotFoundError(Exception):
    """Raised when requested model version is missing for a series."""


class InvalidSeriesIdError(ValueError):
    """Raised when series_id is empty or unsafe for filesystem-backed storage."""


class PlotDataUnavailableError(Exception):
    """Raised when required training points are missing for plot generation."""


class MetadataIncompleteError(Exception):
    """Raised when model metadata is missing/incomplete for introspection."""
