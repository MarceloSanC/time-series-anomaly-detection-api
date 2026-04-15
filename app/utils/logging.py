from __future__ import annotations

import json
import logging
import uuid
from contextvars import ContextVar, Token

request_id_var: ContextVar[str] = ContextVar("request_id", default="none")

# Standard LogRecord attributes that should not be repeated in the JSON "extra" block.
_STDLIB_LOG_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    # Fields we promote to top-level keys in the JSON payload.
    "request_id",
})


class RequestIdFilter(logging.Filter):
    """Inject current request_id from ContextVar into each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("none")
        return True


class JsonFormatter(logging.Formatter):
    """Emit one JSON object per log line with a standard set of top-level fields.

    Top-level fields always present:
        timestamp, level, logger, message, request_id

    Any extra fields passed via ``logger.info(..., extra={...})`` are merged
    into the object after the standard fields.
    """

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        payload: dict = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "request_id": getattr(record, "request_id", "none"),
        }
        for key, value in record.__dict__.items():
            if key not in _STDLIB_LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_level: str = "INFO", log_format: str = "text") -> None:
    """Configure root logging format and ensure RequestIdFilter is attached.

    Args:
        log_level:  Standard Python log level string (e.g. ``"INFO"``).
        log_format: ``"text"`` for human-readable output (default);
                    ``"json"`` for structured JSON lines.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    if log_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(request_id)s] %(levelname)s %(name)s: %(message)s"
        )

    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.addFilter(RequestIdFilter())
        root_logger.addHandler(handler)
        return

    for handler in root_logger.handlers:
        handler.setFormatter(formatter)
        has_request_filter = any(isinstance(f, RequestIdFilter) for f in handler.filters)
        if not has_request_filter:
            handler.addFilter(RequestIdFilter())


def set_request_id(request_id: str | None = None) -> Token[str]:
    """Set request_id in current context and return token for later reset."""
    if request_id is None:
        request_id = str(uuid.uuid4())
    return request_id_var.set(request_id)


def reset_request_id(token: Token[str]) -> None:
    """Restore previous request_id context using the provided token."""
    request_id_var.reset(token)


def get_request_id() -> str:
    """Get request_id from current context or fallback value."""
    return request_id_var.get("none")
