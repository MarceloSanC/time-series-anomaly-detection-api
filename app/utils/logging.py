from __future__ import annotations

import logging
import uuid
from contextvars import ContextVar, Token
from typing import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response

request_id_var: ContextVar[str] = ContextVar("request_id", default="none")


class RequestIdFilter(logging.Filter):
    """Inject current request_id from ContextVar into each log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get("none")
        return True


def setup_logging(log_level: str = "INFO") -> None:
    """Configure root logging format and ensure RequestIdFilter is attached."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

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


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach request_id context to request lifecycle and response header."""
    # TODO(stage2/stage3): move middleware to api layer (app/main.py or app/api/middleware.py).
    token = set_request_id()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = get_request_id()
        return response
    finally:
        reset_request_id(token)
