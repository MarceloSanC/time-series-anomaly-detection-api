from __future__ import annotations

from typing import Awaitable, Callable

from starlette.requests import Request
from starlette.responses import Response

from app.utils.logging import get_request_id, reset_request_id, set_request_id


async def request_id_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """Attach request_id context to request lifecycle and response header."""
    token = set_request_id()
    try:
        response = await call_next(request)
        response.headers["X-Request-ID"] = get_request_id()
        return response
    finally:
        reset_request_id(token)

