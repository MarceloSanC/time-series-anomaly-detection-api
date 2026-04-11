from __future__ import annotations

from contextlib import asynccontextmanager
from time import perf_counter
from typing import AsyncIterator, Awaitable, Callable

from fastapi import FastAPI, Request
from starlette.responses import Response

from app.api.error_handlers import register_error_handlers
from app.api.middleware import request_id_middleware
from app.api.routes import api_router
from app.config import settings
from app.services.metrics_service import MetricsService
from app.utils.concurrency import LockManager
from app.utils.logging import setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialize and expose shared app services during process lifetime."""
    app.state.lock_manager = LockManager()
    app.state.metrics_service = MetricsService(latency_window_size=settings.latency_window_size)
    yield


def create_app() -> FastAPI:
    """Create and configure FastAPI application instance."""
    setup_logging(settings.log_level)
    app = FastAPI(title=settings.app_name, lifespan=lifespan)

    @app.middleware("http")
    async def metrics_middleware(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        start = perf_counter()
        error = False
        response: Response | None = None
        try:
            response = await call_next(request)
            error = response.status_code >= 400
            return response
        except Exception:
            error = True
            raise
        finally:
            duration_ms = (perf_counter() - start) * 1000
            endpoint = request.url.path
            app.state.metrics_service.record(endpoint=endpoint, duration_ms=duration_ms, error=error)
            if response is not None:
                response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"

    # Starlette executes middlewares in reverse registration order.
    # Registered last = executes first.
    app.middleware("http")(request_id_middleware)

    app.include_router(api_router)
    register_error_handlers(app)
    return app


app = create_app()
