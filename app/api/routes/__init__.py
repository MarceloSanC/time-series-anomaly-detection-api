from fastapi import APIRouter

from app.api.routes.fit import router as fit_router
from app.api.routes.healthcheck import router as healthcheck_router
from app.api.routes.plot import router as plot_router
from app.api.routes.predict import router as predict_router

api_router = APIRouter()
api_router.include_router(fit_router)
api_router.include_router(predict_router)
api_router.include_router(healthcheck_router)
api_router.include_router(plot_router)
