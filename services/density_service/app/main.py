from contextlib import asynccontextmanager

from fastapi import FastAPI

from common.logging import setup_logging
from common.storage import get_storage
from services.density_service.app.api import router

logger = setup_logging("density_service")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    storage = get_storage()
    storage.ensure_bucket()
    logger.info("density_service_started bucket=%s", storage.settings.minio_bucket)
    yield


app = FastAPI(title="Density Service", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "density-service"}
