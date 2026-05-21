from celery import Celery

from common.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "dft_worker",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.dft_worker.app.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_default_queue="dft",
    task_routes={"workers.dft_worker.app.tasks.*": {"queue": "dft"}},
)
