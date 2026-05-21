"""Celery task producer for API services (no worker import required)."""

from celery import Celery

from common.settings import get_settings

DFT_TASK_NAME = "workers.dft_worker.app.tasks.run_dft_calculation"
TRAINING_TASK_NAME = "workers.training_worker.app.tasks.train_model"


def _producer() -> Celery:
    settings = get_settings()
    return Celery("dft_workflow_producer", broker=settings.redis_url, backend=settings.redis_url)


def enqueue_dft_calculation(
    molecule_id: str,
    calculation_config: dict | None = None,
) -> str:
    """Enqueue DFT calculation; returns Celery task id."""
    app = _producer()
    result = app.send_task(
        DFT_TASK_NAME,
        args=[molecule_id],
        kwargs={"calculation_config": calculation_config or {}},
        queue="dft",
    )
    return result.id


def enqueue_training(
    dataset_manifest_id: str | None = None,
    train_config: dict | None = None,
    *,
    molecule_id: str | None = None,
    calculation_id: str | None = None,
) -> str:
    """Enqueue model training; returns Celery task id."""
    app = _producer()
    kwargs: dict = {"train_config": train_config or {}}
    if dataset_manifest_id is not None:
        kwargs["dataset_manifest_id"] = dataset_manifest_id
    if molecule_id is not None:
        kwargs.setdefault("train_config", {})["molecule_id"] = molecule_id
    if calculation_id is not None:
        kwargs.setdefault("train_config", {})["calculation_id"] = calculation_id
    result = app.send_task(
        TRAINING_TASK_NAME,
        kwargs=kwargs,
        queue="training",
    )
    return result.id
