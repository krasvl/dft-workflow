"""Training Celery tasks."""

from __future__ import annotations

import logging

from common.ids import new_model_version
from common.manifests import (
    build_active_model_pointer,
    build_model_manifest,
    manifest_to_dict,
)
from common.storage import get_storage
from workers.training_worker.app.celery_app import celery_app
from workers.training_worker.app.dataset import list_completed_dft_manifests
from workers.training_worker.app.training_runner import run_training

logger = logging.getLogger("training_worker")


@celery_app.task(name="workers.training_worker.app.tasks.train_model")
def train_model(
    dataset_manifest_id: str | None = None,
    train_config: dict | None = None,
) -> dict:
    config = dict(train_config or {})
    if dataset_manifest_id:
        config["dataset_manifest_id"] = dataset_manifest_id

    storage = get_storage()
    settings = storage.settings
    model_name = str(config.get("model_name", settings.default_model_name))
    version = str(config.get("version", new_model_version()))

    dft_manifests = list_completed_dft_manifests(storage)
    if not dft_manifests:
        raise ValueError("No completed DFT manifests found in MinIO — nothing to train on")

    logger.info(
        "training_started model=%s version=%s samples=%d source=%s",
        model_name,
        version,
        len(dft_manifests),
        config.get("source", "unknown"),
    )

    model_bytes, model_config, model_metrics = run_training(
        model_name=model_name,
        version=version,
        dft_manifests=dft_manifests,
        train_config=config,
        storage=storage,
    )

    model_manifest = build_model_manifest(model_name, version, settings=settings)
    base = settings.model_dir_prefix(model_name, version)

    storage.put_bytes(
        settings.model_weights_key(model_name, version),
        model_bytes,
        content_type="application/octet-stream",
    )
    storage.put_json(f"{base}/config.json", model_config)
    storage.put_json(f"{base}/metrics.json", model_metrics)
    storage.put_json(
        settings.model_manifest_key(model_name, version),
        manifest_to_dict(model_manifest),
    )

    active = build_active_model_pointer(model_name, version, settings=settings)
    storage.put_json(
        settings.active_model_key(model_name),
        manifest_to_dict(active),
    )

    logger.info(
        "training_completed model=%s version=%s active_key=%s",
        model_name,
        version,
        settings.active_model_key(model_name),
    )

    return {
        "model_name": model_name,
        "version": version,
        "status": "active",
        "samples": len(dft_manifests),
        "model_path": model_manifest.model_path,
        "active_model_key": settings.active_model_key(model_name),
        "metrics": model_metrics,
    }
