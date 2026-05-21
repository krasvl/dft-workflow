"""End-to-end MACE training entry point used by the training Celery worker.

Returns ``(model_bytes, config_dict, metrics_dict)`` — the same contract as
the mock training backend, so the worker dispatch layer is engine-agnostic.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from common.storage import ObjectStorage, get_storage
from models.mace.config import MaceHyperParams
from models.mace.model import build_mace_model, count_params
from models.shared.dataset import MinioDftDataset
from models.shared.serialization import state_to_bytes
from models.shared.training import TrainingConfig, train_density_model

logger = logging.getLogger("models.mace.train")


def run_mace_training(
    *,
    model_name: str,
    version: str,
    train_config: dict[str, Any],
    storage: ObjectStorage | None = None,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    started = time.perf_counter()
    store = storage or get_storage()

    basis_name = str(train_config.get("basis", "sto-3g"))
    epochs = int(train_config.get("epochs", 1))
    lr = float(train_config.get("learning_rate", 1e-3))
    batch_size = int(train_config.get("batch_size", 1))
    max_samples = train_config.get("max_samples")
    device = str(train_config.get("device", "cpu"))
    hparams = MaceHyperParams(**train_config.get("hparams", {}))

    dataset = MinioDftDataset(
        basis_name=basis_name,
        max_samples=int(max_samples) if max_samples is not None else None,
        storage=store,
    )
    unique_basis = dataset.unique_point_basis()

    from graph2mat import BasisTableWithEdges, MatrixDataProcessor

    table = BasisTableWithEdges(unique_basis)
    processor = MatrixDataProcessor(
        basis_table=table,
        symmetric_matrix=True,
        sub_point_matrix=False,
    )

    model = build_mace_model(unique_basis, table, hparams=hparams)
    total_params, trainable_params = count_params(model)
    logger.info(
        "mace_built params_total=%d trainable=%d elements=%d basis=%s",
        total_params, trainable_params, len(unique_basis), basis_name,
    )

    training_cfg = TrainingConfig(
        epochs=epochs,
        learning_rate=lr,
        batch_size=batch_size,
        max_samples=max_samples,
        device=device,
        extra=train_config,
    )

    checkpoint, metrics = train_density_model(
        model=model,
        processor=processor,
        dataset=dataset,
        config=training_cfg,
        extra_state={
            "model_name": model_name,
            "version": version,
            "architecture": "mace",
            "basis_name": basis_name,
            "hparams": hparams.to_dict(),
            "elements": [pb.type for pb in unique_basis],
        },
    )
    model_bytes = state_to_bytes(checkpoint)

    config = {
        "model_name": model_name,
        "version": version,
        "architecture": "mace",
        "input_format": "xyz",
        "output": "density_matrix",
        "basis": basis_name,
        "hparams": hparams.to_dict(),
        "training": {
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "max_samples": max_samples,
            "device": device,
        },
        "samples": metrics.samples,
        "params": {"total": total_params, "trainable": trainable_params},
    }
    metrics_dict = {
        "train_loss": metrics.train_loss,
        "best_loss": metrics.best_loss,
        "epochs": metrics.epochs,
        "steps": metrics.steps,
        "samples": metrics.samples,
        "wall_time_sec": round(time.perf_counter() - started, 4),
        "losses": [round(x, 6) for x in metrics.losses],
    }
    return model_bytes, config, metrics_dict
