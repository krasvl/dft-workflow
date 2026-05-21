"""Dispatch the worker to a training backend (``mock`` or ``mace``).

The actual model code lives in :mod:`models`. This module is thin glue: the
worker task calls :func:`run_training`, which selects the backend by
``TRAINING_ENGINE`` (or by ``train_config["engine"]`` if provided).

Heavy MACE dependencies (``graph2mat``, ``mace-torch``, ``e3nn``,
``torch_geometric``, ``pyscf``) are imported lazily inside the ``mace``
branch so the mock backend works without them.
"""

from __future__ import annotations

import logging
from typing import Any

from common.settings import get_settings
from common.storage import ObjectStorage
from workers.training_worker.app.mock_training import run_mock_training

logger = logging.getLogger("training_worker.runner")


def run_training(
    *,
    model_name: str,
    version: str,
    dft_manifests: list[dict[str, Any]],
    train_config: dict[str, Any],
    storage: ObjectStorage | None = None,
) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    """Return ``(model_bytes, config_dict, metrics_dict)``."""
    settings = get_settings()
    engine = str(train_config.get("engine", settings.training_engine)).strip().lower()
    logger.info(
        "training_engine_selected engine=%s model=%s version=%s samples=%d",
        engine, model_name, version, len(dft_manifests),
    )

    if engine == "mock":
        result = run_mock_training(
            model_name=model_name,
            version=version,
            dft_manifests=dft_manifests,
            train_config=train_config,
        )
        return result.model_bytes, result.config, result.metrics

    if engine == "mace":
        from models.mace.train import run_mace_training

        return run_mace_training(
            model_name=model_name,
            version=version,
            train_config=train_config,
            storage=storage,
        )

    raise ValueError(
        f"Unknown TRAINING_ENGINE={engine!r}; expected 'mock' or 'mace'"
    )
