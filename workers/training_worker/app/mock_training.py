"""Mock training backend.

Produces a tiny deterministic checkpoint without invoking any ML stack.
Used by unit tests and quick smoke runs (``TRAINING_ENGINE=mock``).
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class MockTrainingResult:
    model_bytes: bytes
    config: dict[str, Any]
    metrics: dict[str, Any]
    wall_time_sec: float


def run_mock_training(
    *,
    model_name: str,
    version: str,
    dft_manifests: list[dict[str, Any]],
    train_config: dict[str, Any],
) -> MockTrainingResult:
    """Return a checkpoint whose ``weights`` are a random 8×8 tensor.

    The density-service falls back to :class:`LegacyWeightsDensityModel` for
    this kind of checkpoint, so the full serving pipeline can be exercised
    without depending on the heavy ML stack.
    """
    started = time.perf_counter()
    sample_count = len(dft_manifests)

    state = {
        "model_name": model_name,
        "version": version,
        "weights": torch.randn(8, 8),
        "mock": True,
    }
    buffer = io.BytesIO()
    torch.save(state, buffer)
    model_bytes = buffer.getvalue()

    config = {
        "model_name": model_name,
        "version": version,
        "architecture": "mock_density_predictor",
        "input_format": "xyz",
        "output": "density_matrix",
        "mock": True,
        "training_samples": sample_count,
        "train_config": train_config,
    }
    metrics = {
        "train_loss": 0.042,
        "val_loss": 0.051,
        "epochs": 1,
        "samples": sample_count,
        "wall_time_sec": round(time.perf_counter() - started, 4),
    }
    return MockTrainingResult(
        model_bytes=model_bytes,
        config=config,
        metrics=metrics,
        wall_time_sec=metrics["wall_time_sec"],
    )
