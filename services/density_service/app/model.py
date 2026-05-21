"""Density-model interface used by :class:`ModelCache`.

Implementations:

* :class:`MockDensityModel`         — deterministic stand-in for tests.
* :class:`MaceDensityModel`         — wraps :class:`MaceInferenceModel`
                                      (real MACE + graph2mat forward).
* :class:`LegacyWeightsDensityModel` — fallback for older checkpoints that
                                      only stored a ``weights`` tensor; keeps
                                      the serving pipeline alive when such an
                                      artifact is encountered in MinIO.

:func:`load_density_model` picks the right implementation by inspecting the
checkpoint payload.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from common.storage import ObjectStorage
from common.xyz import parse_atom_count

logger = logging.getLogger("density_service.model")


class DensityModel(ABC):
    @abstractmethod
    def predict(self, xyz_content: bytes) -> np.ndarray:
        """Return predicted density matrix (NxN symmetric)."""


class MockDensityModel(DensityModel):
    """Deterministic stand-in predictor sized from the atom count. Used in tests."""

    def predict(self, xyz_content: bytes) -> np.ndarray:
        n_atoms = parse_atom_count(xyz_content)
        size = max(n_atoms * 2, 4)
        rng = np.random.default_rng(abs(hash(xyz_content[:64])) % (2**32))
        matrix = rng.random((size, size), dtype=np.float64)
        return (matrix + matrix.T) / 2.0


class MaceDensityModel(DensityModel):
    """MACE forward pass through :class:`models.mace.inference.MaceInferenceModel`."""

    def __init__(
        self,
        checkpoint: dict[str, Any],
        *,
        storage: ObjectStorage | None = None,
    ) -> None:
        from models.mace.inference import MaceInferenceModel

        self._impl = MaceInferenceModel(checkpoint, storage=storage)

    def predict(self, xyz_content: bytes) -> np.ndarray:
        return self._impl.predict(xyz_content)


class LegacyWeightsDensityModel(DensityModel):
    """Fallback for checkpoints that only carry a raw ``weights`` tensor.

    The mock training backend saves ``{"weights": torch.randn(8,8), "mock":
    True}`` without a state dict, hyper-parameters or element list — so a
    real MACE forward is impossible. To keep the cache and serving pipeline
    exercisable end-to-end, this fallback synthesises a deterministic NxN
    block from the saved tensor.
    """

    def __init__(self, checkpoint: dict[str, Any]) -> None:
        self._checkpoint = checkpoint

    def predict(self, xyz_content: bytes) -> np.ndarray:
        n_atoms = parse_atom_count(xyz_content)
        size = max(n_atoms * 2, 4)
        weights = self._checkpoint.get("weights")
        if isinstance(weights, torch.Tensor):
            w = weights.detach().cpu().numpy()
            side = min(size, w.shape[0], w.shape[1])
            block = w[:side, :side].astype(np.float64)
            if block.shape[0] < size:
                out = np.zeros((size, size), dtype=np.float64)
                out[: block.shape[0], : block.shape[1]] = block
                block = (out + out.T) / 2.0
            else:
                block = (block + block.T) / 2.0
            return block
        return MockDensityModel().predict(xyz_content)


def _is_real_mace_checkpoint(checkpoint: dict[str, Any]) -> bool:
    return (
        isinstance(checkpoint.get("model_state_dict"), dict)
        and isinstance(checkpoint.get("elements"), (list, tuple))
        and bool(checkpoint.get("elements"))
    )


def load_density_model(
    checkpoint_bytes: bytes,
    *,
    mock: bool = False,
    storage: ObjectStorage | None = None,
) -> DensityModel:
    """Pick the right ``DensityModel`` implementation for the given checkpoint.

    * ``mock=True`` → :class:`MockDensityModel` (no torch load).
    * Real MACE checkpoint (state_dict + elements + hparams) →
      :class:`MaceDensityModel` (real forward pass).
    * Legacy ``weights``-only checkpoint → :class:`LegacyWeightsDensityModel`.
    """
    if mock:
        return MockDensityModel()

    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=False
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("model.pt must contain a dict checkpoint")

    architecture = str(checkpoint.get("architecture", "")).lower()
    if architecture == "mace" and _is_real_mace_checkpoint(checkpoint):
        logger.info(
            "load_mace_inference elements=%s basis=%s",
            checkpoint.get("elements"), checkpoint.get("basis_name"),
        )
        return MaceDensityModel(checkpoint, storage=storage)

    if _is_real_mace_checkpoint(checkpoint):
        logger.info("load_mace_inference architecture=inferred")
        return MaceDensityModel(checkpoint, storage=storage)

    logger.warning(
        "checkpoint has no MACE state_dict; "
        "falling back to LegacyWeightsDensityModel"
    )
    return LegacyWeightsDensityModel(checkpoint)
