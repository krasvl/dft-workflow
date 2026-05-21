"""Unit tests for ``services.density_service.app.model.load_density_model``.

We verify checkpoint dispatch (mock | legacy ``weights`` | real MACE) without
actually building a MACE model in tests — the real ``MaceInferenceModel``
constructor needs the full graph2mat/mace-torch stack and is only exercised
inside the Docker images and the conda dev env. Here we patch it so the
dispatch logic itself is the unit under test.
"""

from __future__ import annotations

import io
from unittest.mock import patch

import numpy as np
import pytest
import torch

from common.xyz import WATER
from services.density_service.app.model import (
    LegacyWeightsDensityModel,
    MaceDensityModel,
    MockDensityModel,
    load_density_model,
)


def _ckpt_bytes(payload: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def test_mock_flag_short_circuits_to_mock_model() -> None:
    model = load_density_model(_ckpt_bytes({"weights": torch.zeros(2, 2)}), mock=True)
    assert isinstance(model, MockDensityModel)


def test_legacy_weights_only_checkpoint_uses_legacy_model() -> None:
    model = load_density_model(_ckpt_bytes({"mock": True, "weights": torch.eye(4)}))
    assert isinstance(model, LegacyWeightsDensityModel)
    dm = model.predict(WATER)
    assert dm.shape[0] == dm.shape[1]
    assert np.allclose(dm, dm.T)


def test_real_mace_checkpoint_routes_to_mace_density_model() -> None:
    real_ckpt = {
        "architecture": "mace",
        "model_state_dict": {"dummy": torch.zeros(1)},
        "elements": ["H", "O"],
        "basis_name": "sto-3g",
        "hparams": {},
    }
    fake_impl = type("FakeImpl", (), {"predict": lambda self, x: np.zeros((2, 2))})()
    with patch("models.mace.inference.MaceInferenceModel", return_value=fake_impl):
        model = load_density_model(_ckpt_bytes(real_ckpt))
    assert isinstance(model, MaceDensityModel)
    assert model.predict(WATER).shape == (2, 2)


def test_load_rejects_non_dict_checkpoint() -> None:
    bad = io.BytesIO()
    torch.save([1, 2, 3], bad)
    with pytest.raises(ValueError, match="dict checkpoint"):
        load_density_model(bad.getvalue())
