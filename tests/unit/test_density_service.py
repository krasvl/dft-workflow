"""Service-layer tests for density-service prediction dispatchers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from common.xyz import WATER
from services.density_service.app.model import MockDensityModel
from services.density_service.app.model_cache import CachedModel, reset_model_cache
from services.density_service.app.service import (
    ActiveModelNotFoundError,
    get_active_model,
    predict_dft,
    predict_mace,
    predict_mace_with_scf,
)


def _fake_manifest():
    from datetime import datetime, timezone

    from common.schemas import ModelManifest

    return ModelManifest(
        model_name="mace",
        version="v1",
        model_path="models/mace/v1/model.pt",
        config_path="models/mace/v1/config.json",
        metrics_path="models/mace/v1/metrics.json",
        manifest_path="models/mace/v1/manifest.json",
        created_at=datetime(2026, 5, 17, tzinfo=timezone.utc),
        status="active",
    )


def _fake_cached() -> CachedModel:
    return CachedModel(
        manifest=_fake_manifest(),
        model=MockDensityModel(),
        selection="best_loss",
        metric=0.123,
        loaded_at=0.0,
        versions_considered=2,
    )


def _storage() -> MagicMock:
    mock = MagicMock()
    mock.settings.default_model_name = "mace"
    mock.settings.inference_artifact_key.side_effect = lambda r: f"inference/artifacts/{r}.npz"
    mock.settings.dft_default_method = "rks"
    mock.settings.dft_default_basis = "sto-3g"
    return mock


def test_mock_density_model_symmetric() -> None:
    matrix = MockDensityModel().predict(WATER)
    assert matrix.shape[0] == matrix.shape[1]
    assert np.allclose(matrix, matrix.T)


def test_predict_mace_returns_uniform_shape() -> None:
    reset_model_cache()
    storage = _storage()

    with patch(
        "services.density_service.app.model_cache.serve_model",
        return_value=_fake_cached(),
    ):
        result = predict_mace(xyz_content=WATER, filename="water.xyz", storage=storage)

    assert result["model"] == "mace"
    assert result["scf_iterations"] == 0
    assert result["wall_time_sec"] >= 0
    assert result["shape"][0] == result["shape"][1] > 0
    assert result["details"]["model_version"] == "v1"
    assert result["details"]["cache_selection"] == "best_loss"
    assert result["details"]["cache_versions_considered"] == 2
    assert result["artifact_path"].startswith("inference/artifacts/")
    storage.put_bytes.assert_called_once()


def test_predict_dft_calls_pyscf_and_uses_pyscf_metadata() -> None:
    from dft.result import DftResult

    fake = DftResult(
        density_matrix=np.eye(7) * 0.5,
        wall_time_sec=0.1,
        scf_iterations=6,
        method="RKS",
        basis="sto-3g",
        energy=-75.0,
        elements=["O", "H", "H"],
        positions=np.zeros((3, 3)),
    )
    storage = _storage()

    with patch("dft.pyscf.run_pyscf_dft", return_value=fake) as run:
        result = predict_dft(xyz_content=WATER, filename="water.xyz", storage=storage)

    run.assert_called_once()
    assert result["model"] == "dft"
    assert result["scf_iterations"] == 6
    assert result["shape"] == [7, 7]
    assert result["details"]["method"] == "RKS"
    assert result["details"]["basis"] == "sto-3g"
    assert "model_version" not in result["details"]  # DFT — no ML model


def test_predict_mace_with_scf_passes_dm0_and_aggregates_time() -> None:
    from dft.result import DftResult

    reset_model_cache()
    storage = _storage()
    fake = DftResult(
        density_matrix=np.eye(6),
        wall_time_sec=0.05,
        scf_iterations=4,
        method="RKS",
        basis="sto-3g",
        energy=-1.0,
        elements=["H", "H"],
        positions=np.zeros((2, 3)),
    )

    with patch(
        "services.density_service.app.model_cache.serve_model",
        return_value=_fake_cached(),
    ), patch("dft.pyscf.run_pyscf_dft", return_value=fake) as run:
        result = predict_mace_with_scf(
            xyz_content=WATER, filename="water.xyz", storage=storage
        )

    assert run.call_count == 1
    call_kwargs = run.call_args.kwargs
    assert "dm0" in call_kwargs
    assert call_kwargs["dm0"] is not None  # MACE guess was passed through

    assert result["model"] == "mace+scf"
    assert result["scf_iterations"] == 4
    assert result["details"]["dm0_used"] is True
    assert result["details"]["model_version"] == "v1"
    assert result["wall_time_sec"] >= result["details"]["scf_wall_sec"]


def test_predict_mace_with_scf_falls_back_when_dm0_shape_mismatch() -> None:
    from dft.result import DftResult

    reset_model_cache()
    storage = _storage()
    good = DftResult(
        density_matrix=np.eye(7),
        wall_time_sec=0.05,
        scf_iterations=5,
        method="RKS",
        basis="sto-3g",
        energy=-75.0,
        elements=["O", "H", "H"],
        positions=np.zeros((3, 3)),
    )
    calls = {"count": 0}

    def fake_run(*args, **kwargs):
        calls["count"] += 1
        if kwargs.get("dm0") is not None:
            raise ValueError("dm0 shape mismatch")
        return good

    with patch(
        "services.density_service.app.model_cache.serve_model",
        return_value=_fake_cached(),
    ), patch("dft.pyscf.run_pyscf_dft", side_effect=fake_run):
        result = predict_mace_with_scf(
            xyz_content=WATER, filename="water.xyz", storage=storage
        )

    assert calls["count"] == 2  # first attempt failed, fallback succeeded
    assert result["model"] == "mace+scf"
    assert result["scf_iterations"] == 5
    assert result["details"]["dm0_used"] is False
    assert "shape mismatch" in result["details"]["dm0_reason"]


def test_active_model_not_found() -> None:
    storage = MagicMock()
    storage.settings.default_model_name = "mace"
    storage.settings.active_model_key.return_value = "models/active/mace.json"
    storage.object_exists.return_value = False

    try:
        get_active_model(storage=storage)
    except ActiveModelNotFoundError:
        return
    raise AssertionError("expected ActiveModelNotFoundError")
