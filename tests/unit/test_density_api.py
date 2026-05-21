"""FastAPI route tests for density-service (TestClient, no MinIO)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
from fastapi.testclient import TestClient

from common.schemas import ModelManifest
from common.xyz import WATER
from dft.result import DftResult
from services.density_service.app.main import app
from services.density_service.app.model import MockDensityModel
from services.density_service.app.model_cache import CachedModel, reset_model_cache
from services.density_service.app.service import (
    ActiveModelNotFoundError,
    MoleculeNotFoundError,
)

client = TestClient(app)


def _fake_manifest() -> ModelManifest:
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


def _fake_cached(selection: str = "best_loss", metric: float = 0.5) -> CachedModel:
    return CachedModel(
        manifest=_fake_manifest(),
        model=MockDensityModel(),
        selection=selection,
        metric=metric,
        loaded_at=0.0,
        versions_considered=3,
    )


def _fake_dft_result(size: int = 7, iters: int = 6) -> DftResult:
    return DftResult(
        density_matrix=np.eye(size) * 0.5,
        wall_time_sec=0.05,
        scf_iterations=iters,
        method="RKS",
        basis="sto-3g",
        energy=-75.0,
        elements=["O", "H", "H"],
        positions=np.zeros((3, 3)),
    )


def _storage() -> MagicMock:
    mock = MagicMock()
    mock.settings.inference_artifact_key.return_value = "inference/artifacts/req.npz"
    mock.settings.dft_default_method = "rks"
    mock.settings.dft_default_basis = "sto-3g"
    return mock


def test_health() -> None:
    r = client.get("/api/density/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "density-service"}


def test_list_models_marks_dft_always_available() -> None:
    reset_model_cache()
    with patch(
        "services.density_service.app.api.serve_model",
        return_value=_fake_cached(),
    ):
        r = client.get("/api/density/models")
    assert r.status_code == 200
    body = r.json()
    names = {m["name"]: m for m in body["models"]}
    assert names["dft"]["available"] is True
    assert names["dft"]["kind"] == "scf"
    assert names["mace"]["available"] is True
    assert names["mace"]["kind"] == "ml"


def test_list_models_reports_mace_unavailable_when_no_active() -> None:
    reset_model_cache()
    with patch(
        "services.density_service.app.api.serve_model",
        side_effect=ActiveModelNotFoundError("no active model"),
    ):
        r = client.get("/api/density/models")
    body = r.json()
    names = {m["name"]: m for m in body["models"]}
    assert names["mace"]["available"] is False
    assert names["dft"]["available"] is True


def test_serving_endpoint_reports_cache_state() -> None:
    reset_model_cache()
    with patch(
        "services.density_service.app.api.serve_model",
        return_value=_fake_cached(selection="best_loss", metric=0.123),
    ):
        r = client.get("/api/density/models/serving")
    assert r.status_code == 200
    body = r.json()
    assert body["selection"] == "best_loss"
    assert body["metric"] == 0.123
    assert body["versions_considered"] == 3
    assert body["manifest"]["version"] == "v1"


def test_serving_endpoint_404_when_no_model() -> None:
    reset_model_cache()
    with patch(
        "services.density_service.app.api.serve_model",
        side_effect=ActiveModelNotFoundError("no active model"),
    ):
        r = client.get("/api/density/models/serving")
    assert r.status_code == 404


def test_invalidate_cache_endpoint() -> None:
    r = client.post("/api/density/models/cache/invalidate")
    assert r.status_code == 200
    assert r.json() == {"status": "invalidated"}


def test_active_endpoint_404_when_pointer_missing() -> None:
    with patch(
        "services.density_service.app.api.get_active_model",
        side_effect=ActiveModelNotFoundError("no pointer"),
    ):
        r = client.get("/api/density/models/active")
    assert r.status_code == 404


def test_predict_dft_route_returns_real_iter_count() -> None:
    with patch(
        "services.density_service.app.service.get_storage", return_value=_storage()
    ), patch(
        "dft.pyscf.run_pyscf_dft", return_value=_fake_dft_result(size=7, iters=6)
    ):
        r = client.post(
            "/api/density/predict/dft",
            files={"file": ("water.xyz", WATER, "chemical/x-xyz")},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "dft"
    assert body["scf_iterations"] == 6
    assert body["shape"] == [7, 7]
    assert body["details"]["method"] == "RKS"
    assert "model_version" not in body["details"]


def test_predict_mace_route_returns_zero_iterations() -> None:
    reset_model_cache()
    with patch(
        "services.density_service.app.model_cache.serve_model",
        return_value=_fake_cached(),
    ), patch(
        "services.density_service.app.service.get_storage", return_value=_storage()
    ):
        r = client.post(
            "/api/density/predict/mace",
            files={"file": ("water.xyz", WATER, "chemical/x-xyz")},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "mace"
    assert body["scf_iterations"] == 0
    assert body["details"]["model_version"] == "v1"
    assert body["details"]["cache_selection"] == "best_loss"


def test_predict_mace_with_scf_route_aggregates_iterations() -> None:
    reset_model_cache()
    with patch(
        "services.density_service.app.model_cache.serve_model",
        return_value=_fake_cached(),
    ), patch(
        "services.density_service.app.service.get_storage", return_value=_storage()
    ), patch(
        "dft.pyscf.run_pyscf_dft", return_value=_fake_dft_result(size=7, iters=4)
    ):
        r = client.post(
            "/api/density/predict/mace/with-scf",
            files={"file": ("water.xyz", WATER, "chemical/x-xyz")},
        )

    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "mace+scf"
    assert body["scf_iterations"] == 4
    assert body["details"]["model_version"] == "v1"
    assert "dm0_used" in body["details"]


def test_predict_route_rejects_invalid_xyz() -> None:
    r = client.post(
        "/api/density/predict/dft",
        files={"file": ("bad.xyz", b"not xyz", "chemical/x-xyz")},
    )
    assert r.status_code == 400


def test_predict_dft_by_id_404_when_molecule_missing() -> None:
    with patch(
        "services.density_service.app.api.predict_dft",
        side_effect=MoleculeNotFoundError("mol_missing"),
    ):
        r = client.post("/api/density/predict/dft/by-id/mol_missing")
    assert r.status_code == 404
