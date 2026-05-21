from common.manifests import (
    build_dft_manifest,
    build_model_manifest,
    build_molecule_manifest,
    manifest_to_dict,
)
from common.settings import Settings


def test_molecule_manifest_fields() -> None:
    manifest = build_molecule_manifest(
        "mol_test",
        "molecules/raw/mol_test.xyz",
        status="uploaded",
    )
    data = manifest_to_dict(manifest)
    assert data["molecule_id"] == "mol_test"
    assert data["format"] == "xyz"
    assert data["status"] == "uploaded"
    assert "created_at" in data


def test_dft_manifest_paths() -> None:
    manifest = build_dft_manifest(
        "dft_test",
        "mol_test",
        "dft/artifacts/mol_test/dft_test.npz",
        wall_time_sec=1.5,
        scf_iterations=3,
    )
    data = manifest_to_dict(manifest)
    assert data["metrics"]["scf_iterations"] == 3
    assert data["method"] == "mock-rks"


def test_model_manifest_paths() -> None:
    settings = Settings()
    manifest = build_model_manifest("density_predictor", "v1", settings=settings)
    data = manifest_to_dict(manifest)
    assert data["model_path"] == "models/density_predictor/v1/model.pt"
    assert data["status"] == "active"
