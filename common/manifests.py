"""Factory helpers for JSON manifests stored in MinIO."""

from datetime import datetime, timezone
from typing import Any

from common.schemas import (
    ActiveModelPointer,
    DftManifest,
    DftMetrics,
    ModelManifest,
    MoleculeManifest,
)
from common.settings import Settings, get_settings


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def build_molecule_manifest(
    molecule_id: str,
    source_file: str,
    *,
    status: str = "uploaded",
    fmt: str = "xyz",
    created_at: datetime | None = None,
) -> MoleculeManifest:
    return MoleculeManifest(
        molecule_id=molecule_id,
        source_file=source_file,
        format=fmt,
        status=status,
        created_at=created_at or utc_now(),
    )


def build_dft_manifest(
    calculation_id: str,
    molecule_id: str,
    artifact_path: str,
    *,
    status: str = "completed",
    method: str = "mock-rks",
    basis: str = "mock-basis",
    wall_time_sec: float = 0.0,
    scf_iterations: int = 0,
    energy: float | None = None,
    created_at: datetime | None = None,
) -> DftManifest:
    return DftManifest(
        calculation_id=calculation_id,
        molecule_id=molecule_id,
        status=status,
        method=method,
        basis=basis,
        artifact_path=artifact_path,
        metrics=DftMetrics(
            wall_time_sec=wall_time_sec,
            scf_iterations=scf_iterations,
            energy=energy,
        ),
        created_at=created_at or utc_now(),
    )


def build_model_manifest(
    model_name: str,
    version: str,
    *,
    status: str = "active",
    created_at: datetime | None = None,
    settings: Settings | None = None,
) -> ModelManifest:
    cfg = settings or get_settings()
    base = cfg.model_dir_prefix(model_name, version)
    return ModelManifest(
        model_name=model_name,
        version=version,
        model_path=f"{base}/model.pt",
        config_path=f"{base}/config.json",
        metrics_path=f"{base}/metrics.json",
        manifest_path=f"{base}/manifest.json",
        created_at=created_at or utc_now(),
        status=status,
    )


def build_active_model_pointer(
    model_name: str,
    version: str,
    *,
    settings: Settings | None = None,
) -> ActiveModelPointer:
    cfg = settings or get_settings()
    return ActiveModelPointer(
        model_name=model_name,
        version=version,
        manifest_path=cfg.model_manifest_key(model_name, version),
    )


def manifest_to_dict(manifest: Any) -> dict[str, Any]:
    """Serialize manifest model to JSON-compatible dict (ISO datetimes)."""
    return manifest.model_dump(mode="json")
