"""Density-service inference orchestration.

Three prediction modes share a single response shape (see ``PredictResponse``)
so the API layer can hand the result straight to pydantic:

* :func:`predict_dft`           — PySCF SCF only (no ML model).
* :func:`predict_mace`          — single MACE forward, no SCF.
* :func:`predict_mace_with_scf` — MACE forward used as the ``dm0`` initial
                                  guess for a PySCF SCF refinement.
"""

from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np

from common.ids import new_request_id
from common.schemas import ActiveModelPointer, ModelManifest
from common.settings import get_settings
from common.storage import ObjectStorage, get_storage
from common.xyz import validate_xyz

logger = logging.getLogger("density_service")


class ActiveModelNotFoundError(Exception):
    pass


class MoleculeNotFoundError(Exception):
    pass


# --- helpers ---------------------------------------------------------------


def _preview(matrix: np.ndarray, rows: int = 3, cols: int = 3) -> list[list[float]]:
    r = min(rows, matrix.shape[0])
    c = min(cols, matrix.shape[1])
    return matrix[:r, :c].tolist()


def _save_artifact(
    matrix: np.ndarray,
    *,
    model: str,
    request_id: str,
    molecule_id: str | None,
    model_version: str | None,
    storage: ObjectStorage,
) -> str:
    key = storage.settings.inference_artifact_key(request_id)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        density_matrix=matrix,
        molecule_id=np.array(molecule_id or ""),
        request_id=np.array(request_id),
        model=np.array(model),
        model_version=np.array(model_version or ""),
    )
    storage.put_bytes(key, buf.getvalue(), content_type="application/octet-stream")
    return key


def _resolve_xyz(
    *,
    xyz_content: bytes | None,
    molecule_id: str | None,
    filename: str | None,
    storage: ObjectStorage,
) -> tuple[bytes, str | None]:
    """Return ``(xyz_bytes, molecule_id_if_known)`` for a prediction request.

    Either the raw XYZ bytes are provided directly or they are fetched from
    MinIO under the given molecule id.
    """
    if xyz_content is not None:
        validate_xyz(xyz_content, filename)
        return xyz_content, molecule_id

    if molecule_id is None:
        raise ValueError("Either xyz_content or molecule_id is required")

    raw_key = storage.settings.molecule_raw_key(molecule_id)
    if not storage.object_exists(raw_key):
        raise MoleculeNotFoundError(molecule_id)
    return storage.get_bytes(raw_key), molecule_id


def get_active_model(
    model_name: str | None = None,
    storage: ObjectStorage | None = None,
) -> tuple[ActiveModelPointer, ModelManifest]:
    """Return ``(pointer, manifest)`` for the latest active model in MinIO."""
    store = storage or get_storage()
    name = model_name or store.settings.default_model_name
    active_key = store.settings.active_model_key(name)
    if not store.object_exists(active_key):
        raise ActiveModelNotFoundError(f"No active model for {name}")
    pointer = ActiveModelPointer.model_validate(store.get_json(active_key))
    manifest = ModelManifest.model_validate(store.get_json(pointer.manifest_path))
    return pointer, manifest


# --- /predict/dft ----------------------------------------------------------


def predict_dft(
    *,
    xyz_content: bytes | None = None,
    molecule_id: str | None = None,
    filename: str | None = None,
    method: str | None = None,
    basis: str | None = None,
    storage: ObjectStorage | None = None,
) -> dict[str, Any]:
    """Run PySCF SCF and return the density matrix in the unified response shape."""
    from dft.pyscf import run_pyscf_dft

    store = storage or get_storage()
    settings = get_settings()

    xyz, mol_id = _resolve_xyz(
        xyz_content=xyz_content,
        molecule_id=molecule_id,
        filename=filename,
        storage=store,
    )

    started = time.perf_counter()
    result = run_pyscf_dft(
        xyz,
        method=method or settings.dft_default_method,
        basis=basis or settings.dft_default_basis,
        max_cycle=settings.dft_max_scf_cycles,
        conv_tol=settings.dft_scf_conv_tol,
    )
    total_wall = round(time.perf_counter() - started, 4)

    request_id = new_request_id()
    artifact_path = _save_artifact(
        result.density_matrix,
        model="dft",
        request_id=request_id,
        molecule_id=mol_id,
        model_version=None,
        storage=store,
    )

    logger.info(
        "predict_dft request_id=%s shape=%s iters=%d wall=%.3fs",
        request_id, result.density_matrix.shape, result.scf_iterations, total_wall,
    )

    return {
        "model": "dft",
        "shape": list(result.density_matrix.shape),
        "preview": _preview(result.density_matrix),
        "scf_iterations": result.scf_iterations,
        "wall_time_sec": total_wall,
        "request_id": request_id,
        "artifact_path": artifact_path,
        "details": {
            "method": result.method,
            "basis": result.basis,
            "molecule_id": mol_id,
        },
    }


# --- /predict/mace ---------------------------------------------------------


def predict_mace(
    *,
    xyz_content: bytes | None = None,
    molecule_id: str | None = None,
    filename: str | None = None,
    storage: ObjectStorage | None = None,
) -> dict[str, Any]:
    """Single MACE forward pass. No SCF — ``scf_iterations`` is always 0."""
    from services.density_service.app.model_cache import serve_model

    store = storage or get_storage()
    xyz, mol_id = _resolve_xyz(
        xyz_content=xyz_content,
        molecule_id=molecule_id,
        filename=filename,
        storage=store,
    )

    cached = serve_model(storage=store)
    started = time.perf_counter()
    matrix = cached.model.predict(xyz)
    wall = round(time.perf_counter() - started, 4)

    request_id = new_request_id()
    artifact_path = _save_artifact(
        matrix,
        model="mace",
        request_id=request_id,
        molecule_id=mol_id,
        model_version=cached.manifest.version,
        storage=store,
    )

    age_sec = round(time.monotonic() - cached.loaded_at, 3)
    logger.info(
        "predict_mace request_id=%s version=%s shape=%s wall=%.3fs",
        request_id, cached.manifest.version, matrix.shape, wall,
    )

    return {
        "model": "mace",
        "shape": list(matrix.shape),
        "preview": _preview(matrix),
        "scf_iterations": 0,
        "wall_time_sec": wall,
        "request_id": request_id,
        "artifact_path": artifact_path,
        "details": {
            "model_version": cached.manifest.version,
            "cache_selection": cached.selection,
            "cache_metric": cached.metric,
            "cache_age_sec": age_sec,
            "cache_versions_considered": cached.versions_considered,
            "molecule_id": mol_id,
        },
    }


# --- /predict/mace/with-scf ------------------------------------------------


def predict_mace_with_scf(
    *,
    xyz_content: bytes | None = None,
    molecule_id: str | None = None,
    filename: str | None = None,
    method: str | None = None,
    basis: str | None = None,
    storage: ObjectStorage | None = None,
) -> dict[str, Any]:
    """MACE forward used as ``dm0`` for a subsequent PySCF SCF refinement.

    The MACE checkpoint records the AO basis it was trained on
    (``basis_name``). If that matches the requested basis, the predicted DM
    is shape-compatible and used as the SCF initial guess. Otherwise (or if
    a legacy/mock-style checkpoint is loaded), PySCF falls back to its
    default initial guess and ``details.dm0_used`` is set to ``false``.
    """
    from dft.pyscf import run_pyscf_dft
    from services.density_service.app.model_cache import serve_model

    store = storage or get_storage()
    settings = get_settings()

    xyz, mol_id = _resolve_xyz(
        xyz_content=xyz_content,
        molecule_id=molecule_id,
        filename=filename,
        storage=store,
    )

    cached = serve_model(storage=store)
    started_mace = time.perf_counter()
    dm_guess = cached.model.predict(xyz)
    mace_wall = round(time.perf_counter() - started_mace, 4)

    started_scf = time.perf_counter()
    try:
        result = run_pyscf_dft(
            xyz,
            method=method or settings.dft_default_method,
            basis=basis or settings.dft_default_basis,
            max_cycle=settings.dft_max_scf_cycles,
            conv_tol=settings.dft_scf_conv_tol,
            dm0=dm_guess,
        )
        dm0_used = True
        dm0_reason: str | None = None
    except (ValueError, TypeError) as exc:
        # Most likely a shape mismatch between the MACE-output DM and the
        # requested AO basis. Fall back to PySCF's default initial guess so
        # the request still succeeds.
        logger.warning(
            "predict_mace_with_scf dm0 rejected (%s); falling back to default guess",
            exc,
        )
        result = run_pyscf_dft(
            xyz,
            method=method or settings.dft_default_method,
            basis=basis or settings.dft_default_basis,
            max_cycle=settings.dft_max_scf_cycles,
            conv_tol=settings.dft_scf_conv_tol,
        )
        dm0_used = False
        dm0_reason = f"{type(exc).__name__}: {exc}"
    scf_wall = round(time.perf_counter() - started_scf, 4)

    request_id = new_request_id()
    artifact_path = _save_artifact(
        result.density_matrix,
        model="mace+scf",
        request_id=request_id,
        molecule_id=mol_id,
        model_version=cached.manifest.version,
        storage=store,
    )

    age_sec = round(time.monotonic() - cached.loaded_at, 3)
    logger.info(
        "predict_mace_with_scf request_id=%s version=%s shape=%s iters=%d "
        "mace=%.3fs scf=%.3fs dm0_used=%s",
        request_id, cached.manifest.version, result.density_matrix.shape,
        result.scf_iterations, mace_wall, scf_wall, dm0_used,
    )

    return {
        "model": "mace+scf",
        "shape": list(result.density_matrix.shape),
        "preview": _preview(result.density_matrix),
        "scf_iterations": result.scf_iterations,
        "wall_time_sec": round(mace_wall + scf_wall, 4),
        "request_id": request_id,
        "artifact_path": artifact_path,
        "details": {
            "method": result.method,
            "basis": result.basis,
            "mace_wall_sec": mace_wall,
            "scf_wall_sec": scf_wall,
            "dm0_used": dm0_used,
            "dm0_reason": dm0_reason,
            "model_version": cached.manifest.version,
            "cache_selection": cached.selection,
            "cache_age_sec": age_sec,
            "molecule_id": mol_id,
        },
    }
