"""DFT Celery tasks."""

from __future__ import annotations

import logging

from common.ids import new_calculation_id
from common.training_trigger import record_dft_completed_and_maybe_enqueue_training
from common.jobs import find_dft_job_for_molecule, update_job
from common.manifests import build_dft_manifest, manifest_to_dict
from common.storage import get_storage
from dft import run_dft
from workers.dft_worker.app.artifacts import dft_result_to_npz_bytes
from workers.dft_worker.app.celery_app import celery_app

logger = logging.getLogger("dft_worker")


def _update_molecule_status(
    molecule_id: str,
    status: str,
    *,
    calculation_id: str | None = None,
) -> None:
    storage = get_storage()
    key = storage.settings.molecule_manifest_key(molecule_id)
    data = storage.get_json(key)
    data["status"] = status
    if calculation_id:
        data["calculation_id"] = calculation_id
    storage.put_json(key, data)


@celery_app.task(name="workers.dft_worker.app.tasks.run_dft_calculation")
def run_dft_calculation(
    molecule_id: str,
    calculation_config: dict | None = None,
) -> dict:
    config = calculation_config or {}
    storage = get_storage()
    settings = storage.settings

    raw_key = settings.molecule_raw_key(molecule_id)
    if not storage.object_exists(raw_key):
        raise FileNotFoundError(f"Molecule XYZ not found: {molecule_id}")

    job_id = find_dft_job_for_molecule(molecule_id, storage)
    if job_id:
        update_job(job_id, status="running", storage=storage)

    _update_molecule_status(molecule_id, "dft_running")

    xyz_content = storage.get_bytes(raw_key)
    method = config.get("method")
    basis = config.get("basis")
    result = run_dft(
        xyz_content,
        method=str(method) if method is not None else None,
        basis=str(basis) if basis is not None else None,
    )
    calculation_id = new_calculation_id()
    artifact_key = settings.dft_artifact_key(molecule_id, calculation_id)

    npz_bytes = dft_result_to_npz_bytes(
        result,
        molecule_id=molecule_id,
        calculation_id=calculation_id,
    )
    storage.put_bytes(artifact_key, npz_bytes, content_type="application/octet-stream")

    manifest = build_dft_manifest(
        calculation_id,
        molecule_id,
        artifact_key,
        status="completed",
        method=result.method,
        basis=result.basis,
        wall_time_sec=result.wall_time_sec,
        scf_iterations=result.scf_iterations,
        energy=result.energy,
    )
    storage.put_json(
        settings.dft_manifest_key(calculation_id),
        manifest_to_dict(manifest),
    )

    _update_molecule_status(molecule_id, "dft_completed", calculation_id=calculation_id)

    if job_id:
        update_job(
            job_id,
            status="completed",
            details={
                "calculation_id": calculation_id,
                "artifact_path": artifact_key,
            },
            storage=storage,
        )

    batch = record_dft_completed_and_maybe_enqueue_training(
        molecule_id,
        calculation_id,
    )

    logger.info(
        "dft_completed molecule_id=%s calculation_id=%s training_triggered=%s",
        molecule_id,
        calculation_id,
        batch.triggered,
    )

    return {
        "molecule_id": molecule_id,
        "calculation_id": calculation_id,
        "artifact_path": artifact_key,
        "status": "completed",
        "training_triggered": batch.triggered,
        "training_task_id": batch.training_task_id,
        "training_batch_threshold": batch.threshold,
    }
