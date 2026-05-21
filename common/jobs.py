"""Job manifest persistence in MinIO."""

from typing import Any

from common.ids import new_job_id
from common.manifests import manifest_to_dict, utc_now
from common.schemas import JobStatus
from common.settings import Settings, get_settings
from common.storage import ObjectStorage, get_storage


def build_dft_job(
    molecule_id: str,
    *,
    job_id: str | None = None,
    status: str = "queued",
    celery_task_id: str | None = None,
) -> JobStatus:
    details: dict[str, Any] = {}
    if celery_task_id:
        details["celery_task_id"] = celery_task_id
    return JobStatus(
        job_id=job_id or new_job_id(),
        molecule_id=molecule_id,
        job_type="dft",
        status=status,
        created_at=utc_now(),
        details=details,
    )


def save_job(job: JobStatus, storage: ObjectStorage | None = None) -> str:
    store = storage or get_storage()
    settings = store.settings
    key = settings.job_key(job.job_id)
    store.put_json(key, manifest_to_dict(job))
    return key


def load_job(job_id: str, storage: ObjectStorage | None = None) -> JobStatus:
    store = storage or get_storage()
    key = store.settings.job_key(job_id)
    if not store.object_exists(key):
        raise FileNotFoundError(f"Job not found: {job_id}")
    data = store.get_json(key)
    return JobStatus.model_validate(data)


def update_job(
    job_id: str,
    *,
    status: str | None = None,
    details: dict[str, Any] | None = None,
    storage: ObjectStorage | None = None,
) -> JobStatus:
    job = load_job(job_id, storage)
    merged_details = {**job.details, **(details or {})}
    updates: dict[str, Any] = {"details": merged_details}
    if status is not None:
        updates["status"] = status
    updated = job.model_copy(update=updates)
    save_job(updated, storage)
    return updated


def find_dft_job_for_molecule(
    molecule_id: str,
    storage: ObjectStorage | None = None,
) -> str | None:
    """Read dft_job_id from molecule manifest if present."""
    store = storage or get_storage()
    key = store.settings.molecule_manifest_key(molecule_id)
    if not store.object_exists(key):
        return None
    data = store.get_json(key)
    job_id = data.get("dft_job_id")
    return str(job_id) if job_id else None
