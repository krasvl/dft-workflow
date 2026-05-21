"""Business logic for molecule upload and job tracking."""

import logging

from common.celery_client import enqueue_dft_calculation
from common.ids import new_molecule_id
from common.jobs import build_dft_job, load_job, save_job
from common.manifests import build_molecule_manifest, manifest_to_dict
from common.schemas import MoleculeManifest
from common.storage import ObjectStorage, get_storage

from services.data_service.app.archive import ArchiveError, iter_archive_xyz
from services.data_service.app.xyz import XyzValidationError, validate_xyz

logger = logging.getLogger("data_service")


class MoleculeNotFoundError(Exception):
    pass


class JobNotFoundError(Exception):
    pass


def upload_molecule(
    content: bytes,
    filename: str | None,
    storage: ObjectStorage | None = None,
) -> tuple[MoleculeManifest, str]:
    """Validate XYZ, persist to MinIO, create a manifest, enqueue a DFT job.

    Returns the molecule manifest and the assigned ``dft_job_id``.
    """
    store = storage or get_storage()
    settings = store.settings

    validate_xyz(content, filename)

    molecule_id = new_molecule_id()
    raw_key = settings.molecule_raw_key(molecule_id)
    manifest_key = settings.molecule_manifest_key(molecule_id)

    store.put_bytes(raw_key, content, content_type="chemical/x-xyz")

    manifest = build_molecule_manifest(
        molecule_id,
        raw_key,
        status="uploaded",
    )
    store.put_json(manifest_key, manifest_to_dict(manifest))

    job = build_dft_job(molecule_id)
    save_job(job, store)

    manifest = manifest.model_copy(update={"status": "dft_queued"})
    manifest_payload = manifest_to_dict(manifest)
    manifest_payload["dft_job_id"] = job.job_id
    store.put_json(manifest_key, manifest_payload)

    celery_task_id = enqueue_dft_calculation(molecule_id)
    save_job(
        job.model_copy(
            update={"details": {**job.details, "celery_task_id": celery_task_id}},
        ),
        store,
    )

    logger.info(
        "molecule_uploaded molecule_id=%s job_id=%s celery_task_id=%s",
        molecule_id,
        job.job_id,
        celery_task_id,
    )
    return manifest, job.job_id


def upload_molecules_from_archive(
    content: bytes,
    filename: str | None,
    storage: ObjectStorage | None = None,
) -> list[dict]:
    """Extract every ``.xyz`` from a zip/tar archive and upload each one.

    Per-entry failures (invalid XYZ, etc.) are recorded in the result but do
    not abort the batch. Raises :class:`ArchiveError` if the container itself
    is invalid or contains no usable entries.
    """
    store = storage or get_storage()
    results: list[dict] = []

    for entry_name, xyz_content in iter_archive_xyz(content, filename):
        try:
            manifest, dft_job_id = upload_molecule(
                xyz_content, entry_name, storage=store
            )
        except XyzValidationError as exc:
            results.append({
                "filename": entry_name,
                "status": "rejected",
                "error": str(exc),
            })
            continue
        except Exception as exc:
            logger.exception("archive_upload_entry_failed entry=%s", entry_name)
            results.append({
                "filename": entry_name,
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
            })
            continue

        results.append({
            "filename": entry_name,
            "status": "queued",
            "molecule_id": manifest.molecule_id,
            "dft_job_id": dft_job_id,
        })

    if not results:
        raise ArchiveError("archive contained no usable .xyz entries")

    logger.info(
        "archive_uploaded total=%d queued=%d rejected=%d",
        len(results),
        sum(1 for r in results if r["status"] == "queued"),
        sum(1 for r in results if r["status"] != "queued"),
    )
    return results


def get_molecule(molecule_id: str, storage: ObjectStorage | None = None) -> dict:
    store = storage or get_storage()
    key = store.settings.molecule_manifest_key(molecule_id)
    if not store.object_exists(key):
        raise MoleculeNotFoundError(molecule_id)
    return store.get_json(key)


def get_job(job_id: str, storage: ObjectStorage | None = None) -> dict:
    try:
        job = load_job(job_id, storage)
    except FileNotFoundError as exc:
        raise JobNotFoundError(job_id) from exc
    return job.model_dump(mode="json")
