from fastapi import APIRouter, File, HTTPException, UploadFile

from services.data_service.app.archive import ArchiveError
from services.data_service.app.schemas import (
    JobResponse,
    MoleculeBatchItem,
    MoleculeBatchUploadResponse,
    MoleculeResponse,
    MoleculeUploadResponse,
)
from services.data_service.app.service import (
    JobNotFoundError,
    MoleculeNotFoundError,
    get_job,
    get_molecule,
    upload_molecule,
    upload_molecules_from_archive,
)
from services.data_service.app.xyz import XyzValidationError

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/health")
def api_health() -> dict[str, str]:
    return {"status": "ok", "service": "data-service"}


@router.post("/molecules", response_model=MoleculeUploadResponse, status_code=201)
async def create_molecule(file: UploadFile = File(...)) -> MoleculeUploadResponse:
    content = await file.read()
    try:
        _manifest, job_id = upload_molecule(content, file.filename)
    except XyzValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to upload molecule") from exc

    molecule_id = _manifest.molecule_id
    return MoleculeUploadResponse(
        molecule_id=molecule_id,
        dft_job_id=job_id,
        status="queued",
    )


@router.post(
    "/molecules/batch",
    response_model=MoleculeBatchUploadResponse,
    status_code=201,
)
async def create_molecules_batch(
    file: UploadFile = File(...),
) -> MoleculeBatchUploadResponse:
    """Bulk upload: accept a .zip / .tar(.gz|.bz2|.xz) archive of .xyz files.

    Per-entry validation failures are reported in the response body but do
    not roll back the rest of the batch.
    """
    content = await file.read()
    try:
        items = upload_molecules_from_archive(content, file.filename)
    except ArchiveError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to process archive") from exc

    queued = sum(1 for it in items if it["status"] == "queued")
    return MoleculeBatchUploadResponse(
        total=len(items),
        queued=queued,
        rejected=len(items) - queued,
        items=[MoleculeBatchItem.model_validate(it) for it in items],
    )


@router.get("/molecules/{molecule_id}", response_model=MoleculeResponse)
def read_molecule(molecule_id: str) -> MoleculeResponse:
    try:
        data = get_molecule(molecule_id)
    except MoleculeNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Molecule not found: {molecule_id}") from exc
    return MoleculeResponse.model_validate({"manifest": data})


@router.get("/jobs/{job_id}", response_model=JobResponse)
def read_job(job_id: str) -> JobResponse:
    try:
        data = get_job(job_id)
    except JobNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}") from exc
    return JobResponse.model_validate({"job": data})
