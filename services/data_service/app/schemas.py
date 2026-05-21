"""API request/response schemas for Data Service."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from common.schemas import JobStatus, MoleculeManifest


class MoleculeUploadResponse(BaseModel):
    molecule_id: str
    dft_job_id: str
    status: str = "queued"


class MoleculeBatchItem(BaseModel):
    filename: str
    status: str = Field(..., description="queued | rejected | error")
    molecule_id: str | None = None
    dft_job_id: str | None = None
    error: str | None = None


class MoleculeBatchUploadResponse(BaseModel):
    total: int
    queued: int
    rejected: int
    items: list[MoleculeBatchItem]


class MoleculeResponse(BaseModel):
    manifest: MoleculeManifest


class JobResponse(BaseModel):
    job: JobStatus


class ErrorResponse(BaseModel):
    detail: str


class MoleculeManifestUpdate(BaseModel):
    """Optional fields for manifest updates from workers."""

    status: str | None = None
    dft_job_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class MoleculeManifestOut(BaseModel):
    molecule_id: str
    source_file: str
    format: str
    status: str
    created_at: datetime
    dft_job_id: str | None = None
