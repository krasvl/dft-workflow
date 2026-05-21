"""Pydantic schemas for manifests and API payloads."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class MoleculeManifest(BaseModel):
    molecule_id: str
    source_file: str
    format: str = "xyz"
    status: str = "uploaded"
    created_at: datetime
    dft_job_id: str | None = None


class DftMetrics(BaseModel):
    wall_time_sec: float
    scf_iterations: int
    energy: float | None = None


class DftManifest(BaseModel):
    calculation_id: str
    molecule_id: str
    status: str
    method: str
    basis: str
    artifact_path: str
    metrics: DftMetrics
    created_at: datetime | None = None


class ModelManifest(BaseModel):
    model_name: str
    version: str
    model_path: str
    config_path: str
    metrics_path: str
    manifest_path: str | None = None
    created_at: datetime
    status: str = "active"


class ActiveModelPointer(BaseModel):
    model_name: str
    version: str
    manifest_path: str


class JobStatus(BaseModel):
    job_id: str
    molecule_id: str | None = None
    job_type: str
    status: str
    created_at: datetime
    details: dict[str, Any] = Field(default_factory=dict)
