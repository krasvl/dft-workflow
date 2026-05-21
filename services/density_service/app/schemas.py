"""API schemas for Density Service."""

from typing import Any, Literal

from pydantic import BaseModel, Field

from common.schemas import ActiveModelPointer, ModelManifest


class ActiveModelResponse(BaseModel):
    pointer: ActiveModelPointer
    manifest: ModelManifest


class ServingModelResponse(BaseModel):
    manifest: ModelManifest
    selection: str = Field(..., description="best_loss | train_loss | active_pointer")
    metric: float | None = None
    cache_age_sec: float
    cache_ttl_sec: int
    versions_considered: int


class ModelInfo(BaseModel):
    name: Literal["dft", "mace"]
    kind: Literal["scf", "ml"]
    available: bool
    description: str


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class PredictJsonBody(BaseModel):
    xyz: str = Field(..., description="XYZ file content as text")


class PredictResponse(BaseModel):
    """Uniform response for every ``/predict/...`` route.

    Fields the user always wants are at the top level; route-specific
    metadata (method/basis/model_version/cache info) goes into ``details``.
    """

    model: Literal["dft", "mace", "mace+scf"]
    shape: list[int]
    preview: list[list[float]]
    scf_iterations: int
    wall_time_sec: float
    request_id: str
    artifact_path: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
