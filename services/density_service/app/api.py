"""FastAPI routes for the density service.

URL layout::

    GET  /api/density/health
    GET  /api/density/models                   — list available models
    GET  /api/density/models/active            — pointer to the active MACE version
    GET  /api/density/models/serving           — what the in-process cache serves now
    POST /api/density/models/cache/invalidate

    POST /api/density/predict/dft              — PySCF SCF
    POST /api/density/predict/mace             — single MACE forward
    POST /api/density/predict/mace/with-scf    — MACE forward used as dm0, then PySCF SCF

All ``/predict`` routes accept either a multipart ``file=@*.xyz`` upload or a
JSON body ``{"xyz": "..."}`` and return the uniform :class:`PredictResponse`.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from common.xyz import XyzValidationError
from services.density_service.app.model_cache import get_model_cache, serve_model
from services.density_service.app.schemas import (
    ActiveModelResponse,
    ModelInfo,
    ModelListResponse,
    PredictJsonBody,
    PredictResponse,
    ServingModelResponse,
)
from services.density_service.app.service import (
    ActiveModelNotFoundError,
    MoleculeNotFoundError,
    get_active_model,
    predict_dft,
    predict_mace,
    predict_mace_with_scf,
)

router = APIRouter(prefix="/api/density", tags=["density"])


@router.get("/health")
def api_health() -> dict[str, str]:
    return {"status": "ok", "service": "density-service"}


# --- Model discovery -------------------------------------------------------


@router.get("/models", response_model=ModelListResponse)
def list_models() -> ModelListResponse:
    mace_available = True
    try:
        serve_model()
    except ActiveModelNotFoundError:
        mace_available = False

    return ModelListResponse(
        models=[
            ModelInfo(
                name="dft",
                kind="scf",
                available=True,
                description="PySCF self-consistent field (RKS / UKS).",
            ),
            ModelInfo(
                name="mace",
                kind="ml",
                available=mace_available,
                description=(
                    "MACE-based density predictor; the model cache picks the "
                    "best-metric version available in MinIO."
                ),
            ),
        ]
    )


@router.get("/models/active", response_model=ActiveModelResponse)
def read_active_model() -> ActiveModelResponse:
    try:
        pointer, manifest = get_active_model()
    except ActiveModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ActiveModelResponse(pointer=pointer, manifest=manifest)


@router.get("/models/serving", response_model=ServingModelResponse)
def read_serving_model() -> ServingModelResponse:
    cache = get_model_cache()
    try:
        cached = serve_model()
    except ActiveModelNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return ServingModelResponse(
        manifest=cached.manifest,
        selection=cached.selection,
        metric=cached.metric,
        cache_age_sec=round(time.monotonic() - cached.loaded_at, 3),
        cache_ttl_sec=cache.ttl_sec,
        versions_considered=cached.versions_considered,
    )


@router.post("/models/cache/invalidate")
def invalidate_cache() -> dict[str, str]:
    get_model_cache().invalidate()
    return {"status": "invalidated"}


# --- Predict endpoints -----------------------------------------------------


async def _read_xyz(
    file: UploadFile | None,
    body: PredictJsonBody | None,
) -> tuple[bytes, str | None]:
    if file is not None:
        return await file.read(), file.filename
    if body is not None:
        return body.xyz.encode("utf-8"), "molecule.xyz"
    raise HTTPException(
        status_code=400,
        detail="Provide multipart file or JSON body with xyz",
    )


def _handle_predict_errors(exc: Exception) -> None:
    if isinstance(exc, XyzValidationError):
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if isinstance(exc, MoleculeNotFoundError):
        raise HTTPException(status_code=404, detail=f"Molecule not found: {exc}") from exc
    if isinstance(exc, ActiveModelNotFoundError):
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@router.post("/predict/dft", response_model=PredictResponse)
async def predict_dft_route(
    file: UploadFile | None = File(default=None),
    body: PredictJsonBody | None = Body(default=None),
) -> PredictResponse:
    content, filename = await _read_xyz(file, body)
    try:
        result = predict_dft(xyz_content=content, filename=filename)
    except Exception as exc:
        _handle_predict_errors(exc)
    return PredictResponse.model_validate(result)


@router.post("/predict/dft/by-id/{molecule_id}", response_model=PredictResponse)
def predict_dft_by_id(molecule_id: str) -> PredictResponse:
    try:
        result = predict_dft(molecule_id=molecule_id)
    except Exception as exc:
        _handle_predict_errors(exc)
    return PredictResponse.model_validate(result)


@router.post("/predict/mace", response_model=PredictResponse)
async def predict_mace_route(
    file: UploadFile | None = File(default=None),
    body: PredictJsonBody | None = Body(default=None),
) -> PredictResponse:
    content, filename = await _read_xyz(file, body)
    try:
        result = predict_mace(xyz_content=content, filename=filename)
    except Exception as exc:
        _handle_predict_errors(exc)
    return PredictResponse.model_validate(result)


@router.post("/predict/mace/by-id/{molecule_id}", response_model=PredictResponse)
def predict_mace_by_id(molecule_id: str) -> PredictResponse:
    try:
        result = predict_mace(molecule_id=molecule_id)
    except Exception as exc:
        _handle_predict_errors(exc)
    return PredictResponse.model_validate(result)


@router.post("/predict/mace/with-scf", response_model=PredictResponse)
async def predict_mace_with_scf_route(
    file: UploadFile | None = File(default=None),
    body: PredictJsonBody | None = Body(default=None),
) -> PredictResponse:
    content, filename = await _read_xyz(file, body)
    try:
        result = predict_mace_with_scf(xyz_content=content, filename=filename)
    except Exception as exc:
        _handle_predict_errors(exc)
    return PredictResponse.model_validate(result)


@router.post("/predict/mace/with-scf/by-id/{molecule_id}", response_model=PredictResponse)
def predict_mace_with_scf_by_id(molecule_id: str) -> PredictResponse:
    try:
        result = predict_mace_with_scf(molecule_id=molecule_id)
    except Exception as exc:
        _handle_predict_errors(exc)
    return PredictResponse.model_validate(result)
