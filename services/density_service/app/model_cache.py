"""TTL cache of the currently served inference model.

On a miss (or after expiry) the cache scans MinIO for every version of
``model_name``, picks the one with the lowest available metric (``best_loss``
preferred over ``train_loss``), loads its checkpoint and returns it. The
"active model" pointer published by the training worker is only used as a
fallback when no version exposes usable metrics.

The cache is process-local: each uvicorn worker maintains its own copy.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Any

from common.schemas import ModelManifest
from common.settings import get_settings
from common.storage import ObjectStorage, get_storage
from services.density_service.app.model import DensityModel, load_density_model
from services.density_service.app.service import (
    ActiveModelNotFoundError,
    get_active_model,
)

logger = logging.getLogger("density_service.cache")

_METRIC_PRIORITY = ("best_loss", "train_loss")


@dataclass
class CachedModel:
    manifest: ModelManifest
    model: DensityModel
    selection: str  # "best_loss" | "train_loss" | "active_pointer"
    metric: float | None
    loaded_at: float  # time.monotonic()
    versions_considered: int


def _score(metrics: dict[str, Any]) -> tuple[float, str] | None:
    for key in _METRIC_PRIORITY:
        if key not in metrics:
            continue
        try:
            value = float(metrics[key])
        except (TypeError, ValueError):
            continue
        if math.isnan(value):
            continue
        return value, key
    return None


def _list_versions(model_name: str, storage: ObjectStorage) -> list[str]:
    settings = storage.settings
    prefix = f"{settings.models_prefix}/{model_name}/"
    versions: set[str] = set()
    for key in storage.list_keys(prefix):
        rest = key[len(prefix):]
        parts = rest.split("/")
        if len(parts) < 2:
            continue
        version = parts[0]
        if not version or version == "active":
            continue
        versions.add(version)
    return sorted(versions)


def _load_checkpoint(manifest: ModelManifest, storage: ObjectStorage) -> DensityModel:
    weights = storage.get_bytes(manifest.model_path)
    config: dict[str, Any] = {}
    if storage.object_exists(manifest.config_path):
        config = storage.get_json(manifest.config_path)
    use_mock = bool(config.get("mock", False))
    return load_density_model(weights, mock=use_mock, storage=storage)


def _pick_best_version(
    model_name: str, storage: ObjectStorage
) -> tuple[ModelManifest, float, str, int] | None:
    settings = storage.settings
    candidates: list[tuple[float, str, str]] = []
    for version in _list_versions(model_name, storage):
        metrics_key = settings.model_metrics_key(model_name, version)
        if not storage.object_exists(metrics_key):
            continue
        try:
            metrics = storage.get_json(metrics_key)
        except Exception:
            continue
        scored = _score(metrics)
        if scored is None:
            continue
        candidates.append((scored[0], scored[1], version))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0])
    score, key, version = candidates[0]
    manifest_key = settings.model_manifest_key(model_name, version)
    if not storage.object_exists(manifest_key):
        return None
    manifest = ModelManifest.model_validate(storage.get_json(manifest_key))
    return manifest, score, key, len(candidates)


class ModelCache:
    def __init__(self, ttl_sec: int) -> None:
        self.ttl_sec = max(0, int(ttl_sec))
        self._cached: dict[str, CachedModel] = {}
        self._lock = threading.Lock()

    def _is_fresh(self, entry: CachedModel) -> bool:
        if self.ttl_sec == 0:
            return False
        return (time.monotonic() - entry.loaded_at) < self.ttl_sec

    def peek(self, model_name: str) -> CachedModel | None:
        with self._lock:
            return self._cached.get(model_name)

    def get(self, model_name: str, storage: ObjectStorage) -> CachedModel:
        with self._lock:
            cached = self._cached.get(model_name)
            if cached is not None and self._is_fresh(cached):
                return cached

            entry = self._refresh(model_name, storage)
            self._cached[model_name] = entry
            return entry

    def _refresh(self, model_name: str, storage: ObjectStorage) -> CachedModel:
        picked = _pick_best_version(model_name, storage)
        if picked is not None:
            manifest, score, key, n_candidates = picked
            model = _load_checkpoint(manifest, storage)
            logger.info(
                "model_cache_refresh selection=%s version=%s %s=%.4e candidates=%d",
                key, manifest.version, key, score, n_candidates,
            )
            return CachedModel(
                manifest=manifest,
                model=model,
                selection=key,
                metric=score,
                loaded_at=time.monotonic(),
                versions_considered=n_candidates,
            )

        _pointer, manifest = get_active_model(model_name=model_name, storage=storage)
        model = _load_checkpoint(manifest, storage)
        logger.info(
            "model_cache_refresh selection=active_pointer version=%s "
            "(no metrics found)",
            manifest.version,
        )
        return CachedModel(
            manifest=manifest,
            model=model,
            selection="active_pointer",
            metric=None,
            loaded_at=time.monotonic(),
            versions_considered=0,
        )

    def invalidate(self, model_name: str | None = None) -> None:
        with self._lock:
            if model_name is None:
                self._cached.clear()
            else:
                self._cached.pop(model_name, None)


_cache: ModelCache | None = None
_cache_lock = threading.Lock()


def get_model_cache() -> ModelCache:
    global _cache
    with _cache_lock:
        if _cache is None:
            settings = get_settings()
            _cache = ModelCache(ttl_sec=settings.model_cache_ttl_sec)
        return _cache


def reset_model_cache() -> None:
    """For tests: drop the singleton so the next call rebuilds it from settings."""
    global _cache
    with _cache_lock:
        _cache = None


def serve_model(
    model_name: str | None = None,
    storage: ObjectStorage | None = None,
) -> CachedModel:
    """Return a :class:`CachedModel` ready to handle a prediction request."""
    store = storage or get_storage()
    name = model_name or store.settings.default_model_name
    return get_model_cache().get(name, store)
