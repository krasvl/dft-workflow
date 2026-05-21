"""Unit tests for the density-service model cache."""

from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from common.schemas import ActiveModelPointer, ModelManifest
from services.density_service.app.model import MockDensityModel
from services.density_service.app.model_cache import (
    ModelCache,
    _pick_best_version,
    _score,
    reset_model_cache,
)


def _checkpoint_bytes() -> bytes:
    buf = io.BytesIO()
    torch.save({"mock": True, "weights": torch.zeros(2, 2)}, buf)
    return buf.getvalue()


def _manifest(name: str, version: str) -> dict:
    return {
        "model_name": name,
        "version": version,
        "model_path": f"models/{name}/{version}/model.pt",
        "config_path": f"models/{name}/{version}/config.json",
        "metrics_path": f"models/{name}/{version}/metrics.json",
        "manifest_path": f"models/{name}/{version}/manifest.json",
        "created_at": "2026-05-17T00:00:00+00:00",
        "status": "active",
    }


def _make_storage(name: str, versions: dict[str, dict]) -> MagicMock:
    """versions: {version: metrics_dict_or_None}."""
    store = MagicMock()
    store.settings.default_model_name = name
    store.settings.models_prefix = "models"
    store.settings.model_dir_prefix = lambda n, v: f"models/{n}/{v}"
    store.settings.model_manifest_key = lambda n, v: f"models/{n}/{v}/manifest.json"
    store.settings.model_metrics_key = lambda n, v: f"models/{n}/{v}/metrics.json"
    store.settings.active_model_key = lambda n: f"models/active/{n}.json"

    keys = []
    for v in versions:
        keys.append(f"models/{name}/{v}/manifest.json")
        keys.append(f"models/{name}/{v}/metrics.json")
        keys.append(f"models/{name}/{v}/model.pt")
        keys.append(f"models/{name}/{v}/config.json")
    store.list_keys.return_value = keys

    def object_exists(key: str) -> bool:
        if key.endswith("/metrics.json"):
            v = key.split("/")[-2]
            return versions.get(v) is not None
        return any(key == k for k in keys)

    def get_json(key: str):
        v = key.split("/")[-2]
        if key.endswith("/manifest.json"):
            return _manifest(name, v)
        if key.endswith("/metrics.json"):
            return versions[v] or {}
        if key.endswith("/config.json"):
            return {"mock": True}
        raise KeyError(key)

    store.object_exists.side_effect = object_exists
    store.get_json.side_effect = get_json
    store.get_bytes.return_value = _checkpoint_bytes()
    return store


def test_score_prefers_best_loss_over_train_loss() -> None:
    assert _score({"best_loss": 0.1, "train_loss": 0.9}) == (0.1, "best_loss")
    assert _score({"train_loss": 0.5}) == (0.5, "train_loss")
    assert _score({"best_loss": float("nan"), "train_loss": 0.5}) == (0.5, "train_loss")
    assert _score({}) is None
    assert _score({"best_loss": "not a number"}) is None


def test_pick_best_version_picks_lowest_loss() -> None:
    store = _make_storage(
        "density_predictor",
        {"v1": {"train_loss": 0.5}, "v2": {"best_loss": 0.1}, "v3": {"best_loss": 0.2}},
    )
    result = _pick_best_version("density_predictor", store)
    assert result is not None
    manifest, metric, key, n = result
    assert manifest.version == "v2"
    assert metric == 0.1
    assert key == "best_loss"
    assert n == 3


def test_pick_best_version_skips_versions_without_metrics() -> None:
    store = _make_storage(
        "density_predictor",
        {"v1": None, "v2": {"train_loss": 0.4}, "v3": None},
    )
    result = _pick_best_version("density_predictor", store)
    assert result is not None
    manifest, _, _, n = result
    assert manifest.version == "v2"
    assert n == 1


def test_pick_best_version_returns_none_when_no_metrics() -> None:
    store = _make_storage("density_predictor", {"v1": None, "v2": None})
    assert _pick_best_version("density_predictor", store) is None


def test_cache_returns_same_entry_within_ttl() -> None:
    store = _make_storage("density_predictor", {"v1": {"best_loss": 0.5}})
    cache = ModelCache(ttl_sec=60)
    a = cache.get("density_predictor", store)
    b = cache.get("density_predictor", store)
    assert a is b
    assert store.get_bytes.call_count == 1  # loaded only once


def test_cache_reloads_after_ttl_expires() -> None:
    store = _make_storage("density_predictor", {"v1": {"best_loss": 0.5}})
    cache = ModelCache(ttl_sec=60)
    a = cache.get("density_predictor", store)

    with patch("services.density_service.app.model_cache.time.monotonic") as fake_time:
        fake_time.return_value = a.loaded_at + 120  # past TTL
        b = cache.get("density_predictor", store)
    assert a is not b
    assert store.get_bytes.call_count == 2


def test_cache_zero_ttl_always_refreshes() -> None:
    store = _make_storage("density_predictor", {"v1": {"best_loss": 0.5}})
    cache = ModelCache(ttl_sec=0)
    cache.get("density_predictor", store)
    cache.get("density_predictor", store)
    assert store.get_bytes.call_count == 2


def test_cache_invalidate_forces_reload() -> None:
    store = _make_storage("density_predictor", {"v1": {"best_loss": 0.5}})
    cache = ModelCache(ttl_sec=600)
    cache.get("density_predictor", store)
    cache.invalidate()
    cache.get("density_predictor", store)
    assert store.get_bytes.call_count == 2


def test_cache_falls_back_to_active_pointer_when_no_metrics() -> None:
    store = _make_storage("density_predictor", {"v1": None})

    pointer = ActiveModelPointer(
        model_name="density_predictor",
        version="v1",
        manifest_path="models/density_predictor/v1/manifest.json",
    )
    active_key = "models/active/density_predictor.json"

    orig_object_exists = store.object_exists.side_effect
    orig_get_json = store.get_json.side_effect

    def object_exists(key: str) -> bool:
        if key == active_key:
            return True
        return orig_object_exists(key)

    def get_json(key: str):
        if key == active_key:
            return pointer.model_dump()
        return orig_get_json(key)

    store.object_exists.side_effect = object_exists
    store.get_json.side_effect = get_json

    cache = ModelCache(ttl_sec=600)
    entry = cache.get("density_predictor", store)
    assert entry.selection == "active_pointer"
    assert entry.metric is None
    assert entry.manifest.version == "v1"
