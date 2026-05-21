"""Discover DFT artifacts available for training from MinIO."""

from __future__ import annotations

from typing import Any

from common.storage import ObjectStorage, get_storage


def list_completed_dft_manifests(
    storage: ObjectStorage | None = None,
) -> list[dict[str, Any]]:
    """Load all DFT manifests with status ``completed``."""
    store = storage or get_storage()
    prefix = f"{store.settings.dft_manifests_prefix}/"
    manifests: list[dict[str, Any]] = []

    for key in store.list_keys(prefix):
        if not key.endswith(".json"):
            continue
        data = store.get_json(key)
        if data.get("status") == "completed":
            manifests.append(data)

    return manifests
