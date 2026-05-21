"""Integration test against a running MinIO (Docker Compose)."""

import os

import pytest

from common.manifests import build_molecule_manifest, manifest_to_dict
from common.storage import ObjectStorage

MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")


def _minio_reachable() -> bool:
    try:
        storage = ObjectStorage()
        storage.settings.minio_endpoint = MINIO_ENDPOINT
        storage._client = None  # reset cached client
        storage.ensure_bucket()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _minio_reachable(),
    reason=f"MinIO not reachable at {MINIO_ENDPOINT}",
)


@pytest.fixture
def storage() -> ObjectStorage:
    s = ObjectStorage()
    s.settings.minio_endpoint = MINIO_ENDPOINT
    s._client = None
    s.ensure_bucket()
    return s


def test_roundtrip_json(storage: ObjectStorage) -> None:
    key = "tests/integration_probe.json"
    manifest = build_molecule_manifest(
        "mol_integration",
        "molecules/raw/mol_integration.xyz",
    )
    payload = manifest_to_dict(manifest)

    storage.put_json(key, payload)
    assert storage.object_exists(key)

    loaded = storage.get_json(key)
    assert loaded["molecule_id"] == "mol_integration"
    assert loaded["format"] == "xyz"
