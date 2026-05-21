"""FastAPI route tests for data-service (TestClient, no MinIO/Redis)."""

from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from common.xyz import WATER
from services.data_service.app.archive import ArchiveError
from services.data_service.app.main import app
from services.data_service.app.service import (
    JobNotFoundError,
    MoleculeNotFoundError,
)

client = TestClient(app)


def test_health() -> None:
    r = client.get("/api/data/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "service": "data-service"}


def test_upload_molecule_route_success() -> None:
    fake_manifest = MagicMock()
    fake_manifest.molecule_id = "mol_test1"
    with patch(
        "services.data_service.app.api.upload_molecule",
        return_value=(fake_manifest, "job_test1"),
    ) as mock_upload:
        r = client.post(
            "/api/data/molecules",
            files={"file": ("water.xyz", WATER, "chemical/x-xyz")},
        )

    assert r.status_code == 201
    assert r.json() == {
        "molecule_id": "mol_test1",
        "dft_job_id": "job_test1",
        "status": "queued",
    }
    mock_upload.assert_called_once()


def test_upload_molecule_route_validation_error() -> None:
    r = client.post(
        "/api/data/molecules",
        files={"file": ("bad.xyz", b"not xyz", "chemical/x-xyz")},
    )
    assert r.status_code == 400
    assert "XYZ" in r.json()["detail"]


def test_batch_upload_route_returns_per_item_status() -> None:
    items = [
        {"filename": "a.xyz", "status": "queued",
         "molecule_id": "mol_a", "dft_job_id": "job_a"},
        {"filename": "b.xyz", "status": "rejected", "error": "bad xyz"},
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("dummy.xyz", b"x")

    with patch(
        "services.data_service.app.api.upload_molecules_from_archive",
        return_value=items,
    ):
        r = client.post(
            "/api/data/molecules/batch",
            files={"file": ("mols.zip", buf.getvalue(), "application/zip")},
        )

    assert r.status_code == 201
    body = r.json()
    assert body["total"] == 2
    assert body["queued"] == 1
    assert body["rejected"] == 1
    names = {it["filename"] for it in body["items"]}
    assert names == {"a.xyz", "b.xyz"}


def test_batch_upload_route_rejects_bad_archive() -> None:
    with patch(
        "services.data_service.app.api.upload_molecules_from_archive",
        side_effect=ArchiveError("not an archive"),
    ):
        r = client.post(
            "/api/data/molecules/batch",
            files={"file": ("mols.zip", b"junk", "application/zip")},
        )
    assert r.status_code == 400
    assert "not an archive" in r.json()["detail"]


def test_get_molecule_404() -> None:
    with patch(
        "services.data_service.app.api.get_molecule",
        side_effect=MoleculeNotFoundError("mol_404"),
    ):
        r = client.get("/api/data/molecules/mol_404")
    assert r.status_code == 404


def test_get_job_404() -> None:
    with patch(
        "services.data_service.app.api.get_job",
        side_effect=JobNotFoundError("job_404"),
    ):
        r = client.get("/api/data/jobs/job_404")
    assert r.status_code == 404
