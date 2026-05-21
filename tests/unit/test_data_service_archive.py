"""upload_molecules_from_archive: aggregates per-entry upload results."""

from __future__ import annotations

import io
import zipfile
from unittest.mock import MagicMock, patch

import pytest

from common.xyz import WATER
from services.data_service.app.archive import ArchiveError
from services.data_service.app.service import upload_molecules_from_archive

H2 = b"2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n"
BAD_XYZ = b"this is not xyz at all"


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w") as zf:
        for n, d in files.items():
            zf.writestr(n, d)
    return buf.getvalue()


def _fake_upload_factory():
    """Stand-in for ``upload_molecule`` that returns deterministic ids."""
    counter = {"i": 0}

    def fake_upload(content, filename, storage=None):
        counter["i"] += 1
        manifest = MagicMock()
        manifest.molecule_id = f"mol_{counter['i']}"
        return manifest, f"job_{counter['i']}"

    return fake_upload


def test_upload_archive_all_ok() -> None:
    payload = _zip({"water.xyz": WATER, "h2.xyz": H2})
    with patch(
        "services.data_service.app.service.upload_molecule",
        side_effect=_fake_upload_factory(),
    ):
        results = upload_molecules_from_archive(payload, "batch.zip", storage=MagicMock())

    assert len(results) == 2
    assert {r["status"] for r in results} == {"queued"}
    assert {r["filename"] for r in results} == {"water.xyz", "h2.xyz"}
    assert all(r["molecule_id"].startswith("mol_") for r in results)


def test_upload_archive_partial_failure_does_not_abort() -> None:
    payload = _zip({"good.xyz": WATER, "bad.xyz": BAD_XYZ})
    fake = _fake_upload_factory()

    from common.xyz import XyzValidationError

    def side_effect(content, filename, storage=None):
        if content == BAD_XYZ:
            raise XyzValidationError("malformed atom count")
        return fake(content, filename, storage=storage)

    with patch(
        "services.data_service.app.service.upload_molecule",
        side_effect=side_effect,
    ):
        results = upload_molecules_from_archive(payload, "batch.zip", storage=MagicMock())

    statuses = {r["filename"]: r["status"] for r in results}
    assert statuses == {"good.xyz": "queued", "bad.xyz": "rejected"}
    bad = next(r for r in results if r["filename"] == "bad.xyz")
    assert "malformed" in bad["error"]


def test_upload_archive_empty_raises() -> None:
    payload = _zip({"readme.md": b"x"})  # no .xyz at all
    with pytest.raises(ArchiveError, match="no usable .xyz"):
        upload_molecules_from_archive(payload, "batch.zip", storage=MagicMock())


def test_upload_archive_bad_container_raises() -> None:
    with pytest.raises(ArchiveError):
        upload_molecules_from_archive(b"", "batch.zip", storage=MagicMock())
