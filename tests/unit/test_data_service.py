from unittest.mock import MagicMock, patch

import pytest

from services.data_service.app.service import (
    JobNotFoundError,
    MoleculeNotFoundError,
    get_job,
    get_molecule,
    upload_molecule,
)
from services.data_service.app.xyz import WATER


@pytest.fixture
def mock_storage() -> MagicMock:
    store = MagicMock()
    store.settings.molecule_raw_key.side_effect = lambda mid: f"molecules/raw/{mid}.xyz"
    store.settings.molecule_manifest_key.side_effect = lambda mid: f"molecules/manifests/{mid}.json"
    store.settings.job_key.side_effect = lambda jid: f"jobs/{jid}.json"
    store.settings.minio_bucket = "dft-workflow"
    return store


def test_upload_molecule_persists_and_enqueues(mock_storage: MagicMock) -> None:
    with patch("services.data_service.app.service.enqueue_dft_calculation", return_value="celery-task-1"):
        manifest, job_id = upload_molecule(WATER, "water.xyz", storage=mock_storage)

    assert manifest.molecule_id.startswith("mol_")
    assert job_id.startswith("job_")
    mock_storage.put_bytes.assert_called_once()
    assert mock_storage.put_json.call_count >= 2


def test_upload_sets_dft_job_id_before_enqueue(mock_storage: MagicMock) -> None:
    def enqueue_after_manifest(molecule_id: str, *args: object, **kwargs: object) -> str:
        key = mock_storage.settings.molecule_manifest_key(molecule_id)
        manifest_calls = [
            call.args[1]
            for call in mock_storage.put_json.call_args_list
            if call.args[0] == key
        ]
        assert any(payload.get("dft_job_id") for payload in manifest_calls)
        return "celery-task-1"

    with patch(
        "services.data_service.app.service.enqueue_dft_calculation",
        side_effect=enqueue_after_manifest,
    ):
        upload_molecule(WATER, "water.xyz", storage=mock_storage)


def test_get_molecule_not_found(mock_storage: MagicMock) -> None:
    mock_storage.object_exists.return_value = False
    with pytest.raises(MoleculeNotFoundError):
        get_molecule("mol_missing", storage=mock_storage)


def test_get_job_not_found(mock_storage: MagicMock) -> None:
    mock_storage.object_exists.return_value = False
    with pytest.raises(JobNotFoundError):
        get_job("job_missing", storage=mock_storage)
