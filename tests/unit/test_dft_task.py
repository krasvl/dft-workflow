from unittest.mock import MagicMock, patch

from common.training_trigger import TrainingBatchResult
from common.xyz import WATER
from dft.mock import run_mock_dft
from workers.dft_worker.app.tasks import run_dft_calculation


def test_run_dft_calculation() -> None:
    mock_storage = MagicMock()
    mock_storage.settings.molecule_raw_key.side_effect = lambda mid: f"molecules/raw/{mid}.xyz"
    mock_storage.settings.molecule_manifest_key.side_effect = lambda mid: f"molecules/manifests/{mid}.json"
    mock_storage.settings.dft_artifact_key.side_effect = lambda mid, cid: f"dft/artifacts/{mid}/{cid}.npz"
    mock_storage.settings.dft_manifest_key.side_effect = lambda cid: f"dft/manifests/{cid}.json"
    mock_storage.object_exists.return_value = True
    mock_storage.get_bytes.return_value = WATER
    mock_storage.get_json.return_value = {
        "molecule_id": "mol_test",
        "dft_job_id": "job_test",
        "status": "dft_queued",
    }

    with (
        patch("workers.dft_worker.app.tasks.get_storage", return_value=mock_storage),
        patch(
            "workers.dft_worker.app.tasks.run_dft",
            side_effect=lambda xyz, method=None, basis=None: run_mock_dft(xyz),
        ),
        patch("workers.dft_worker.app.tasks.find_dft_job_for_molecule", return_value="job_test"),
        patch("workers.dft_worker.app.tasks.update_job") as mock_update_job,
        patch(
            "workers.dft_worker.app.tasks.record_dft_completed_and_maybe_enqueue_training",
            return_value=TrainingBatchResult(triggered=False, threshold=1000),
        ),
    ):
        result = run_dft_calculation("mol_test", {})

    assert result["status"] == "completed"
    assert result["molecule_id"] == "mol_test"
    assert result["calculation_id"].startswith("dft_")
    assert result["training_triggered"] is False
    assert result["training_task_id"] is None
    mock_storage.put_bytes.assert_called_once()
    assert mock_storage.put_json.call_count >= 2
    mock_update_job.assert_any_call("job_test", status="running", storage=mock_storage)
    completed_calls = [
        c
        for c in mock_update_job.call_args_list
        if c.kwargs.get("status") == "completed"
    ]
    assert len(completed_calls) == 1
