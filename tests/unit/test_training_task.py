from unittest.mock import MagicMock, patch

from workers.training_worker.app.tasks import train_model


def test_train_model_persists_artifacts() -> None:
    mock_storage = MagicMock()
    mock_storage.settings.default_model_name = "mace"
    mock_storage.settings.dft_manifests_prefix = "dft/manifests"
    mock_storage.settings.model_dir_prefix.side_effect = (
        lambda name, ver: f"models/{name}/{ver}"
    )
    mock_storage.settings.model_weights_key.side_effect = (
        lambda name, ver: f"models/{name}/{ver}/model.pt"
    )
    mock_storage.settings.model_manifest_key.side_effect = (
        lambda name, ver: f"models/{name}/{ver}/manifest.json"
    )
    mock_storage.settings.active_model_key.side_effect = (
        lambda name: f"models/active/{name}.json"
    )

    mock_result = MagicMock()
    mock_result.model_bytes = b"mock-pt"
    mock_result.config = {"mock": True}
    mock_result.metrics = {"train_loss": 0.1, "samples": 1}

    with (
        patch("workers.training_worker.app.tasks.get_storage", return_value=mock_storage),
        patch(
            "workers.training_worker.app.tasks.list_completed_dft_manifests",
            return_value=[{"calculation_id": "dft_1", "status": "completed"}],
        ),
        patch(
            "workers.training_worker.app.tasks.run_training",
            return_value=(mock_result.model_bytes, mock_result.config, mock_result.metrics),
        ),
        patch(
            "workers.training_worker.app.tasks.new_model_version",
            return_value="v_test",
        ),
    ):
        out = train_model(train_config={"source": "test"})

    assert out["model_name"] == "mace"
    assert out["version"] == "v_test"
    assert out["status"] == "active"
    mock_storage.put_bytes.assert_called_once()
    assert mock_storage.put_json.call_count == 4
