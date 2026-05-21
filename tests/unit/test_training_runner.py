from unittest.mock import patch

import pytest

from workers.training_worker.app.training_runner import run_training


def test_run_training_mock_engine() -> None:
    with patch("workers.training_worker.app.training_runner.get_settings") as gs:
        gs.return_value.training_engine = "mock"
        out_bytes, config, metrics = run_training(
            model_name="density_predictor",
            version="v_test",
            dft_manifests=[{"calculation_id": "dft_a", "status": "completed"}],
            train_config={"source": "test"},
        )

    assert out_bytes
    assert config["model_name"] == "density_predictor"
    assert config["mock"] is True
    assert "train_loss" in metrics


def test_run_training_engine_override_via_train_config() -> None:
    with patch("workers.training_worker.app.training_runner.get_settings") as gs:
        gs.return_value.training_engine = "mace"  # default would pick mace
        out_bytes, config, metrics = run_training(
            model_name="density_predictor",
            version="v_test",
            dft_manifests=[],
            train_config={"engine": "mock"},
        )

    assert config["mock"] is True
    assert metrics["samples"] == 0


def test_run_training_unknown_engine() -> None:
    with patch("workers.training_worker.app.training_runner.get_settings") as gs:
        gs.return_value.training_engine = "bogus"
        with pytest.raises(ValueError, match="Unknown TRAINING_ENGINE"):
            run_training(
                model_name="m",
                version="v",
                dft_manifests=[],
                train_config={},
            )
