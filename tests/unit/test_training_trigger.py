from unittest.mock import MagicMock, patch

from common.settings import Settings
from common.training_trigger import (
    TrainingBatchResult,
    record_dft_completed_and_maybe_enqueue_training,
)


def test_no_training_below_threshold() -> None:
    mock_script = MagicMock(return_value=[0, 5])
    settings = Settings(training_batch_min_samples=1000)

    with patch(
        "common.training_trigger.get_batch_counter_script",
        return_value=mock_script,
    ):
        result = record_dft_completed_and_maybe_enqueue_training(
            "mol_a",
            "dft_a",
            settings=settings,
        )

    assert result == TrainingBatchResult(
        triggered=False,
        pending_count=5,
        threshold=1000,
    )
    mock_script.assert_called_once_with(
        keys=[settings.training_batch_counter_key],
        args=["1000"],
    )


def test_training_enqueued_at_threshold() -> None:
    mock_script = MagicMock(return_value=[1, 1000])
    settings = Settings(training_batch_min_samples=1000)

    with (
        patch(
            "common.training_trigger.get_batch_counter_script",
            return_value=mock_script,
        ),
        patch(
            "common.training_trigger.enqueue_training",
            return_value="celery-train-1",
        ) as mock_enqueue,
    ):
        result = record_dft_completed_and_maybe_enqueue_training(
            "mol_b",
            "dft_b",
            settings=settings,
        )

    assert result.triggered is True
    assert result.training_task_id == "celery-train-1"
    assert result.completed_count_at_trigger == 1000
    mock_enqueue.assert_called_once()
    config = mock_enqueue.call_args.kwargs["train_config"]
    assert config["source"] == "dft_batch"
    assert config["batch_size"] == 1000
