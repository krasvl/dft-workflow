"""Batch training trigger: enqueue training after N completed DFT runs (Redis counter)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache

import redis
from redis.commands.core import Script

from common.celery_client import enqueue_training
from common.settings import Settings, get_settings

logger = logging.getLogger("training_trigger")

# Atomic INCR + threshold check + reset.
# Returns {triggered_flag, count} where triggered_flag is 1 if training should run.
_BATCH_LUA = """
local count = redis.call('INCR', KEYS[1])
local threshold = tonumber(ARGV[1])
if count >= threshold then
    redis.call('SET', KEYS[1], 0)
    return {1, count}
end
return {0, count}
"""


@dataclass
class TrainingBatchResult:
    """Outcome of incrementing the DFT completion counter."""

    triggered: bool
    pending_count: int = 0
    completed_count_at_trigger: int | None = None
    training_task_id: str | None = None
    threshold: int = 0


@lru_cache
def get_redis_client() -> redis.Redis:
    settings = get_settings()
    return redis.Redis.from_url(settings.redis_url, decode_responses=True)


@lru_cache
def get_batch_counter_script() -> Script:
    return get_redis_client().register_script(_BATCH_LUA)


def _increment_batch_counter(key: str, threshold: int) -> tuple[int, int]:
    """Run the atomic INCR / threshold / reset Redis script.

    Returns ``(triggered_flag, count)`` where ``triggered_flag`` is 1 if the
    batch threshold was reached and the counter was reset to 0.
    """
    raw = get_batch_counter_script()(keys=[key], args=[str(threshold)])
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise RuntimeError(f"Unexpected Redis script response: {raw!r}")
    return int(raw[0]), int(raw[1])


def record_dft_completed_and_maybe_enqueue_training(
    molecule_id: str,
    calculation_id: str,
    *,
    settings: Settings | None = None,
) -> TrainingBatchResult:
    """
    Increment Redis counter of completed DFT calculations.

    When the counter reaches ``training_batch_min_samples``, reset it and enqueue
    exactly one Celery training job for the accumulated batch.
    """
    cfg = settings or get_settings()
    threshold = cfg.training_batch_min_samples
    key = cfg.training_batch_counter_key

    triggered_flag, pending_count = _increment_batch_counter(key, threshold)

    if triggered_flag == 0:
        logger.info(
            "training_batch_pending count=%s threshold=%s molecule_id=%s",
            pending_count,
            threshold,
            molecule_id,
        )
        return TrainingBatchResult(
            triggered=False,
            pending_count=pending_count,
            threshold=threshold,
        )

    logger.info(
        "training_batch_triggered count=%s threshold=%s molecule_id=%s",
        pending_count,
        threshold,
        molecule_id,
    )
    training_task_id = enqueue_training(
        train_config={
            "source": "dft_batch",
            "trigger_molecule_id": molecule_id,
            "trigger_calculation_id": calculation_id,
            "batch_size": pending_count,
        },
    )
    return TrainingBatchResult(
        triggered=True,
        pending_count=0,
        completed_count_at_trigger=pending_count,
        training_task_id=training_task_id,
        threshold=threshold,
    )
