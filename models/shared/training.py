"""Generic density-matrix training loop (MSE on graph2mat node/edge labels).

Architecture-specific code only has to supply a ``torch.nn.Module`` that
accepts a ``TorchBasisMatrixData`` batch and returns
``{"node_labels", "edge_labels"}``. The loop here owns the optimiser,
batching, loss computation and "best so far" checkpointing.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger("models.training")


@dataclass
class TrainingConfig:
    epochs: int = 1
    learning_rate: float = 1e-3
    batch_size: int = 1
    max_samples: int | None = None
    log_every: int = 50
    device: str = "cpu"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingMetrics:
    epochs: int
    samples: int
    steps: int
    train_loss: float
    best_loss: float
    wall_time_sec: float
    losses: list[float] = field(default_factory=list)


def _to_torch_batch(items, processor):
    """Pack dataset dicts into a single ``Batch`` of ``TorchBasisMatrixData``."""
    from graph2mat.bindings.torch import TorchBasisMatrixData
    from torch_geometric.data import Batch

    datas = [
        TorchBasisMatrixData.new(item["config"], data_processor=processor)
        for item in items
    ]
    return Batch.from_data_list(datas)


def _iter_batches(dataset, batch_size: int):
    buf: list[Any] = []
    for i in range(len(dataset)):
        buf.append(dataset[i])
        if len(buf) == batch_size:
            yield buf
            buf = []
    if buf:
        yield buf


def train_density_model(
    *,
    model,
    processor,
    dataset,
    config: TrainingConfig,
    extra_state: dict[str, Any] | None = None,
    loss_fn: Callable[..., Any] | None = None,
) -> tuple[dict[str, Any], TrainingMetrics]:
    """Train ``model`` on density-matrix labels with MSE.

    Returns ``(state, metrics)`` where ``state`` is a checkpoint dict ready for
    :func:`models.shared.serialization.state_to_bytes` and ``metrics``
    summarises the training run.
    """
    import torch
    from graph2mat import metrics as g2m_metrics

    if loss_fn is None:
        loss_fn = g2m_metrics.elementwise_mse

    device = torch.device(config.device)
    model = model.to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    losses: list[float] = []
    best_loss = float("inf")
    best_state: dict[str, Any] | None = None
    samples_seen = 0
    step = 0

    started = time.perf_counter()
    for epoch in range(config.epochs):
        for batch_items in _iter_batches(dataset, config.batch_size):
            data = _to_torch_batch(batch_items, processor).to(device)
            optimizer.zero_grad()

            preds = model(data)
            loss, _info = loss_fn(
                nodes_pred=preds.get("node_labels"),
                nodes_ref=data.point_labels,
                edges_pred=preds.get("edge_labels"),
                edges_ref=data.edge_labels,
            )
            loss.backward()
            optimizer.step()

            loss_value = float(loss.detach().cpu().item())
            losses.append(loss_value)
            samples_seen += len(batch_items)
            step += 1

            if loss_value < best_loss:
                best_loss = loss_value
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            if step % config.log_every == 0:
                logger.info(
                    "train step=%d epoch=%d loss=%.4e best=%.4e",
                    step, epoch, loss_value, best_loss,
                )

    wall_time = time.perf_counter() - started
    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    checkpoint: dict[str, Any] = {
        "model_state_dict": best_state,
        "training_loss": float(losses[-1]) if losses else float("nan"),
        "best_loss": float(best_loss),
        "epochs": config.epochs,
        "steps": step,
        "samples": samples_seen,
    }
    if extra_state:
        checkpoint.update(extra_state)

    metrics = TrainingMetrics(
        epochs=config.epochs,
        samples=samples_seen,
        steps=step,
        train_loss=float(losses[-1]) if losses else float("nan"),
        best_loss=float(best_loss),
        wall_time_sec=round(wall_time, 4),
        losses=losses,
    )
    return checkpoint, metrics
