"""Construct the MACE + MatrixMACE density predictor."""

from __future__ import annotations

from typing import Any

from models.mace.config import MaceHyperParams


def build_mace_model(unique_basis, table, *, hparams: MaceHyperParams | None = None) -> Any:
    """Build a ``MatrixMACE`` wrapping a ``MACE`` core for density-matrix prediction.

    Args:
        unique_basis: list of graph2mat ``PointBasis`` covering every chemical
            element the model should support (typically the union of elements
            seen across the training dataset).
        table: a ``graph2mat.BasisTableWithEdges`` built from ``unique_basis``.
        hparams: optional override of :class:`MaceHyperParams` defaults.
    """
    import torch
    from e3nn import o3
    from mace.modules import MACE, RealAgnosticResidualInteractionBlock
    from graph2mat.models import MatrixMACE

    hp = hparams or MaceHyperParams()
    n_elements = len(unique_basis)

    mace_core = MACE(
        r_max=hp.r_max,
        num_bessel=hp.num_bessel,
        num_polynomial_cutoff=hp.num_polynomial_cutoff,
        max_ell=hp.max_ell,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        num_interactions=hp.num_interactions,
        num_elements=n_elements,
        hidden_irreps=o3.Irreps(hp.hidden_irreps),
        MLP_irreps=o3.Irreps(hp.mlp_irreps),
        atomic_energies=torch.zeros(n_elements),
        avg_num_neighbors=hp.avg_num_neighbors,
        atomic_numbers=list(range(n_elements)),
        correlation=hp.correlation,
        gate=None,
    )

    return MatrixMACE(
        mace_core,
        unique_basis=table,
        readout_per_interaction=hp.readout_per_interaction,
        edge_hidden_irreps=o3.Irreps(hp.edge_hidden_irreps),
        symmetric=hp.symmetric,
    )


def count_params(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
