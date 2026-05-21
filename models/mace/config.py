"""MACE hyper-parameters used by the density predictor."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MaceHyperParams:
    r_max: float = 10.0
    num_bessel: int = 10
    num_polynomial_cutoff: int = 10
    max_ell: int = 2
    num_interactions: int = 3
    hidden_irreps: str = "1x0e + 1x1o"
    mlp_irreps: str = "2x0e"
    edge_hidden_irreps: str = "10x0e + 10x1o + 10x2e"
    avg_num_neighbors: float = 2.0
    correlation: int = 2
    symmetric: bool = True
    readout_per_interaction: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
