"""DFT calculation result — shared dataclass for all DFT backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DftResult:
    density_matrix: np.ndarray
    wall_time_sec: float
    scf_iterations: int
    method: str
    basis: str
    energy: float
    elements: list[str]
    positions: np.ndarray  # Angstroms, shape (n_atoms, 3)
