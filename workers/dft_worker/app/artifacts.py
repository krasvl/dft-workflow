"""Serialize DFT results to .npz bytes."""

from __future__ import annotations

import io

import numpy as np

from dft.result import DftResult


def dft_result_to_npz_bytes(
    result: DftResult,
    *,
    molecule_id: str,
    calculation_id: str,
) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        density_matrix=result.density_matrix,
        elements=np.asarray(result.elements),
        positions=np.asarray(result.positions, dtype=np.float64),
        molecule_id=np.array(molecule_id),
        calculation_id=np.array(calculation_id),
        energy=np.array(result.energy),
        method=np.array(result.method),
        basis=np.array(result.basis),
    )
    return buffer.getvalue()
