"""Cross-check PySCF DFT integration against known reference values.

Skipped when pyscf is not installed (e.g. in lightweight CI). Runs locally
after `make env-sync-mace`.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pyscf")

from common.xyz import WATER  # noqa: E402
from dft.pyscf import run_pyscf_dft  # noqa: E402


def test_water_sto3g_rks_energy() -> None:
    """RKS/sto-3g energy for water sits in the well-known window ≈ -75 Ha."""
    result = run_pyscf_dft(WATER, method="rks", basis="sto-3g")

    assert result.method == "RKS"
    assert result.basis == "sto-3g"
    assert result.scf_iterations >= 1
    assert result.wall_time_sec > 0.0

    # H2O at sto-3g (LDA RKS) is ≈ -74.96 Ha; allow ±1.5 Ha for functional/geometry drift.
    assert -76.5 < result.energy < -73.5, f"energy out of window: {result.energy}"

    dm = result.density_matrix
    assert dm.shape == (7, 7)  # 5 (O) + 1 + 1 (H)
    assert abs(dm - dm.T).max() < 1e-8

    # In a non-orthogonal AO basis trace(D) ≤ N_electrons; for sto-3g the
    # ratio is ~0.9. We use a loose window (Ne = 10 for neutral water).
    assert 5.0 <= dm.trace() <= 10.05, f"trace out of window: {dm.trace()}"

    assert result.elements == ["O", "H", "H"]


def test_h2_sto3g_rks_energy() -> None:
    """H2 at 0.74 Å, sto-3g, RKS energy ≈ -1.117 Ha."""
    xyz = b"2\nh2\nH 0.0 0.0 0.0\nH 0.0 0.0 0.74\n"
    result = run_pyscf_dft(xyz, method="rks", basis="sto-3g")
    assert result.scf_iterations >= 1
    assert result.density_matrix.shape == (2, 2)
    # 2 electrons, trace bound in non-orthogonal basis: 0.5*Ne..Ne
    assert 1.0 <= result.density_matrix.trace() <= 2.05
    assert -1.5 < result.energy < -0.5, f"energy out of window: {result.energy}"
