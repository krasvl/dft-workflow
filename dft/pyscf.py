"""Real DFT calculation via PySCF (RKS/UKS).

Supports an optional ``dm0`` initial guess so the function can be used both
for cold-start ground-truth runs (training data) and hybrid ML→SCF refinement.
"""

from __future__ import annotations

import time

import numpy as np

from common.xyz import validate_xyz
from dft.result import DftResult


class ScfNotConvergedError(RuntimeError):
    pass


def xyz_to_pyscf_atom(xyz_content: bytes) -> str:
    """Build PySCF ``atom`` string from XYZ bytes (Angstrom)."""
    validate_xyz(xyz_content)
    lines = xyz_content.decode("utf-8").strip().splitlines()
    n = int(lines[0].split()[0])
    coord_lines = [ln.strip() for ln in lines[2:] if ln.strip()][:n]
    parts = []
    for ln in coord_lines:
        s = ln.split()
        parts.append(f"{s[0]} {s[1]} {s[2]} {s[3]}")
    return "; ".join(parts)


def _build_mean_field(mol, method: str):
    from pyscf import dft as pyscf_dft

    key = method.lower().removeprefix("mock-")
    if key in ("uks", "u-ks"):
        return pyscf_dft.UKS(mol), "UKS"
    if key in ("rks", "r-ks", "dft", "dft-rks", "", "mock"):
        return pyscf_dft.RKS(mol), "RKS"
    raise ValueError(f"Unsupported DFT method for PySCF: {method}")


def run_pyscf_dft(
    xyz_content: bytes,
    *,
    method: str = "rks",
    basis: str = "sto-3g",
    max_cycle: int = 100,
    conv_tol: float = 1e-9,
    dm0: np.ndarray | None = None,
) -> DftResult:
    """Run RKS/UKS SCF; if ``dm0`` is given it's used as the initial guess."""
    from pyscf import gto

    atom = xyz_to_pyscf_atom(xyz_content)
    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)
    mf, method_label = _build_mean_field(mol, method)
    mf.verbose = 0
    mf.max_cycle = max_cycle
    mf.conv_tol = conv_tol

    started = time.perf_counter()
    if dm0 is not None:
        energy = float(mf.kernel(dm0=np.asarray(dm0, dtype=np.float64)))
    else:
        energy = float(mf.kernel())
    wall_time = time.perf_counter() - started

    if not mf.converged:
        raise ScfNotConvergedError(
            f"SCF did not converge after {mf.cycles} cycles (energy={energy})"
        )

    dm = np.asarray(mf.make_rdm1(), dtype=np.float64)
    if dm.ndim != 2:
        # UKS returns (2, n, n); collapse to total density α + β.
        if dm.ndim == 3 and dm.shape[0] == 2:
            dm = dm[0] + dm[1]
        else:
            raise ValueError(f"Unexpected density matrix shape {dm.shape}")

    elements = [mol.atom_symbol(i) for i in range(mol.natm)]
    positions = np.asarray(mol.atom_coords(unit="Angstrom"), dtype=np.float64)

    return DftResult(
        density_matrix=dm,
        wall_time_sec=round(wall_time, 4),
        scf_iterations=int(mf.cycles),
        method=method_label,
        basis=basis,
        energy=energy,
        elements=elements,
        positions=positions,
    )
