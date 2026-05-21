"""Mock DFT calculation — used when DFT_ENGINE=mock."""

from __future__ import annotations

import time

import numpy as np

from dft.result import DftResult


def parse_atom_count(xyz_content: bytes) -> int:
    lines = xyz_content.decode("utf-8").strip().splitlines()
    if not lines:
        raise ValueError("Empty XYZ file")
    return int(lines[0].split()[0])


def parse_xyz_atoms(xyz_content: bytes) -> tuple[list[str], np.ndarray]:
    lines = xyz_content.decode("utf-8").strip().splitlines()
    n = int(lines[0].split()[0])
    coord_lines = [ln.strip() for ln in lines[2:] if ln.strip()][:n]
    elements: list[str] = []
    positions: list[list[float]] = []
    for ln in coord_lines:
        parts = ln.split()
        elements.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elements, np.asarray(positions, dtype=np.float64)


def run_mock_dft(
    xyz_content: bytes,
    *,
    method: str = "mock-rks",
    basis: str = "mock-basis",
) -> DftResult:
    """Produce a deterministic symmetric fake density matrix sized from atom count."""
    started = time.perf_counter()
    n_atoms = parse_atom_count(xyz_content)
    size = max(n_atoms * 2, 4)

    rng = np.random.default_rng(abs(hash(xyz_content[:64])) % (2**32))
    density = rng.random((size, size), dtype=np.float64)
    density = (density + density.T) / 2.0

    elements, positions = parse_xyz_atoms(xyz_content)

    wall_time = time.perf_counter() - started

    return DftResult(
        density_matrix=density,
        wall_time_sec=round(wall_time, 4),
        scf_iterations=1,
        method=method,
        basis=basis,
        energy=float(-1.0 * n_atoms),
        elements=elements,
        positions=positions,
    )
