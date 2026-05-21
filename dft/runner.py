"""Dispatch DFT calculation to the configured backend (mock | pyscf)."""

from __future__ import annotations

from common.settings import get_settings
from dft.mock import run_mock_dft
from dft.result import DftResult


def _normalise_for_pyscf(value: str | None, default: str) -> str:
    if not value:
        return default
    v = str(value)
    if v.startswith("mock") or v == "mock":
        return default
    return v


def run_dft(
    xyz_content: bytes,
    *,
    method: str | None = None,
    basis: str | None = None,
) -> DftResult:
    """Dispatch DFT calculation to mock or PySCF based on ``DFT_ENGINE``."""
    settings = get_settings()
    engine = (settings.dft_engine or "mock").strip().lower()

    if engine == "mock":
        return run_mock_dft(
            xyz_content,
            method=method or "mock-rks",
            basis=basis or "mock-basis",
        )

    if engine == "pyscf":
        from dft.pyscf import run_pyscf_dft

        return run_pyscf_dft(
            xyz_content,
            method=_normalise_for_pyscf(method, settings.dft_default_method),
            basis=_normalise_for_pyscf(basis, settings.dft_default_basis),
            max_cycle=settings.dft_max_scf_cycles,
            conv_tol=settings.dft_scf_conv_tol,
        )

    raise ValueError(f"Unknown DFT_ENGINE={engine!r} (use 'mock' or 'pyscf')")
