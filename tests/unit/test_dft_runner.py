from unittest.mock import patch

import pytest

from common.xyz import WATER
from dft import run_dft


def test_run_dft_dispatches_to_mock_by_default() -> None:
    with patch("dft.runner.get_settings") as gs:
        gs.return_value.dft_engine = "mock"
        result = run_dft(WATER)

    assert result.method == "mock-rks"
    assert result.basis == "mock-basis"
    assert result.elements == ["O", "H", "H"]
    assert result.density_matrix.shape[0] == result.density_matrix.shape[1]


def test_run_dft_unknown_engine_raises() -> None:
    with patch("dft.runner.get_settings") as gs:
        gs.return_value.dft_engine = "bogus"
        with pytest.raises(ValueError, match="Unknown DFT_ENGINE"):
            run_dft(WATER)


def test_run_dft_pyscf_engine_normalises_mock_method() -> None:
    pyscf = pytest.importorskip("pyscf")  # noqa: F841
    with patch("dft.runner.get_settings") as gs:
        gs.return_value.dft_engine = "pyscf"
        gs.return_value.dft_default_method = "rks"
        gs.return_value.dft_default_basis = "sto-3g"
        gs.return_value.dft_max_scf_cycles = 50
        gs.return_value.dft_scf_conv_tol = 1e-8

        result = run_dft(WATER, method="mock-rks", basis="mock-basis")

    assert result.method == "RKS"
    assert result.basis == "sto-3g"
    assert result.scf_iterations >= 1
    assert result.energy < 0
