import io

import numpy as np

from dft.mock import parse_atom_count, run_mock_dft
from services.data_service.app.xyz import WATER
from workers.dft_worker.app.artifacts import dft_result_to_npz_bytes


def test_parse_atom_count() -> None:
    assert parse_atom_count(WATER) == 3


def test_mock_dft_symmetric_matrix() -> None:
    result = run_mock_dft(WATER)
    assert result.density_matrix.shape[0] == result.density_matrix.shape[1]
    assert np.allclose(result.density_matrix, result.density_matrix.T)
    assert result.method == "mock-rks"
    assert result.scf_iterations == 1


def test_mock_dft_carries_atom_geometry() -> None:
    result = run_mock_dft(WATER)
    assert result.elements == ["O", "H", "H"]
    assert result.positions.shape == (3, 3)
    assert np.allclose(result.positions[0], [0.0, 0.0, 0.0])


def test_npz_roundtrip() -> None:
    result = run_mock_dft(WATER)
    raw = dft_result_to_npz_bytes(
        result,
        molecule_id="mol_test",
        calculation_id="dft_test",
    )
    loaded = np.load(io.BytesIO(raw))
    assert "density_matrix" in loaded.files
    assert loaded["density_matrix"].shape == result.density_matrix.shape
    assert list(loaded["elements"]) == ["O", "H", "H"]
    assert loaded["positions"].shape == (3, 3)
