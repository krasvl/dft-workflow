"""Tests for the model-agnostic dataset (no graph2mat required)."""

import io
from unittest.mock import MagicMock

import numpy as np
import pytest

from dft.mock import run_mock_dft
from models.shared.dataset import MinioDftDataset, load_dft_npz
from services.data_service.app.xyz import WATER
from workers.dft_worker.app.artifacts import dft_result_to_npz_bytes


def _make_storage_with_artifacts(npz_payloads: list[tuple[str, bytes]]) -> MagicMock:
    store = MagicMock()
    store.settings.dft_artifacts_prefix = "dft/artifacts"
    keys = [k for k, _ in npz_payloads]
    payload_map = dict(npz_payloads)
    store.list_keys.return_value = keys
    store.get_bytes.side_effect = lambda key: payload_map[key]
    return store


def test_load_dft_npz_round_trip() -> None:
    result = run_mock_dft(WATER)
    raw = dft_result_to_npz_bytes(result, molecule_id="mol_a", calculation_id="dft_a")
    record = load_dft_npz(raw)

    assert record["elements"] == ["O", "H", "H"]
    assert record["positions"].shape == (3, 3)
    assert record["density_matrix"].shape == result.density_matrix.shape
    assert record["molecule_id"] == "mol_a"


def test_load_dft_npz_rejects_old_format() -> None:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, density_matrix=np.eye(4))
    with pytest.raises(ValueError, match="missing fields"):
        load_dft_npz(buffer.getvalue())


def test_minio_dataset_yields_records_without_graph2mat() -> None:
    r1 = run_mock_dft(WATER)
    r2 = run_mock_dft(WATER + b"\n")
    raw1 = dft_result_to_npz_bytes(r1, molecule_id="mol_a", calculation_id="dft_a")
    raw2 = dft_result_to_npz_bytes(r2, molecule_id="mol_b", calculation_id="dft_b")

    storage = _make_storage_with_artifacts(
        [
            ("dft/artifacts/mol_a/dft_a.npz", raw1),
            ("dft/artifacts/mol_b/dft_b.npz", raw2),
            ("dft/artifacts/mol_b/dft_b.json", b"not npz"),
        ]
    )
    ds = MinioDftDataset(storage=storage)
    assert len(ds) == 2
    assert sorted(ds.keys) == [
        "dft/artifacts/mol_a/dft_a.npz",
        "dft/artifacts/mol_b/dft_b.npz",
    ]


def test_minio_dataset_raises_when_empty() -> None:
    storage = _make_storage_with_artifacts([])
    with pytest.raises(FileNotFoundError):
        MinioDftDataset(storage=storage)
