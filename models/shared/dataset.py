"""MinIO-backed dataset of DFT artifacts for graph2mat-based models.

Reads the ``.npz`` files written by the DFT worker. Each file must contain
``density_matrix`` (n×n), ``elements`` (n_atoms,) and ``positions``
(n_atoms × 3, in Angstroms).
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np

from common.storage import ObjectStorage, get_storage

logger = logging.getLogger("models.dataset")


def _decode_str_array(arr: np.ndarray) -> list[str]:
    if arr.dtype.kind in ("U", "S", "O"):
        return [str(x) for x in arr.tolist()]
    raise TypeError(f"Unexpected dtype for str array: {arr.dtype}")


REQUIRED_NPZ_FIELDS = {"density_matrix", "elements", "positions"}


def load_dft_npz(raw: bytes) -> dict[str, Any]:
    """Decode a DFT artifact ``.npz`` into plain Python/numpy values."""
    data = np.load(io.BytesIO(raw), allow_pickle=False)
    keys = set(data.files)
    missing = REQUIRED_NPZ_FIELDS - keys
    if missing:
        raise ValueError(f"DFT npz missing fields: {sorted(missing)}")

    return {
        "density_matrix": np.asarray(data["density_matrix"], dtype=np.float64),
        "elements": _decode_str_array(data["elements"]),
        "positions": np.asarray(data["positions"], dtype=np.float64),
        "molecule_id": str(data["molecule_id"]) if "molecule_id" in keys else "",
        "calculation_id": str(data["calculation_id"]) if "calculation_id" in keys else "",
        "energy": float(data["energy"]) if "energy" in keys else float("nan"),
        "method": str(data["method"]) if "method" in keys else "",
        "basis": str(data["basis"]) if "basis" in keys else "",
    }


def _npz_has_required_fields(raw: bytes) -> bool:
    """Quick header-only check: does the npz expose all REQUIRED_NPZ_FIELDS?"""
    try:
        data = np.load(io.BytesIO(raw), allow_pickle=False)
    except Exception:
        return False
    return REQUIRED_NPZ_FIELDS.issubset(set(data.files))


def list_dft_artifact_keys(storage: ObjectStorage | None = None) -> list[str]:
    store = storage or get_storage()
    prefix = f"{store.settings.dft_artifacts_prefix}/"
    return [k for k in store.list_keys(prefix) if k.endswith(".npz")]


class MinioDftDataset:
    """Dataset of DFT records read from MinIO.

    Each item is a dict ``{config, density_matrix, elements, positions, ...}``
    where ``config`` is a graph2mat ``BasisConfiguration`` ready to be passed
    to ``TorchBasisMatrixData.new(config, data_processor=processor)``.

    The element-wise basis is built once from all elements seen across the
    dataset and cached in MinIO under ``models/_cache/``.
    """

    def __init__(
        self,
        *,
        basis_name: str = "sto-3g",
        max_samples: int | None = None,
        storage: ObjectStorage | None = None,
        artifact_keys: list[str] | None = None,
    ) -> None:
        self.storage = storage or get_storage()
        self.basis_name = basis_name
        raw_keys = (
            list(artifact_keys)
            if artifact_keys is not None
            else list_dft_artifact_keys(self.storage)
        )
        if not raw_keys:
            raise FileNotFoundError("No DFT artifacts found in MinIO for training")

        # Pre-scan and skip artifacts that don't expose the required fields
        # (e.g. objects produced by earlier versions of the DFT worker). One
        # malformed file must not block training on the rest.
        self.keys: list[str] = []
        self._records_cache: dict[int, dict[str, Any]] = {}
        skipped: list[str] = []
        for key in raw_keys:
            try:
                raw = self.storage.get_bytes(key)
                if not _npz_has_required_fields(raw):
                    skipped.append(key)
                    continue
                self._records_cache[len(self.keys)] = load_dft_npz(raw)
                self.keys.append(key)
            except Exception as exc:
                logger.warning("dataset_skip_unreadable key=%s err=%s", key, exc)
                skipped.append(key)

        if skipped:
            logger.warning(
                "dataset_skipped_artifacts count=%d kept=%d examples=%s",
                len(skipped), len(self.keys), skipped[:3],
            )
        if not self.keys:
            raise FileNotFoundError(
                f"No usable DFT artifacts found in MinIO "
                f"(scanned {len(raw_keys)}, all missing required fields "
                f"{sorted(REQUIRED_NPZ_FIELDS)})"
            )

        if max_samples is not None:
            self.keys = self.keys[:max_samples]
            self._records_cache = {
                i: rec for i, rec in self._records_cache.items() if i < max_samples
            }

        self._unique_basis = None  # populated lazily

    def __len__(self) -> int:
        return len(self.keys)

    def _load_record(self, idx: int) -> dict[str, Any]:
        if idx in self._records_cache:
            return self._records_cache[idx]
        raw = self.storage.get_bytes(self.keys[idx])
        record = load_dft_npz(raw)
        self._records_cache[idx] = record
        return record

    def _all_elements(self) -> list[str]:
        seen: set[str] = set()
        for i in range(len(self)):
            seen.update(self._load_record(i)["elements"])
        return sorted(seen)

    def unique_point_basis(self):
        """Return one ``PointBasis`` per element present in the dataset."""
        if self._unique_basis is None:
            from models.shared.basis import load_or_build_basis

            elements = self._all_elements()
            self._unique_basis = load_or_build_basis(
                elements,
                basis_name=self.basis_name,
                storage=self.storage,
            )
        return self._unique_basis

    def __getitem__(self, idx: int) -> dict[str, Any]:
        from graph2mat import BasisConfiguration

        record = self._load_record(idx)
        basis = self.unique_point_basis()
        config = BasisConfiguration(
            point_types=record["elements"],
            positions=np.asarray(record["positions"], dtype=np.float64),
            basis=basis,
            cell=np.eye(3) * 100.0,
            pbc=(False, False, False),
            matrix=record["density_matrix"],
        )
        return {
            "config": config,
            "density_matrix": record["density_matrix"],
            "elements": record["elements"],
            "positions": record["positions"],
            "molecule_id": record["molecule_id"],
            "calculation_id": record["calculation_id"],
        }
