"""MACE inference: forward pass producing a predicted density matrix.

Rebuilds the same network that was assembled at training time
(see :mod:`models.mace.train`) from a checkpoint dict, runs a single
``MatrixMACE`` forward, and reassembles the density matrix with the
graph2mat data processor.

Checkpoint layout (produced by :mod:`models.shared.training` +
:mod:`models.mace.train`)::

    {
        "model_state_dict": dict[str, torch.Tensor],
        "hparams":          dict,               # MaceHyperParams.to_dict()
        "elements":         list[str],          # element symbols from training set
        "basis_name":       str,                # e.g. "sto-3g"
        "architecture":     "mace",
        ...                                     # plus loss/epoch bookkeeping
    }

The element list is the *training* set, not per-molecule — it pins the basis
table the network expects. Requests for molecules containing an unseen element
are rejected up front with a clear error.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from common.storage import ObjectStorage, get_storage
from common.xyz import validate_xyz
from models.mace.config import MaceHyperParams
from models.mace.model import build_mace_model
from models.shared.basis import load_or_build_basis

logger = logging.getLogger("models.mace.inference")


def _parse_xyz(xyz_content: bytes) -> tuple[list[str], np.ndarray]:
    """Return (elements, positions[N,3]) from XYZ bytes (Angstrom)."""
    validate_xyz(xyz_content)
    lines = xyz_content.decode("utf-8").strip().splitlines()
    n = int(lines[0].split()[0])
    coord_lines = [ln.strip() for ln in lines[2:] if ln.strip()][:n]
    elements: list[str] = []
    positions = np.zeros((n, 3), dtype=np.float64)
    for i, line in enumerate(coord_lines):
        parts = line.split()
        elements.append(parts[0])
        positions[i] = [float(parts[1]), float(parts[2]), float(parts[3])]
    return elements, positions


class MaceInferenceModel:
    """Trained MACE + graph2mat model wrapped for density-matrix inference.

    Heavy ML imports (``torch``, ``e3nn``, ``mace-torch``, ``graph2mat``,
    ``torch_geometric``) are deferred to ``__init__`` / ``predict`` so callers
    that don't actually run MACE (e.g. tests against mock checkpoints) avoid
    the import cost.
    """

    def __init__(
        self,
        checkpoint: dict[str, Any],
        *,
        storage: ObjectStorage | None = None,
    ) -> None:
        import torch
        from graph2mat import BasisTableWithEdges, MatrixDataProcessor

        store = storage or get_storage()

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise ValueError("Checkpoint missing 'model_state_dict' (real MACE inference needs it)")

        elements = checkpoint.get("elements")
        if not isinstance(elements, (list, tuple)) or not elements:
            raise ValueError("Checkpoint missing 'elements' list (basis table cannot be rebuilt)")

        basis_name = str(checkpoint.get("basis_name", "sto-3g"))
        hparams_dict = dict(checkpoint.get("hparams") or {})
        hparams = MaceHyperParams(**hparams_dict)

        self.elements_seen: list[str] = sorted({str(e) for e in elements})
        self.basis_name = basis_name
        self.hparams = hparams

        self.unique_basis = load_or_build_basis(
            self.elements_seen,
            basis_name=basis_name,
            storage=store,
        )
        self.table = BasisTableWithEdges(self.unique_basis)
        self.processor = MatrixDataProcessor(
            basis_table=self.table,
            symmetric_matrix=True,
            sub_point_matrix=False,
        )

        self.model = build_mace_model(self.unique_basis, self.table, hparams=hparams)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            logger.warning(
                "mace_load_state_dict missing=%d unexpected=%d",
                len(missing), len(unexpected),
            )
        self.model.eval()
        self._torch = torch

        logger.info(
            "mace_inference_ready elements=%s basis=%s",
            self.elements_seen, self.basis_name,
        )

    def predict(self, xyz_content: bytes) -> np.ndarray:
        from graph2mat import BasisConfiguration
        from graph2mat.bindings.torch import TorchBasisMatrixData
        from torch_geometric.data import Batch

        elements, positions = _parse_xyz(xyz_content)
        unknown = sorted(set(elements) - set(self.elements_seen))
        if unknown:
            raise ValueError(
                f"Model was not trained on elements {unknown}; "
                f"known elements: {self.elements_seen}"
            )

        config = BasisConfiguration(
            point_types=elements,
            positions=positions,
            basis=self.unique_basis,
            cell=np.eye(3) * 100.0,
            pbc=(False, False, False),
        )
        data = TorchBasisMatrixData.new(config, data_processor=self.processor)
        # MACE expects a torch_geometric Batch (with a ``batch`` index tensor)
        # even for a single molecule. Mirrors what training does.
        batch = Batch.from_data_list([data])

        with self._torch.no_grad():
            preds = self.model(batch)

        matrices = self.processor.matrix_from_data(
            batch,
            predictions={
                "node_labels": preds.get("node_labels"),
                "edge_labels": preds.get("edge_labels"),
            },
            out_format="numpy",
        )
        matrix = matrices[0] if isinstance(matrices, tuple) else matrices
        return np.asarray(matrix, dtype=np.float64)
