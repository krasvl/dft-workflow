"""DFT primitives shared by the DFT worker and the density service.

Public re-exports:

* :class:`DftResult`    — backend-agnostic calculation result
* :func:`run_dft`       — dispatch helper that selects ``mock`` or ``pyscf``
                          based on :class:`Settings.dft_engine`
"""

from dft.result import DftResult
from dft.runner import run_dft

__all__ = ["DftResult", "run_dft"]
