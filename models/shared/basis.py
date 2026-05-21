"""Build a graph2mat ``PointBasis`` per chemical element, cached in MinIO.

The basis description is derived from PySCF (so the AO layout matches what the
DFT worker produces) and stored as a JSON ``{element: basis_str}`` map under
``models/_cache/``. Subsequent calls reuse the cache.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from common.storage import ObjectStorage, get_storage

logger = logging.getLogger("models.basis")

GROUND_STATE_SPIN: dict[str, int] = {
    "H": 1, "He": 0, "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3, "O": 2, "F": 1, "Ne": 0,
    "Na": 1, "Mg": 0, "Al": 1, "Si": 2, "P": 3, "S": 2, "Cl": 1, "Ar": 0,
    "K": 1, "Ca": 0, "Br": 1, "I": 1,
}

_ELEMENT_Z: dict[str, int] = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18,
    "K": 19, "Ca": 20, "Br": 35, "I": 53,
}


def _guess_spin(symbol: str) -> int:
    if symbol in GROUND_STATE_SPIN:
        return GROUND_STATE_SPIN[symbol]
    z = _ELEMENT_Z.get(symbol)
    if z is None:
        return 0
    return 1 if (z % 2 == 1) else 0


def basis_cache_key(basis_name: str) -> str:
    return f"models/_cache/basis_{basis_name}.json"


def _build_basis_str_from_pyscf(element: str, basis_name: str) -> str:
    from pyscf import gto

    mol = gto.M(
        atom=f"{element} 0 0 0",
        basis=basis_name,
        unit="Angstrom",
        cart=False,
        charge=0,
        spin=_guess_spin(element),
        verbose=0,
    )
    mol.build()
    counts: dict[int, int] = {}
    for ib in range(mol.nbas):
        l = mol.bas_angular(ib)
        counts[l] = counts.get(l, 0) + mol.bas_nctr(ib)
    return " + ".join(
        f"{counts[l]}x{l}{'e' if l % 2 == 0 else 'o'}" for l in sorted(counts)
    )


def load_or_build_basis(
    elements: list[str],
    *,
    basis_name: str = "sto-3g",
    storage: ObjectStorage | None = None,
):
    """Return one ``PointBasis`` per requested element.

    Missing entries are computed via PySCF and appended to the MinIO cache so
    subsequent calls are zero-compute. Existing entries are reused verbatim.
    """
    from graph2mat import PointBasis

    store = storage or get_storage()
    cache_key = basis_cache_key(basis_name)

    cache: dict[str, str] = {}
    if store.object_exists(cache_key):
        cache = {str(k): str(v) for k, v in store.get_json(cache_key).items()}

    missing = sorted({e for e in elements if e not in cache})
    if missing:
        logger.info("basis_cache_miss elements=%s basis=%s", missing, basis_name)
        for el in missing:
            cache[el] = _build_basis_str_from_pyscf(el, basis_name)
        store.put_json(cache_key, cache)

    return [
        PointBasis(el, R=6.0, basis=cache[el], basis_convention="spherical")
        for el in sorted(set(elements))
    ]


@lru_cache(maxsize=8)
def _cached_basis(basis_name: str, elements_tuple: tuple[str, ...]):
    return load_or_build_basis(list(elements_tuple), basis_name=basis_name)


def get_basis_for(elements: list[str], *, basis_name: str = "sto-3g"):
    """Cached wrapper around ``load_or_build_basis`` for repeated calls."""
    return _cached_basis(basis_name, tuple(sorted(set(elements))))
