"""ML models for electron density prediction.

Layout::

    models/shared/  — model-agnostic infrastructure (basis, dataset, training loop)
    models/mace/    — MACE + graph2mat architecture (training and inference)

Heavy dependencies (``graph2mat``, ``mace-torch``, ``e3nn``, ``torch_geometric``,
``pyscf``) are imported lazily inside concrete model modules so the rest of the
codebase stays importable without the ML stack installed.
"""
