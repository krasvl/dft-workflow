"""MACE-based density predictor (graph2mat + mace-torch).

Side effect: registers ``slice`` as a safe global for ``torch.load`` so e3nn's
shipped ``constants.pt`` (pickled with the new ``weights_only`` default in
PyTorch >= 2.6) can be loaded by e3nn 0.4.4. Without this the very first import
inside ``mace.modules`` fails.
"""

from __future__ import annotations

import torch

try:
    torch.serialization.add_safe_globals([slice])
except (AttributeError, RuntimeError):
    pass
