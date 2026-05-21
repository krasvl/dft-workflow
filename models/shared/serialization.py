"""Torch checkpoint (de)serialisation helpers shared by all architectures."""

from __future__ import annotations

import io
from typing import Any


def state_to_bytes(state: dict[str, Any]) -> bytes:
    """Serialise a checkpoint dict with ``torch.save`` and return the bytes."""
    import torch

    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def bytes_to_state(raw: bytes) -> dict[str, Any]:
    """Inverse of :func:`state_to_bytes`. Raises if the payload is not a dict."""
    import torch

    obj = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint must be a dict")
    return obj
