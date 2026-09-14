"""Small PyTorch runtime helpers shared by the DINOv2 adapters."""

from __future__ import annotations

import importlib
from contextlib import AbstractContextManager
from typing import Any

import numpy as np
import torch

from vision_pipeline.contracts import (
    ComputeKind,
    ComputePlacement,
    MemoryKind,
    MemoryPlacement,
)

from . import DinoV2Error


def normalize_device(name: str) -> torch.device:
    normalized = f"cuda:{name}" if name.isdigit() else name
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DinoV2Error(f"CUDA device {name!r} was requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise DinoV2Error("MPS was requested but is unavailable")
    return device


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def placement(device: torch.device) -> tuple[ComputePlacement, MemoryPlacement]:
    if device.type == "cpu":
        return (
            ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
            MemoryPlacement(MemoryKind.HOST),
        )
    if device.type == "mps":
        return (
            ComputePlacement(ComputeKind.HOST_GPU, "apple-mps"),
            MemoryPlacement(MemoryKind.DEVICE_LOCAL, "apple-mps"),
        )
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        device_id = f"cuda:{index}"
        return (
            ComputePlacement(ComputeKind.HOST_GPU, device_id),
            MemoryPlacement(MemoryKind.DEVICE_LOCAL, device_id),
        )
    return (
        ComputePlacement(ComputeKind.UNKNOWN, str(device)),
        MemoryPlacement(MemoryKind.DEVICE_LOCAL, str(device)),
    )


def official_checkpoint_safe_globals() -> AbstractContextManager[Any]:
    """Allow only NumPy scalar metadata used by Meta's official checkpoints.

    PyTorch 2.6 and newer defaults to ``weights_only=True``. Older DINOv2 task-head
    checkpoints contain NumPy scalar metadata, so loading them safely requires these
    narrow allow-list entries. This does not disable weights-only loading.
    """

    numpy_scalar: Any = importlib.import_module("numpy._core.multiarray").scalar
    safe_globals: list[Any] = [
        (numpy_scalar, "numpy.core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
        (type(np.dtype(np.float32)), "numpy.dtype[float32]"),
    ]
    return torch.serialization.safe_globals(safe_globals)


__all__ = [
    "normalize_device",
    "official_checkpoint_safe_globals",
    "placement",
    "synchronize",
]
