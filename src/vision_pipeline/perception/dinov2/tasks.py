"""Typed outputs for learned dense DINOv2 task heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from vision_pipeline.contracts import ComputePlacement, MemoryPlacement, SampleHeader

TensorT = TypeVar("TensorT")


@dataclass(frozen=True, slots=True)
class DenseTaskTiming:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    wall_ms: float

    def __post_init__(self) -> None:
        for name in ("preprocess_ms", "inference_ms", "postprocess_ms", "wall_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be non-negative and finite")


@dataclass(frozen=True, slots=True)
class DepthPrediction(Generic[TensorT]):
    """Metric monocular depth, in metres, aligned to one source image."""

    source: SampleHeader
    depth_metres: TensorT
    min_depth_metres: float
    max_depth_metres: float
    model_id: str
    produced_on: ComputePlacement
    memory: MemoryPlacement
    timing: DenseTaskTiming


@dataclass(frozen=True, slots=True)
class SemanticSegmentation(Generic[TensorT]):
    """Per-pixel semantic class IDs and winning-class confidence."""

    source: SampleHeader
    class_ids: TensorT
    confidence: TensorT
    class_names: tuple[str, ...]
    model_id: str
    produced_on: ComputePlacement
    memory: MemoryPlacement
    timing: DenseTaskTiming


__all__ = ["DenseTaskTiming", "DepthPrediction", "SemanticSegmentation"]
