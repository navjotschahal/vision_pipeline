"""Contracts for appearance descriptors attached to detected image regions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

from vision_pipeline.perception.contracts import DetectionBatch, FeatureBatch

RegionTensorT = TypeVar("RegionTensorT")


@dataclass(frozen=True, slots=True)
class FeatureGridRegion:
    """Half-open patch-grid bounds used to pool one detection descriptor."""

    detection_index: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def __post_init__(self) -> None:
        for name in ("detection_index", "x_min", "y_min", "x_max", "y_max"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.detection_index < 0 or self.x_min < 0 or self.y_min < 0:
            raise ValueError("region indices and minimum coordinates must be non-negative")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("region maximum coordinates must exceed minimum coordinates")


@dataclass(frozen=True, slots=True)
class RegionDescriptorBatch(Generic[RegionTensorT]):
    """One normalized appearance descriptor for every detection in a frame."""

    detections: DetectionBatch
    features: FeatureBatch[RegionTensorT]
    descriptors: RegionTensorT
    regions: tuple[FeatureGridRegion, ...]
    pooling: str

    def __post_init__(self) -> None:
        if self.detections.source != self.features.source:
            raise ValueError("detections and features must derive from the same source sample")
        if len(self.regions) != len(self.detections.detections):
            raise ValueError("every detection must have one feature-grid region")
        if not isinstance(self.pooling, str) or not self.pooling.strip():
            raise ValueError("pooling must be a non-empty string")


__all__ = ["FeatureGridRegion", "RegionDescriptorBatch"]
