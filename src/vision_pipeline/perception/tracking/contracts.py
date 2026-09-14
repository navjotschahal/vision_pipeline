"""Typed temporal-association results."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, TypeVar

from vision_pipeline.perception.regions import RegionDescriptorBatch

TrackingTensorT = TypeVar("TrackingTensorT")


@dataclass(frozen=True, slots=True)
class TrackedDetection2D:
    """Track identity assigned to one current-frame detection."""

    detection_index: int
    track_id: int
    age_frames: int
    hits: int
    association_score: float | None

    def __post_init__(self) -> None:
        for name in ("detection_index", "track_id", "age_frames", "hits"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.detection_index < 0:
            raise ValueError("detection_index must be non-negative")
        if self.track_id <= 0 or self.age_frames <= 0 or self.hits <= 0:
            raise ValueError("track_id, age_frames, and hits must be positive")
        if self.association_score is not None:
            if not math.isfinite(self.association_score):
                raise ValueError("association_score must be finite")
            if not 0 <= self.association_score <= 1:
                raise ValueError("association_score must be between zero and one")


@dataclass(frozen=True, slots=True)
class TrackingBatch(Generic[TrackingTensorT]):
    """Track identities for every detection observed in the current frame."""

    regions: RegionDescriptorBatch[TrackingTensorT]
    tracked_detections: tuple[TrackedDetection2D, ...]
    active_track_count: int
    association: str

    def __post_init__(self) -> None:
        detection_count = len(self.regions.detections.detections)
        if len(self.tracked_detections) != detection_count:
            raise ValueError("every detection must receive exactly one track result")
        if tuple(item.detection_index for item in self.tracked_detections) != tuple(
            range(detection_count)
        ):
            raise ValueError("tracked detections must preserve detection order")
        if self.active_track_count < detection_count:
            raise ValueError("active track count cannot be below current detection count")
        if not self.association.strip():
            raise ValueError("association must be non-empty")


__all__ = ["TrackedDetection2D", "TrackingBatch"]
