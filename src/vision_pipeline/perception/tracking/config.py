"""Configuration and errors shared by temporal-association implementations."""

from __future__ import annotations

from dataclasses import dataclass


class TrackingError(RuntimeError):
    """Temporal inputs are invalid or cannot be associated safely."""


@dataclass(frozen=True, slots=True)
class AppearanceTrackerConfig:
    appearance_weight: float = 0.7
    minimum_cosine_similarity: float = 0.65
    minimum_iou: float = 0.01
    minimum_score: float = 0.45
    descriptor_momentum: float = 0.8
    max_missed_frames: int = 8
    class_aware: bool = True

    def __post_init__(self) -> None:
        for name in (
            "appearance_weight",
            "minimum_cosine_similarity",
            "minimum_iou",
            "minimum_score",
            "descriptor_momentum",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number")
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between zero and one")
        if (
            isinstance(self.max_missed_frames, bool)
            or not isinstance(self.max_missed_frames, int)
            or self.max_missed_frames < 0
        ):
            raise ValueError("max_missed_frames must be a non-negative integer")
        if not isinstance(self.class_aware, bool):
            raise TypeError("class_aware must be a boolean")


__all__ = ["AppearanceTrackerConfig", "TrackingError"]
