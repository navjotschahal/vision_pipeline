"""Temporal association of per-detection appearance descriptors."""

from .association import GreedyAppearanceTracker
from .config import AppearanceTrackerConfig, TrackingError
from .contracts import TrackedDetection2D, TrackingBatch

__all__ = [
    "AppearanceTrackerConfig",
    "GreedyAppearanceTracker",
    "TrackedDetection2D",
    "TrackingBatch",
    "TrackingError",
]
