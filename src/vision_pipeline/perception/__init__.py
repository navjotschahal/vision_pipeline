"""Model-independent perception results and replaceable model adapters."""

from .contracts import (
    BoundingBox2D,
    Detection2D,
    DetectionBatch,
    DetectionTiming,
    Detector,
    FeatureBatch,
    FeatureExtractor,
    FeatureGeometry,
    FeatureTiming,
)

__all__ = [
    "BoundingBox2D",
    "Detection2D",
    "DetectionBatch",
    "DetectionTiming",
    "Detector",
    "FeatureBatch",
    "FeatureExtractor",
    "FeatureGeometry",
    "FeatureTiming",
]
