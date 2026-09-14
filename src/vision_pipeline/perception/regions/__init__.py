"""Region-level perception derived from detections and dense feature maps."""

from .contracts import FeatureGridRegion, RegionDescriptorBatch
from .pooling import pool_detection_regions

__all__ = ["FeatureGridRegion", "RegionDescriptorBatch", "pool_detection_regions"]
