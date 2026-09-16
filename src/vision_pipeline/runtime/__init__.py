"""Scheduling and cross-process transport primitives for live perception pipelines."""

from .box_view_channel import (
    CATEGORY_ABOVE_PLANE,
    CATEGORY_CLUSTER,
    CATEGORY_OUTSIDE_WORKSPACE,
    CATEGORY_PLANE,
    CHANNEL_NAME,
    BoxViewFrame,
    BoxViewGeometry,
    BoxViewReader,
    BoxViewWriter,
    box_yaw_radians,
)
from .latest_frame import (
    LatestFrameDetectionPipeline,
    PipelineClosedError,
    PipelineStats,
    PipelineWorkerError,
    ProcessedFrame,
)
from .shm_seqlock import SeqlockReader, SeqlockTornReadError, SeqlockWriter

__all__ = [
    "CATEGORY_ABOVE_PLANE",
    "CATEGORY_CLUSTER",
    "CATEGORY_OUTSIDE_WORKSPACE",
    "CATEGORY_PLANE",
    "CHANNEL_NAME",
    "BoxViewFrame",
    "BoxViewGeometry",
    "BoxViewReader",
    "BoxViewWriter",
    "LatestFrameDetectionPipeline",
    "PipelineClosedError",
    "PipelineStats",
    "PipelineWorkerError",
    "ProcessedFrame",
    "SeqlockReader",
    "SeqlockTornReadError",
    "SeqlockWriter",
    "box_yaw_radians",
]
