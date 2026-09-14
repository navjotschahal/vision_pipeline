"""Stable semantic contracts for model-independent perception capabilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from vision_pipeline.contracts import ComputePlacement, MemoryPlacement, SampleHeader, TimePoint
from vision_pipeline.image import ImageFrame

FeatureTensorT = TypeVar("FeatureTensorT")


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class BoundingBox2D:
    """Axis-aligned pixel coordinates in the source image: left, top, right, bottom."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            _require_finite(getattr(self, name), name)
        if self.x_min < 0 or self.y_min < 0:
            raise ValueError("box minimum coordinates must be non-negative")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError("box maximum coordinates must exceed minimum coordinates")


@dataclass(frozen=True, slots=True)
class Detection2D:
    box: BoundingBox2D
    class_id: int
    label: str
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.box, BoundingBox2D):
            raise TypeError("box must be a BoundingBox2D")
        if isinstance(self.class_id, bool) or not isinstance(self.class_id, int):
            raise TypeError("class_id must be an integer")
        if self.class_id < 0:
            raise ValueError("class_id must be non-negative")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        _require_finite(self.confidence, "confidence")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class DetectionTiming:
    """Backend and observed wall timings, all in milliseconds for one image."""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    wall_ms: float

    def __post_init__(self) -> None:
        for name in ("preprocess_ms", "inference_ms", "postprocess_ms", "wall_ms"):
            value = getattr(self, name)
            _require_finite(value, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """Detections derived from exactly one source image."""

    source: SampleHeader
    image_width: int
    image_height: int
    detections: tuple[Detection2D, ...]
    model_id: str
    implementation: str
    produced_on: ComputePlacement
    started_at: TimePoint
    finished_at: TimePoint
    timing: DetectionTiming

    def __post_init__(self) -> None:
        if not isinstance(self.source, SampleHeader):
            raise TypeError("source must be a SampleHeader")
        if self.image_width <= 0 or self.image_height <= 0:
            raise ValueError("source image dimensions must be positive")
        if not isinstance(self.detections, tuple) or not all(
            isinstance(item, Detection2D) for item in self.detections
        ):
            raise TypeError("detections must be a tuple of Detection2D values")
        if not self.model_id.strip() or not self.implementation.strip():
            raise ValueError("model_id and implementation must be non-empty")
        if not isinstance(self.produced_on, ComputePlacement):
            raise TypeError("produced_on must be a ComputePlacement")
        if not isinstance(self.started_at, TimePoint) or not isinstance(
            self.finished_at, TimePoint
        ):
            raise TypeError("started_at and finished_at must be TimePoint values")
        if self.started_at.clock != self.finished_at.clock:
            raise ValueError("processing timestamps must share a clock domain")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        if not isinstance(self.timing, DetectionTiming):
            raise TypeError("timing must be DetectionTiming")


class Detector(Protocol):
    """Replaceable synchronous detector boundary."""

    def detect(self, image: ImageFrame, source: SampleHeader) -> DetectionBatch:
        """Detect objects and retain the source measurement identity."""


@dataclass(frozen=True, slots=True)
class FeatureGeometry:
    """Mapping between a source image, model input, and dense feature grid."""

    source_width: int
    source_height: int
    input_width: int
    input_height: int
    content_left: int
    content_top: int
    content_width: int
    content_height: int
    patch_width: int
    patch_height: int
    grid_width: int
    grid_height: int

    def __post_init__(self) -> None:
        for name in (
            "source_width",
            "source_height",
            "input_width",
            "input_height",
            "content_width",
            "content_height",
            "patch_width",
            "patch_height",
            "grid_width",
            "grid_height",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("content_left", "content_top"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.content_left + self.content_width > self.input_width:
            raise ValueError("content exceeds model input width")
        if self.content_top + self.content_height > self.input_height:
            raise ValueError("content exceeds model input height")
        if self.grid_width * self.patch_width != self.input_width:
            raise ValueError("input width must equal grid width times patch width")
        if self.grid_height * self.patch_height != self.input_height:
            raise ValueError("input height must equal grid height times patch height")

    @property
    def scale_x(self) -> float:
        return self.content_width / self.source_width

    @property
    def scale_y(self) -> float:
        return self.content_height / self.source_height


@dataclass(frozen=True, slots=True)
class FeatureTiming:
    """Observed wall timings for feature extraction, in milliseconds."""

    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    wall_ms: float

    def __post_init__(self) -> None:
        for name in ("preprocess_ms", "inference_ms", "postprocess_ms", "wall_ms"):
            value = getattr(self, name)
            _require_finite(value, name)
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True, slots=True)
class FeatureBatch(Generic[FeatureTensorT]):
    """Global and dense features derived from exactly one source image.

    Tensor storage is generic so a backend can retain features in PyTorch, ONNX, or
    another runtime's device memory without forcing a host copy at this boundary.
    """

    source: SampleHeader
    global_features: FeatureTensorT
    spatial_features: FeatureTensorT
    extra_tokens: FeatureTensorT | None
    geometry: FeatureGeometry
    model_id: str
    implementation: str
    produced_on: ComputePlacement
    memory: MemoryPlacement
    started_at: TimePoint
    finished_at: TimePoint
    timing: FeatureTiming

    def __post_init__(self) -> None:
        if not isinstance(self.source, SampleHeader):
            raise TypeError("source must be a SampleHeader")
        if self.global_features is None or self.spatial_features is None:
            raise ValueError("global and spatial features must not be None")
        if not isinstance(self.geometry, FeatureGeometry):
            raise TypeError("geometry must be FeatureGeometry")
        if not self.model_id.strip() or not self.implementation.strip():
            raise ValueError("model_id and implementation must be non-empty")
        if not isinstance(self.produced_on, ComputePlacement):
            raise TypeError("produced_on must be a ComputePlacement")
        if not isinstance(self.memory, MemoryPlacement):
            raise TypeError("memory must be a MemoryPlacement")
        if not isinstance(self.started_at, TimePoint) or not isinstance(
            self.finished_at, TimePoint
        ):
            raise TypeError("started_at and finished_at must be TimePoint values")
        if self.started_at.clock != self.finished_at.clock:
            raise ValueError("processing timestamps must share a clock domain")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        if not isinstance(self.timing, FeatureTiming):
            raise TypeError("timing must be FeatureTiming")


class FeatureExtractor(Protocol[FeatureTensorT]):
    """Replaceable synchronous image-feature extraction boundary."""

    def extract(
        self, image: ImageFrame, source: SampleHeader
    ) -> FeatureBatch[FeatureTensorT]:
        """Extract features while retaining source identity and image geometry."""
