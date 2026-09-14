"""Object-level 3D results exposed to prediction and control adapters."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from vision_pipeline.contracts import SampleHeader, TimePoint
from vision_pipeline.geometry.pointcloud import GeometryKind
from vision_pipeline.geometry.spatial import OrientedBox3D, Pose3D, Twist3D
from vision_pipeline.perception.contracts import BoundingBox2D

PoseCovariance = tuple[
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
]


class SelectionKind(StrEnum):
    """Image evidence used to isolate an object's geometry."""

    SEGMENTATION_MASK = "segmentation_mask"
    BOUNDING_BOX = "bounding_box"
    FULL_CLOUD = "full_cloud"


class TrackStatus(StrEnum):
    """Whether a track is measured now or only propagated by its motion model."""

    TENTATIVE = "tentative"
    TRACKED = "tracked"
    PREDICTED = "predicted"
    LOST = "lost"


@dataclass(frozen=True, slots=True)
class ObjectObservation3D:
    """One object geometry estimate derived from one source measurement."""

    source: SampleHeader
    label: str
    confidence: float
    bounds: OrientedBox3D
    selection_kind: SelectionKind
    geometry_kind: GeometryKind
    point_count: int
    method: str
    image_box: BoundingBox2D | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source, SampleHeader):
            raise TypeError("source must be a SampleHeader")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and between zero and one")
        if not isinstance(self.bounds, OrientedBox3D):
            raise TypeError("bounds must be an OrientedBox3D")
        if not isinstance(self.selection_kind, SelectionKind):
            raise TypeError("selection_kind must be a SelectionKind")
        if not isinstance(self.geometry_kind, GeometryKind):
            raise TypeError("geometry_kind must be a GeometryKind")
        if isinstance(self.point_count, bool) or not isinstance(self.point_count, int):
            raise TypeError("point_count must be an integer")
        if self.point_count <= 0:
            raise ValueError("point_count must be positive")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")
        if self.image_box is not None and not isinstance(self.image_box, BoundingBox2D):
            raise TypeError("image_box must be a BoundingBox2D or None")
        if self.bounds.pose.reference_frame != self.source.frame_id:
            raise ValueError("observation bounds must be expressed in the source frame")


@dataclass(frozen=True, slots=True)
class ObjectTrack3D:
    """Controller-facing object state at one explicit time."""

    track_id: int
    observation: ObjectObservation3D
    state_time: TimePoint
    status: TrackStatus
    twist: Twist3D | None = None
    pose_covariance: PoseCovariance | None = None

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise TypeError("track_id must be an integer")
        if self.track_id <= 0:
            raise ValueError("track_id must be positive")
        if not isinstance(self.observation, ObjectObservation3D):
            raise TypeError("observation must be an ObjectObservation3D")
        if not isinstance(self.state_time, TimePoint):
            raise TypeError("state_time must be a TimePoint")
        if not isinstance(self.status, TrackStatus):
            raise TypeError("status must be a TrackStatus")
        if self.twist is not None:
            if not isinstance(self.twist, Twist3D):
                raise TypeError("twist must be a Twist3D or None")
            if self.twist.reference_frame != self.observation.bounds.pose.reference_frame:
                raise ValueError("track twist and pose must use the same reference frame")
        if self.pose_covariance is not None:
            if not isinstance(self.pose_covariance, tuple) or len(self.pose_covariance) != 36:
                raise TypeError("pose_covariance must be a 36-element row-major tuple or None")
            if not all(math.isfinite(value) for value in self.pose_covariance):
                raise ValueError("pose_covariance elements must be finite")


@dataclass(frozen=True, slots=True)
class GraspCandidate3D:
    """Arm-independent parallel-jaw grasp hypothesis for one tracked object."""

    track_id: int
    grasp_id: str
    gripper_pose: Pose3D
    jaw_width_metres: float
    score: float
    method: str

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int):
            raise TypeError("track_id must be an integer")
        if self.track_id <= 0:
            raise ValueError("track_id must be positive")
        if not isinstance(self.grasp_id, str) or not self.grasp_id.strip():
            raise ValueError("grasp_id must be a non-empty string")
        if not isinstance(self.gripper_pose, Pose3D):
            raise TypeError("gripper_pose must be a Pose3D")
        if not math.isfinite(self.jaw_width_metres) or self.jaw_width_metres <= 0:
            raise ValueError("jaw_width_metres must be positive and finite")
        if not math.isfinite(self.score) or not 0 <= self.score <= 1:
            raise ValueError("score must be finite and between zero and one")
        if not isinstance(self.method, str) or not self.method.strip():
            raise ValueError("method must be a non-empty string")


__all__ = [
    "GraspCandidate3D",
    "ObjectObservation3D",
    "ObjectTrack3D",
    "PoseCovariance",
    "SelectionKind",
    "TrackStatus",
]
