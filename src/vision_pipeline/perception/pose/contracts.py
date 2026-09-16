"""Model-independent human-pose observations.

Human pose estimated from an image is a derived observation, not a physical sensor
measurement.  These contracts retain the exact source header so capture/receipt time,
camera frame, and calibration provenance are not lost between perception and control.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from vision_pipeline.contracts import ComputePlacement, FrameId, SampleHeader
from vision_pipeline.image import ImageFrame


class HumanJoint(StrEnum):
    """BlazePose's 33 anatomical landmark names, independent of the backend."""

    NOSE = "nose"
    LEFT_EYE_INNER = "left_eye_inner"
    LEFT_EYE = "left_eye"
    LEFT_EYE_OUTER = "left_eye_outer"
    RIGHT_EYE_INNER = "right_eye_inner"
    RIGHT_EYE = "right_eye"
    RIGHT_EYE_OUTER = "right_eye_outer"
    LEFT_EAR = "left_ear"
    RIGHT_EAR = "right_ear"
    MOUTH_LEFT = "mouth_left"
    MOUTH_RIGHT = "mouth_right"
    LEFT_SHOULDER = "left_shoulder"
    RIGHT_SHOULDER = "right_shoulder"
    LEFT_ELBOW = "left_elbow"
    RIGHT_ELBOW = "right_elbow"
    LEFT_WRIST = "left_wrist"
    RIGHT_WRIST = "right_wrist"
    LEFT_PINKY = "left_pinky"
    RIGHT_PINKY = "right_pinky"
    LEFT_INDEX = "left_index"
    RIGHT_INDEX = "right_index"
    LEFT_THUMB = "left_thumb"
    RIGHT_THUMB = "right_thumb"
    LEFT_HIP = "left_hip"
    RIGHT_HIP = "right_hip"
    LEFT_KNEE = "left_knee"
    RIGHT_KNEE = "right_knee"
    LEFT_ANKLE = "left_ankle"
    RIGHT_ANKLE = "right_ankle"
    LEFT_HEEL = "left_heel"
    RIGHT_HEEL = "right_heel"
    LEFT_FOOT_INDEX = "left_foot_index"
    RIGHT_FOOT_INDEX = "right_foot_index"


def _confidence(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty identifier")


@dataclass(frozen=True, slots=True)
class HumanKeypoint2D:
    """One anatomical point in the unmirrored source image's pixel coordinates."""

    joint: HumanJoint
    image_xy: tuple[float, float]
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.joint, HumanJoint):
            raise TypeError("joint must be a HumanJoint")
        if not isinstance(self.image_xy, tuple) or len(self.image_xy) != 2:
            raise TypeError("image_xy must be a 2-element tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in self.image_xy
        ):
            raise TypeError("image_xy elements must be numbers")
        if not all(math.isfinite(value) for value in self.image_xy):
            raise ValueError("image_xy elements must be finite")
        _confidence(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class HumanKeypoint3D:
    """A depth-supported joint in metres in a named optical/reference frame."""

    joint: HumanJoint
    position_metres: tuple[float, float, float]
    confidence: float
    depth_mad_metres: float
    depth_sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.joint, HumanJoint):
            raise TypeError("joint must be a HumanJoint")
        if not isinstance(self.position_metres, tuple) or len(self.position_metres) != 3:
            raise TypeError("position_metres must be a 3-element tuple")
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in self.position_metres
        ):
            raise TypeError("position_metres elements must be numbers")
        if not all(math.isfinite(value) for value in self.position_metres):
            raise ValueError("position_metres elements must be finite")
        _confidence(self.confidence, "confidence")
        if (
            isinstance(self.depth_mad_metres, bool)
            or not isinstance(self.depth_mad_metres, int | float)
        ):
            raise TypeError("depth_mad_metres must be a number")
        if not math.isfinite(self.depth_mad_metres) or self.depth_mad_metres < 0:
            raise ValueError("depth_mad_metres must be finite and non-negative")
        if (
            isinstance(self.depth_sample_count, bool)
            or not isinstance(self.depth_sample_count, int)
        ):
            raise TypeError("depth_sample_count must be an integer")
        if self.depth_sample_count <= 0:
            raise ValueError("depth_sample_count must be positive")


def _validate_unique_joints(
    landmarks: tuple[HumanKeypoint2D, ...] | tuple[HumanKeypoint3D, ...],
) -> None:
    joints = [landmark.joint for landmark in landmarks]
    if len(set(joints)) != len(joints):
        raise ValueError("landmarks must not contain duplicate joints")


@dataclass(frozen=True, slots=True)
class HumanPose2D:
    """Pose inferred from one exact, unmirrored color measurement."""

    source_header: SampleHeader
    person_id: str
    landmarks: tuple[HumanKeypoint2D, ...]
    producer: str
    produced_on: ComputePlacement

    def __post_init__(self) -> None:
        if not isinstance(self.source_header, SampleHeader):
            raise TypeError("source_header must be a SampleHeader")
        _identifier(self.person_id, "person_id")
        if not isinstance(self.landmarks, tuple):
            raise TypeError("landmarks must be a tuple")
        if any(not isinstance(item, HumanKeypoint2D) for item in self.landmarks):
            raise TypeError("landmarks must contain HumanKeypoint2D values")
        _validate_unique_joints(self.landmarks)
        _identifier(self.producer, "producer")
        if not isinstance(self.produced_on, ComputePlacement):
            raise TypeError("produced_on must be a ComputePlacement")

    def get(self, joint: HumanJoint) -> HumanKeypoint2D | None:
        return next((item for item in self.landmarks if item.joint is joint), None)


@dataclass(frozen=True, slots=True)
class HumanPose3D:
    """Metric pose lifted from RGB keypoints using aligned depth."""

    source_header: SampleHeader
    person_id: str
    reference_frame: FrameId
    landmarks: tuple[HumanKeypoint3D, ...]
    producer: str
    produced_on: ComputePlacement

    def __post_init__(self) -> None:
        if not isinstance(self.source_header, SampleHeader):
            raise TypeError("source_header must be a SampleHeader")
        _identifier(self.person_id, "person_id")
        if not isinstance(self.reference_frame, FrameId):
            raise TypeError("reference_frame must be a FrameId")
        if not isinstance(self.landmarks, tuple):
            raise TypeError("landmarks must be a tuple")
        if any(not isinstance(item, HumanKeypoint3D) for item in self.landmarks):
            raise TypeError("landmarks must contain HumanKeypoint3D values")
        _validate_unique_joints(self.landmarks)
        _identifier(self.producer, "producer")
        if not isinstance(self.produced_on, ComputePlacement):
            raise TypeError("produced_on must be a ComputePlacement")

    def get(self, joint: HumanJoint) -> HumanKeypoint3D | None:
        return next((item for item in self.landmarks if item.joint is joint), None)


class HumanPoseEstimator(Protocol):
    """Replaceable 2D pose backend; scheduling is a separate concern."""

    def estimate(self, image: ImageFrame, source_header: SampleHeader) -> HumanPose2D | None:
        """Infer at most one explicitly selected operator from a color frame."""


__all__ = [
    "HumanJoint",
    "HumanKeypoint2D",
    "HumanKeypoint3D",
    "HumanPose2D",
    "HumanPose3D",
    "HumanPoseEstimator",
]
