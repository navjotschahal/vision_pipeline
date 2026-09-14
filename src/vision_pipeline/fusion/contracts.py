"""Typed inputs and outputs for asynchronous kinematic state fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vision_pipeline.contracts import FrameId, TimePoint
from vision_pipeline.geometry.spatial import Vector3

Covariance3 = tuple[float, ...]
Covariance6 = tuple[float, ...]


def _vector3(value: Vector3, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{name} must be a three-element tuple")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise TypeError(f"{name} elements must be numbers")
    if not all(math.isfinite(item) for item in value):
        raise ValueError(f"{name} elements must be finite")


def _covariance(value: tuple[float, ...], length: int, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != length:
        raise TypeError(f"{name} must be a {length}-element row-major tuple")
    if not all(isinstance(item, int | float) and math.isfinite(item) for item in value):
        raise ValueError(f"{name} elements must be finite numbers")


@dataclass(frozen=True, slots=True)
class ImuSample:
    """IMU motion sample prepared for a translational fusion filter.

    Raw accelerometer measurements normally include gravity and use the moving sensor
    frame. This first estimator deliberately accepts only acceleration that an upstream
    attitude/gravity stage has transformed into the common fusion frame.
    """

    source_id: str
    measured_at: TimePoint
    reference_frame: FrameId
    linear_acceleration_metres_per_second2: Vector3
    angular_velocity_radians_per_second: Vector3
    acceleration_covariance: Covariance3
    gravity_compensated: bool

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.measured_at, TimePoint):
            raise TypeError("measured_at must be a TimePoint")
        if not isinstance(self.reference_frame, FrameId):
            raise TypeError("reference_frame must be a FrameId")
        _vector3(
            self.linear_acceleration_metres_per_second2,
            "linear_acceleration_metres_per_second2",
        )
        _vector3(
            self.angular_velocity_radians_per_second,
            "angular_velocity_radians_per_second",
        )
        _covariance(self.acceleration_covariance, 9, "acceleration_covariance")
        if not isinstance(self.gravity_compensated, bool):
            raise TypeError("gravity_compensated must be a boolean")


@dataclass(frozen=True, slots=True)
class PositionObservation3D:
    """A metric 3D correction produced by RGB-D, stereo, or another pose sensor."""

    source_id: str
    measured_at: TimePoint
    reference_frame: FrameId
    position_metres: Vector3
    covariance: Covariance3

    def __post_init__(self) -> None:
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(self.measured_at, TimePoint):
            raise TypeError("measured_at must be a TimePoint")
        if not isinstance(self.reference_frame, FrameId):
            raise TypeError("reference_frame must be a FrameId")
        _vector3(self.position_metres, "position_metres")
        _covariance(self.covariance, 9, "covariance")


@dataclass(frozen=True, slots=True)
class FusedKinematicState:
    """Estimated translational state and uncertainty in a shared frame."""

    state_time: TimePoint
    reference_frame: FrameId
    position_metres: Vector3
    velocity_metres_per_second: Vector3
    covariance: Covariance6
    contributing_sources: tuple[str, ...]
    correction_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.state_time, TimePoint):
            raise TypeError("state_time must be a TimePoint")
        if not isinstance(self.reference_frame, FrameId):
            raise TypeError("reference_frame must be a FrameId")
        _vector3(self.position_metres, "position_metres")
        _vector3(self.velocity_metres_per_second, "velocity_metres_per_second")
        _covariance(self.covariance, 36, "covariance")
        if not isinstance(self.contributing_sources, tuple) or not all(
            isinstance(source, str) and source.strip() for source in self.contributing_sources
        ):
            raise TypeError("contributing_sources must be a tuple of non-empty strings")
        if isinstance(self.correction_count, bool) or not isinstance(self.correction_count, int):
            raise TypeError("correction_count must be an integer")
        if self.correction_count < 0:
            raise ValueError("correction_count must be non-negative")


__all__ = [
    "Covariance3",
    "Covariance6",
    "FusedKinematicState",
    "ImuSample",
    "PositionObservation3D",
]
