"""Small, middleware-independent spatial contracts for robotics perception."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vision_pipeline.contracts import FrameId

Vector3 = tuple[float, float, float]
QuaternionXyzw = tuple[float, float, float, float]


def _finite_tuple(value: tuple[float, ...], length: int, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != length:
        raise TypeError(f"{name} must be a {length}-element tuple")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise TypeError(f"{name} elements must be numbers")
    if not all(math.isfinite(item) for item in value):
        raise ValueError(f"{name} elements must be finite")


@dataclass(frozen=True, slots=True)
class Pose3D:
    """Rigid pose expressed in one named reference frame.

    Quaternion component order is explicitly ``(x, y, z, w)``.
    """

    reference_frame: FrameId
    position_metres: Vector3
    orientation_xyzw: QuaternionXyzw

    def __post_init__(self) -> None:
        if not isinstance(self.reference_frame, FrameId):
            raise TypeError("reference_frame must be a FrameId")
        _finite_tuple(self.position_metres, 3, "position_metres")
        _finite_tuple(self.orientation_xyzw, 4, "orientation_xyzw")
        norm = math.sqrt(sum(component * component for component in self.orientation_xyzw))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("orientation_xyzw must be a unit quaternion")


@dataclass(frozen=True, slots=True)
class Twist3D:
    """Linear and angular object velocity expressed in one reference frame."""

    reference_frame: FrameId
    linear_metres_per_second: Vector3
    angular_radians_per_second: Vector3

    def __post_init__(self) -> None:
        if not isinstance(self.reference_frame, FrameId):
            raise TypeError("reference_frame must be a FrameId")
        _finite_tuple(self.linear_metres_per_second, 3, "linear_metres_per_second")
        _finite_tuple(self.angular_radians_per_second, 3, "angular_radians_per_second")


@dataclass(frozen=True, slots=True)
class OrientedBox3D:
    """A cuboid whose pose and full side lengths are expressed in metres."""

    pose: Pose3D
    size_metres: Vector3

    def __post_init__(self) -> None:
        if not isinstance(self.pose, Pose3D):
            raise TypeError("pose must be a Pose3D")
        _finite_tuple(self.size_metres, 3, "size_metres")
        if any(length <= 0 for length in self.size_metres):
            raise ValueError("all oriented-box side lengths must be positive")


__all__ = ["OrientedBox3D", "Pose3D", "QuaternionXyzw", "Twist3D", "Vector3"]
