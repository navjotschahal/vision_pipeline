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


def _unit_quaternion(value: QuaternionXyzw, name: str) -> None:
    _finite_tuple(value, 4, name)
    norm = math.sqrt(sum(component * component for component in value))
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError(f"{name} must be a unit quaternion")


def _normalize_quaternion(value: QuaternionXyzw) -> QuaternionXyzw:
    norm = math.sqrt(sum(component * component for component in value))
    return (
        value[0] / norm,
        value[1] / norm,
        value[2] / norm,
        value[3] / norm,
    )


def _multiply_quaternions(left: QuaternionXyzw, right: QuaternionXyzw) -> QuaternionXyzw:
    left_x, left_y, left_z, left_w = left
    right_x, right_y, right_z, right_w = right
    return (
        left_w * right_x
        + left_x * right_w
        + left_y * right_z
        - left_z * right_y,
        left_w * right_y
        - left_x * right_z
        + left_y * right_w
        + left_z * right_x,
        left_w * right_z
        + left_x * right_y
        - left_y * right_x
        + left_z * right_w,
        left_w * right_w
        - left_x * right_x
        - left_y * right_y
        - left_z * right_z,
    )


def _rotate_vector(rotation_xyzw: QuaternionXyzw, vector: Vector3) -> Vector3:
    rotation_xyzw = _normalize_quaternion(rotation_xyzw)
    quaternion_vector = (vector[0], vector[1], vector[2], 0.0)
    conjugate = (
        -rotation_xyzw[0],
        -rotation_xyzw[1],
        -rotation_xyzw[2],
        rotation_xyzw[3],
    )
    rotated = _multiply_quaternions(
        _multiply_quaternions(rotation_xyzw, quaternion_vector),
        conjugate,
    )
    return (rotated[0], rotated[1], rotated[2])


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
        _unit_quaternion(self.orientation_xyzw, "orientation_xyzw")


@dataclass(frozen=True, slots=True)
class RigidTransform3D:
    """Rigid change of coordinates from ``source_frame`` into ``target_frame``.

    For a point ``p_source``, this transform applies the explicit convention
    ``p_target = rotate(rotation_xyzw, p_source) + translation_metres``. The
    quaternion therefore represents the target-from-source rotation and uses
    component order ``(x, y, z, w)``.
    """

    source_frame: FrameId
    target_frame: FrameId
    translation_metres: Vector3
    rotation_xyzw: QuaternionXyzw

    def __post_init__(self) -> None:
        if not isinstance(self.source_frame, FrameId):
            raise TypeError("source_frame must be a FrameId")
        if not isinstance(self.target_frame, FrameId):
            raise TypeError("target_frame must be a FrameId")
        _finite_tuple(self.translation_metres, 3, "translation_metres")
        _unit_quaternion(self.rotation_xyzw, "rotation_xyzw")

    def apply_point(self, point: Vector3) -> Vector3:
        """Express a source-frame point in the target frame."""

        _finite_tuple(point, 3, "point")
        rotated = _rotate_vector(self.rotation_xyzw, point)
        return (
            rotated[0] + self.translation_metres[0],
            rotated[1] + self.translation_metres[1],
            rotated[2] + self.translation_metres[2],
        )

    def apply_vector(self, vector: Vector3) -> Vector3:
        """Express a free vector in the target frame without translating it."""

        _finite_tuple(vector, 3, "vector")
        return _rotate_vector(self.rotation_xyzw, vector)

    def apply_pose(self, pose: Pose3D) -> Pose3D:
        """Express a source-frame object pose in the target frame."""

        if not isinstance(pose, Pose3D):
            raise TypeError("pose must be a Pose3D")
        if pose.reference_frame != self.source_frame:
            raise ValueError(
                f"pose frame {pose.reference_frame.value!r} does not match "
                f"transform source frame {self.source_frame.value!r}"
            )
        orientation = _normalize_quaternion(
            _multiply_quaternions(self.rotation_xyzw, pose.orientation_xyzw)
        )
        return Pose3D(
            reference_frame=self.target_frame,
            position_metres=self.apply_point(pose.position_metres),
            orientation_xyzw=orientation,
        )

    def inverse(self) -> RigidTransform3D:
        """Return the target-to-source inverse transform."""

        inverse_rotation = _normalize_quaternion(
            (
                -self.rotation_xyzw[0],
                -self.rotation_xyzw[1],
                -self.rotation_xyzw[2],
                self.rotation_xyzw[3],
            )
        )
        inverse_translation = _rotate_vector(
            inverse_rotation,
            (
                -self.translation_metres[0],
                -self.translation_metres[1],
                -self.translation_metres[2],
            ),
        )
        return RigidTransform3D(
            source_frame=self.target_frame,
            target_frame=self.source_frame,
            translation_metres=inverse_translation,
            rotation_xyzw=inverse_rotation,
        )

    def then(self, next_transform: RigidTransform3D) -> RigidTransform3D:
        """Compose this transform followed by ``next_transform``."""

        if not isinstance(next_transform, RigidTransform3D):
            raise TypeError("next_transform must be a RigidTransform3D")
        if self.target_frame != next_transform.source_frame:
            raise ValueError(
                f"transform target frame {self.target_frame.value!r} does not match "
                f"next transform source frame {next_transform.source_frame.value!r}"
            )
        rotation = _normalize_quaternion(
            _multiply_quaternions(next_transform.rotation_xyzw, self.rotation_xyzw)
        )
        return RigidTransform3D(
            source_frame=self.source_frame,
            target_frame=next_transform.target_frame,
            translation_metres=next_transform.apply_point(self.translation_metres),
            rotation_xyzw=rotation,
        )


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


__all__ = [
    "OrientedBox3D",
    "Pose3D",
    "QuaternionXyzw",
    "RigidTransform3D",
    "Twist3D",
    "Vector3",
]
