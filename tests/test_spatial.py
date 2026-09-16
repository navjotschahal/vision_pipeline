from __future__ import annotations

import math

import pytest

from vision_pipeline.contracts import FrameId
from vision_pipeline.geometry import Pose3D, RigidTransform3D

SOURCE = FrameId("camera_optical")
MIDDLE = FrameId("cell")
TARGET = FrameId("robot_base")
IDENTITY = (0.0, 0.0, 0.0, 1.0)
SQRT_HALF = math.sqrt(0.5)
QUARTER_TURN_Z = (0.0, 0.0, SQRT_HALF, SQRT_HALF)


def assert_same_orientation(
    actual: tuple[float, float, float, float],
    expected: tuple[float, float, float, float],
) -> None:
    dot = sum(left * right for left, right in zip(actual, expected, strict=True))
    assert abs(dot) == pytest.approx(1.0)


def test_rigid_transform_applies_rotation_and_translation_to_points_only() -> None:
    transform = RigidTransform3D(
        source_frame=SOURCE,
        target_frame=TARGET,
        translation_metres=(1.0, 2.0, 3.0),
        rotation_xyzw=QUARTER_TURN_Z,
    )

    assert transform.apply_vector((1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))
    assert transform.apply_point((1.0, 0.0, 0.0)) == pytest.approx((1.0, 3.0, 3.0))


def test_rigid_transform_does_not_scale_for_an_accepted_near_unit_quaternion() -> None:
    transform = RigidTransform3D(
        source_frame=SOURCE,
        target_frame=TARGET,
        translation_metres=(0.0, 0.0, 0.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.000001),
    )

    assert transform.apply_vector((1.0, 2.0, 3.0)) == pytest.approx((1.0, 2.0, 3.0))


def test_rigid_transform_applies_position_orientation_and_target_frame_to_pose() -> None:
    transform = RigidTransform3D(
        source_frame=SOURCE,
        target_frame=TARGET,
        translation_metres=(1.0, 2.0, 3.0),
        rotation_xyzw=QUARTER_TURN_Z,
    )
    pose = Pose3D(
        reference_frame=SOURCE,
        position_metres=(1.0, 0.0, 0.0),
        orientation_xyzw=(SQRT_HALF, 0.0, 0.0, SQRT_HALF),
    )

    transformed = transform.apply_pose(pose)

    assert transformed.reference_frame == TARGET
    assert transformed.position_metres == pytest.approx((1.0, 3.0, 3.0))
    assert_same_orientation(transformed.orientation_xyzw, (0.5, 0.5, 0.5, 0.5))


def test_inverse_round_trips_points_and_poses() -> None:
    transform = RigidTransform3D(
        source_frame=SOURCE,
        target_frame=TARGET,
        translation_metres=(0.4, -0.2, 1.5),
        rotation_xyzw=QUARTER_TURN_Z,
    )
    point = (0.7, -1.1, 2.0)
    pose = Pose3D(SOURCE, point, (SQRT_HALF, 0.0, 0.0, SQRT_HALF))

    inverse = transform.inverse()

    assert inverse.source_frame == TARGET
    assert inverse.target_frame == SOURCE
    assert inverse.apply_point(transform.apply_point(point)) == pytest.approx(point)
    round_tripped_pose = inverse.apply_pose(transform.apply_pose(pose))
    assert round_tripped_pose.reference_frame == SOURCE
    assert round_tripped_pose.position_metres == pytest.approx(pose.position_metres)
    assert_same_orientation(round_tripped_pose.orientation_xyzw, pose.orientation_xyzw)


def test_then_matches_sequential_application() -> None:
    source_to_middle = RigidTransform3D(
        source_frame=SOURCE,
        target_frame=MIDDLE,
        translation_metres=(1.0, 0.0, 0.0),
        rotation_xyzw=QUARTER_TURN_Z,
    )
    middle_to_target = RigidTransform3D(
        source_frame=MIDDLE,
        target_frame=TARGET,
        translation_metres=(0.0, 2.0, 0.0),
        rotation_xyzw=(0.0, SQRT_HALF, 0.0, SQRT_HALF),
    )
    point = (0.2, 0.3, 0.4)
    vector = (1.0, 2.0, 3.0)

    combined = source_to_middle.then(middle_to_target)

    assert combined.source_frame == SOURCE
    assert combined.target_frame == TARGET
    assert combined.apply_point(point) == pytest.approx(
        middle_to_target.apply_point(source_to_middle.apply_point(point))
    )
    assert combined.apply_vector(vector) == pytest.approx(
        middle_to_target.apply_vector(source_to_middle.apply_vector(vector))
    )


def test_transform_rejects_invalid_values_and_frame_mismatches() -> None:
    with pytest.raises(ValueError, match="unit quaternion"):
        RigidTransform3D(SOURCE, TARGET, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="finite"):
        RigidTransform3D(SOURCE, TARGET, (math.nan, 0.0, 0.0), IDENTITY)

    transform = RigidTransform3D(SOURCE, TARGET, (0.0, 0.0, 0.0), IDENTITY)
    with pytest.raises(ValueError, match="pose frame"):
        transform.apply_pose(Pose3D(MIDDLE, (0.0, 0.0, 0.0), IDENTITY))
    with pytest.raises(ValueError, match="next transform source frame"):
        transform.then(RigidTransform3D(MIDDLE, SOURCE, (0.0, 0.0, 0.0), IDENTITY))
