from __future__ import annotations

from dataclasses import replace

import pytest

from vision_pipeline.contracts import (
    ClockDomain,
    ClockKind,
    ComputeKind,
    ComputePlacement,
    FrameId,
    MeasurementKind,
    SampleHeader,
    TimePoint,
)
from vision_pipeline.geometry import Pose3D
from vision_pipeline.perception.pose import HumanJoint, HumanKeypoint3D, HumanPose3D
from vision_pipeline.teleoperation import (
    ArmRetargetingConfig,
    ArmSide,
    AxisAlignedWorkspace,
    BimanualRetargeter,
    BimanualRetargetingConfig,
    TeleopState,
)

CLOCK = ClockDomain("host/test-teleop", ClockKind.HOST_MONOTONIC)
OPTICAL = FrameId("realsense_color_optical")
LEFT_BASE = FrameId("left/panda_link0")
RIGHT_BASE = FrameId("right/panda_link0")
LEFT_TOOL = FrameId("left/franka_hand_tcp")
RIGHT_TOOL = FrameId("right/franka_hand_tcp")
IDENTITY = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
CPU = ComputePlacement(ComputeKind.HOST_CPU, "test-cpu")


def keypoint(
    joint: HumanJoint,
    position: tuple[float, float, float],
    confidence: float = 1.0,
) -> HumanKeypoint3D:
    return HumanKeypoint3D(joint, position, confidence, 0.002, 9)


def operator_pose(
    *,
    sequence: int,
    received_ns: int,
    operator_id: str = "operator-1",
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    left_wrist_delta: tuple[float, float, float] = (0.0, 0.0, 0.0),
    right_wrist_delta: tuple[float, float, float] = (0.0, 0.0, 0.0),
    right_wrist_confidence: float = 1.0,
) -> HumanPose3D:
    def moved(
        base: tuple[float, float, float],
        extra: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> tuple[float, float, float]:
        return tuple(
            base[index] + translation[index] + extra[index] for index in range(3)
        )  # type: ignore[return-value]

    source_header = SampleHeader(
        sample_id=f"rgb/{sequence}",
        source_id="realsense/rgb",
        sequence_number=sequence,
        measurement_kind=MeasurementKind.RGB_IMAGE,
        captured_at=None,
        received_at=TimePoint(received_ns, CLOCK),
        frame_id=OPTICAL,
        calibration=None,
        producer="test-camera",
        produced_on=CPU,
    )
    return HumanPose3D(
        source_header=source_header,
        person_id=operator_id,
        reference_frame=OPTICAL,
        landmarks=(
            keypoint(HumanJoint.LEFT_SHOULDER, moved((-0.2, 0.0, 1.5))),
            keypoint(HumanJoint.LEFT_ELBOW, moved((-0.4, 0.1, 1.3))),
            keypoint(HumanJoint.LEFT_WRIST, moved((-0.6, 0.2, 1.1), left_wrist_delta)),
            keypoint(HumanJoint.RIGHT_SHOULDER, moved((0.2, 0.0, 1.5))),
            keypoint(HumanJoint.RIGHT_ELBOW, moved((0.4, 0.1, 1.3))),
            keypoint(
                HumanJoint.RIGHT_WRIST,
                moved((0.6, 0.2, 1.1), right_wrist_delta),
                right_wrist_confidence,
            ),
        ),
        producer="fake-pose+depth",
        produced_on=CPU,
    )


def arm_config(side: ArmSide, *, speed: float = 10.0) -> ArmRetargetingConfig:
    left = side is ArmSide.LEFT
    return ArmRetargetingConfig(
        side=side,
        operator_frame=OPTICAL,
        robot_base_frame=LEFT_BASE if left else RIGHT_BASE,
        controlled_tool_frame=LEFT_TOOL if left else RIGHT_TOOL,
        operator_to_robot_rotation=IDENTITY,
        translation_gain=(2.0, 1.0, 0.5),
        workspace=AxisAlignedWorkspace((0.0, -0.5, 0.1), (1.0, 0.5, 1.0)),
        maximum_linear_speed_metres_per_second=speed,
        minimum_landmark_confidence=0.6,
    )


def robot_pose(frame: FrameId, position: tuple[float, float, float]) -> Pose3D:
    return Pose3D(frame, position, (0.0, 1.0, 0.0, 0.0))


def retargeter(*, speed: float = 10.0) -> BimanualRetargeter:
    return BimanualRetargeter(
        BimanualRetargetingConfig(
            left=arm_config(ArmSide.LEFT, speed=speed),
            right=arm_config(ArmSide.RIGHT, speed=speed),
        )
    )


def engage(subject: BimanualRetargeter) -> None:
    subject.engage(
        operator_pose(sequence=0, received_ns=1_000_000_000),
        left_robot_pose=robot_pose(LEFT_BASE, (0.4, 0.2, 0.5)),
        right_robot_pose=robot_pose(RIGHT_BASE, (0.4, -0.2, 0.5)),
        now=TimePoint(1_000_000_000, CLOCK),
    )


def test_clutch_relative_mapping_is_neutral_and_torso_translation_invariant() -> None:
    subject = retargeter()
    engage(subject)

    target = subject.update(
        operator_pose(
            sequence=1,
            received_ns=1_050_000_000,
            translation=(0.3, -0.1, 0.2),
        ),
        now=TimePoint(1_050_000_000, CLOCK),
        deadman_pressed=True,
    )

    assert target.state is TeleopState.ACTIVE
    assert target.left is not None and target.right is not None
    assert target.left.pose.position_metres == pytest.approx((0.4, 0.2, 0.5))
    assert target.right.pose.position_metres == pytest.approx((0.4, -0.2, 0.5))
    assert target.left.pose.orientation_xyzw == (0.0, 1.0, 0.0, 0.0)
    assert target.expires_at.nanoseconds == 1_150_000_000


def test_each_human_wrist_maps_only_to_its_robot_arm_with_axis_gain() -> None:
    subject = retargeter()
    engage(subject)

    target = subject.update(
        operator_pose(
            sequence=1,
            received_ns=1_050_000_000,
            left_wrist_delta=(0.1, -0.1, 0.2),
            right_wrist_delta=(-0.05, 0.0, 0.0),
        ),
        now=TimePoint(1_050_000_000, CLOCK),
        deadman_pressed=True,
    )

    assert target.left is not None and target.right is not None
    assert target.left.pose.position_metres == pytest.approx((0.6, 0.1, 0.6))
    assert target.right.pose.position_metres == pytest.approx((0.3, -0.2, 0.5))


@pytest.mark.parametrize(
    ("pose", "deadman", "reason"),
    [
        (
            operator_pose(
                sequence=1,
                received_ns=1_050_000_000,
                right_wrist_confidence=0.1,
            ),
            True,
            "right-right_wrist-low-confidence",
        ),
        (
            operator_pose(sequence=1, received_ns=1_050_000_000),
            False,
            "deadman-released",
        ),
        (
            operator_pose(
                sequence=1,
                received_ns=1_050_000_000,
                operator_id="someone-else",
            ),
            True,
            "operator-identity-changed",
        ),
        (
            operator_pose(sequence=1, received_ns=800_000_000),
            True,
            "stale-observation",
        ),
    ],
)
def test_any_invalid_arm_or_safety_gate_atomically_holds_both(
    pose: HumanPose3D, deadman: bool, reason: str
) -> None:
    subject = retargeter()
    engage(subject)

    target = subject.update(
        pose,
        now=TimePoint(1_050_000_000, CLOCK),
        deadman_pressed=deadman,
    )

    assert target.state is TeleopState.HOLD
    assert target.reason == reason
    assert target.left is not None and target.right is not None
    assert target.left.pose.position_metres == (0.4, 0.2, 0.5)
    assert target.right.pose.position_metres == (0.4, -0.2, 0.5)


def test_workspace_and_velocity_limits_are_explicit() -> None:
    subject = retargeter(speed=0.2)
    engage(subject)

    target = subject.update(
        operator_pose(
            sequence=1,
            received_ns=1_100_000_000,
            left_wrist_delta=(2.0, 0.0, 0.0),
        ),
        now=TimePoint(1_100_000_000, CLOCK),
        deadman_pressed=True,
    )

    assert target.left is not None
    assert target.left.pose.position_metres == pytest.approx((0.42, 0.2, 0.5))
    assert target.left.limited


def test_duplicate_observation_is_held_and_disarm_removes_targets() -> None:
    subject = retargeter()
    engage(subject)
    pose = operator_pose(sequence=1, received_ns=1_050_000_000)
    first = subject.update(
        pose,
        now=TimePoint(1_050_000_000, CLOCK),
        deadman_pressed=True,
    )
    duplicate = subject.update(
        pose,
        now=TimePoint(1_060_000_000, CLOCK),
        deadman_pressed=True,
    )
    subject.disarm()
    disarmed = subject.update(
        replace(pose, source_header=replace(pose.source_header, sequence_number=2)),
        now=TimePoint(1_070_000_000, CLOCK),
        deadman_pressed=True,
    )

    assert first.state is TeleopState.ACTIVE
    assert duplicate.state is TeleopState.HOLD
    assert duplicate.reason == "duplicate-or-out-of-order-observation"
    assert disarmed.state is TeleopState.DISARMED
    assert disarmed.left is None and disarmed.right is None

