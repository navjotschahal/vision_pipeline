from __future__ import annotations

import pytest
from test_bimanual_retargeting import (
    CLOCK,
    LEFT_BASE,
    ArmSide,
    arm_config,
    operator_pose,
    robot_pose,
)

from vision_pipeline.contracts import TimePoint
from vision_pipeline.teleoperation import SingleArmRetargeter, TeleopState


def test_single_arm_moves_and_holds_without_right_arm() -> None:
    retargeter = SingleArmRetargeter(arm_config(ArmSide.LEFT))
    retargeter.engage(
        operator_pose(sequence=0, received_ns=1_000_000_000),
        robot_pose=robot_pose(LEFT_BASE, (0.4, 0.2, 0.5)),
        now=TimePoint(1_000_000_000, CLOCK),
    )
    moving = retargeter.update(
        operator_pose(sequence=1, received_ns=1_050_000_000, left_wrist_delta=(0.1, 0, 0)),
        now=TimePoint(1_050_000_000, CLOCK),
        deadman_pressed=True,
    )
    assert moving.state is TeleopState.ACTIVE
    assert moving.arm is not None
    assert moving.arm.pose.position_metres == pytest.approx((0.6, 0.2, 0.5))
    held = retargeter.update(
        operator_pose(sequence=2, received_ns=1_100_000_000),
        now=TimePoint(1_100_000_000, CLOCK),
        deadman_pressed=False,
    )
    assert held.state is TeleopState.HOLD
    assert held.arm == moving.arm
