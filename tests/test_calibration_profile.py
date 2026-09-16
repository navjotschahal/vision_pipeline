from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from vision_pipeline.calibration.hand_eye import HandEyeMount, invert_transform
from vision_pipeline.calibration.robot_profile import (
    CalibrationProfileError,
    load_calibration_rig,
    urdf_origin_matrix,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "calibration"


@pytest.mark.parametrize("version", ["v1", "v2"])
def test_openarm_profiles_load_with_both_arms(version: str) -> None:
    rig = load_calibration_rig(CONFIGS / f"openarm_{version}_bimanual.yaml")

    assert [arm.name for arm in rig.arms] == ["left", "right"]
    assert rig.reference_frame == "openarm_body_link0"
    for arm in rig.arms:
        assert arm.mount is HandEyeMount.EYE_TO_HAND
        assert len(arm.joint_names) == 7
        assert (
            arm.trajectory_action
            == f"/{arm.name}_joint_trajectory_controller/follow_joint_trajectory"
        )
        assert arm.within_limits(
            [
                0.5 * (lo + hi)
                for lo, hi in zip(arm.joint_lower_rad, arm.joint_upper_rad, strict=True)
            ]
        )
    left, right = rig.arm("left"), rig.arm("right")
    assert left.nominal_reference_from_base is not None
    assert right.nominal_reference_from_base is not None
    left_from_right = (
        invert_transform(left.nominal_reference_from_base) @ right.nominal_reference_from_base
    )
    # The URDF bases are 62 mm apart at the same height.
    assert np.linalg.norm(left_from_right[:3, 3]) == pytest.approx(0.062)


def test_joint_limit_margin_is_enforced() -> None:
    arm = load_calibration_rig(CONFIGS / "openarm_v2_bimanual.yaml").arm("left")
    at_limit = list(arm.joint_lower_rad)
    assert not arm.within_limits(at_limit)
    with pytest.raises(ValueError, match="7 joint positions"):
        arm.within_limits([0.0])


def test_profiles_reject_inconsistent_arms() -> None:
    arm = load_calibration_rig(CONFIGS / "openarm_v2_bimanual.yaml").arm("left")
    with pytest.raises(CalibrationProfileError, match="absolute"):
        dataclasses.replace(arm, trajectory_action="left_controller/follow_joint_trajectory")
    with pytest.raises(CalibrationProfileError, match="limit pair"):
        dataclasses.replace(arm, joint_lower_rad=arm.joint_lower_rad[:-1])
    with pytest.raises(CalibrationProfileError, match=r"\(0, 1\]"):
        dataclasses.replace(arm, max_joint_speed_rad_s=3.0)


def test_heterogeneous_rig_without_shared_body(tmp_path: Path) -> None:
    profile = tmp_path / "mixed.yaml"
    profile.write_text(
        """
rig: kuka_franka_example
camera: {frame: camera_color_optical}
board:
  squares_x: 7
  squares_y: 5
  square_length_metres: 0.03
  marker_length_metres: 0.022
  dictionary: DICT_4X4_50
arms:
  - name: kuka
    robot_model: example_a
    mount: eye_to_hand
    base_frame: a_base
    flange_frame: a_flange
    trajectory_action: /a_controller/follow_joint_trajectory
    joints: [{name: a1, lower: -2.9, upper: 2.9}, {name: a2, lower: -2.0, upper: 2.0}]
  - name: franka
    robot_model: example_b
    mount: eye_in_hand
    base_frame: b_base
    flange_frame: b_flange
    trajectory_action: /b_controller/follow_joint_trajectory
    joints: [{name: b1, lower: -2.7, upper: 2.7}]
"""
    )

    rig = load_calibration_rig(profile)

    assert rig.reference_frame is None
    assert rig.arm("franka").mount is HandEyeMount.EYE_IN_HAND
    assert rig.arm("kuka").nominal_reference_from_base is None


def test_urdf_rpy_matches_fixed_axis_convention() -> None:
    # Roll of -90 degrees maps local +y onto parent -z (OpenArm v1 left arm base).
    origin = urdf_origin_matrix((0.0, 0.031, 0.698), (-np.pi / 2, 0.0, 0.0))
    assert origin[:3, :3] @ np.array([0.0, 1.0, 0.0]) == pytest.approx((0.0, 0.0, -1.0), abs=1e-12)
    yaw = urdf_origin_matrix((0, 0, 0), (0.0, 0.0, np.pi / 2))
    assert yaw[:3, :3] @ np.array([1.0, 0.0, 0.0]) == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
