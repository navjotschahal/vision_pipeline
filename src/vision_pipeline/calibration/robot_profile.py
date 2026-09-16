"""Robot-agnostic description of a calibration rig: one camera, a board, one or more arms.

Everything robot-specific that calibration needs lives in a profile file, so the same
planner, capture, and solver serve an OpenArm, a Franka, a KUKA LBR, or a mixed pair:

- the TF frames that forward kinematics connects (``base_frame`` -> ``flange_frame``),
- the joint names and limits used to plan and bound calibration poses,
- the ROS 2 ``FollowJointTrajectory`` action that executes a pose,
- whether the camera is fixed (eye-to-hand) or carried by this arm (eye-in-hand),
- optionally, where each arm base nominally sits in a shared reference frame (for
  example a bimanual body link from the URDF), which lets two independent calibrations
  be checked against each other.

Profiles for other robots should be written from the running system, not guessed:
``ros2 run tf2_tools view_frames`` for frames, ``ros2 topic echo /joint_states --once``
for joint names, ``ros2 action list -t`` for the trajectory action.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .charuco import CharucoBoardSpec
from .hand_eye import HandEyeMount, Matrix, make_transform


class CalibrationProfileError(ValueError):
    """A calibration rig profile is missing a field or is internally inconsistent."""


def _require_name(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise CalibrationProfileError(f"{name} must be a non-empty string without padding")
    return value


def urdf_origin_matrix(xyz: Sequence[float], rpy: Sequence[float]) -> Matrix:
    """A URDF ``<origin xyz rpy>``: fixed-axis roll, pitch, yaw (R = Rz · Ry · Rx)."""

    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation = np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )
    return make_transform(rotation, [float(value) for value in xyz])


@dataclass(frozen=True, slots=True)
class ArmProfile:
    """One arm's calibration interface. Speeds are the caps for calibration motion only."""

    name: str
    robot_model: str
    mount: HandEyeMount
    base_frame: str
    flange_frame: str
    trajectory_action: str
    joint_names: tuple[str, ...]
    joint_lower_rad: tuple[float, ...]
    joint_upper_rad: tuple[float, ...]
    max_joint_speed_rad_s: float = 0.3
    joint_limit_margin_rad: float = 0.1
    nominal_reference_from_base: Matrix | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        for attribute in ("name", "robot_model", "base_frame", "flange_frame", "trajectory_action"):
            _require_name(getattr(self, attribute), attribute)
        if not isinstance(self.mount, HandEyeMount):
            raise CalibrationProfileError("mount must be a HandEyeMount")
        if not self.trajectory_action.startswith("/"):
            raise CalibrationProfileError("trajectory_action must be an absolute ROS 2 name")
        if self.base_frame == self.flange_frame:
            raise CalibrationProfileError("base_frame and flange_frame must differ")
        count = len(self.joint_names)
        if count == 0 or len(set(self.joint_names)) != count:
            raise CalibrationProfileError(f"arm {self.name!r} needs unique joint names")
        if len(self.joint_lower_rad) != count or len(self.joint_upper_rad) != count:
            raise CalibrationProfileError(f"arm {self.name!r} needs one limit pair per joint")
        margin = self.joint_limit_margin_rad
        for joint, low, high in zip(
            self.joint_names, self.joint_lower_rad, self.joint_upper_rad, strict=True
        ):
            if not (math.isfinite(low) and math.isfinite(high)) or high - low <= 2 * margin:
                raise CalibrationProfileError(
                    f"joint {joint!r} has invalid limits for margin {margin}"
                )
        if (
            not math.isfinite(self.max_joint_speed_rad_s)
            or not 0 < self.max_joint_speed_rad_s <= 1.0
        ):
            raise CalibrationProfileError("max_joint_speed_rad_s must be in (0, 1] for calibration")
        if (
            self.nominal_reference_from_base is not None
            and self.nominal_reference_from_base.shape != (4, 4)
        ):
            raise CalibrationProfileError("nominal_reference_from_base must be a 4x4 transform")

    def within_limits(self, positions: Sequence[float]) -> bool:
        """True when every joint is inside its limits by at least the safety margin."""

        if len(positions) != len(self.joint_names):
            raise ValueError(f"expected {len(self.joint_names)} joint positions")
        margin = self.joint_limit_margin_rad
        return all(
            low + margin <= value <= high - margin
            for value, low, high in zip(
                positions, self.joint_lower_rad, self.joint_upper_rad, strict=True
            )
        )


@dataclass(frozen=True, slots=True)
class CalibrationRig:
    name: str
    camera_frame: str
    camera_serial: str | None
    board: CharucoBoardSpec
    arms: tuple[ArmProfile, ...]
    reference_frame: str | None = None

    def __post_init__(self) -> None:
        _require_name(self.name, "rig name")
        _require_name(self.camera_frame, "camera_frame")
        if not self.arms:
            raise CalibrationProfileError("a rig needs at least one arm")
        names = [arm.name for arm in self.arms]
        if len(set(names)) != len(names):
            raise CalibrationProfileError("arm names must be unique within a rig")
        joints = [joint for arm in self.arms for joint in arm.joint_names]
        if len(set(joints)) != len(joints):
            raise CalibrationProfileError("joint names must be unique across arms")
        nominal = [arm.nominal_reference_from_base is not None for arm in self.arms]
        if any(nominal) and self.reference_frame is None:
            raise CalibrationProfileError("nominal arm mounts require a reference_frame")

    def arm(self, name: str) -> ArmProfile:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise KeyError(f"rig {self.name!r} has no arm {name!r}")


def _mapping(raw: Mapping[str, Any], key: str, path: str | Path) -> Mapping[str, Any]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise CalibrationProfileError(f"{path}: {key!r} must be a mapping")
    return value


def load_calibration_rig(path: str | Path) -> CalibrationRig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, Mapping):
        raise CalibrationProfileError(f"{path}: top level must be a mapping")
    try:
        camera = _mapping(raw, "camera", path)
        board = _mapping(raw, "board", path)
        reference = raw.get("reference_frame")
        arms = []
        for arm in raw["arms"]:
            joints = arm["joints"]
            nominal = arm.get("nominal_base_in_reference")
            arms.append(
                ArmProfile(
                    name=str(arm["name"]),
                    robot_model=str(arm["robot_model"]),
                    mount=HandEyeMount(arm["mount"]),
                    base_frame=str(arm["base_frame"]),
                    flange_frame=str(arm["flange_frame"]),
                    trajectory_action=str(arm["trajectory_action"]),
                    joint_names=tuple(str(joint["name"]) for joint in joints),
                    joint_lower_rad=tuple(float(joint["lower"]) for joint in joints),
                    joint_upper_rad=tuple(float(joint["upper"]) for joint in joints),
                    max_joint_speed_rad_s=float(arm.get("max_joint_speed_rad_s", 0.3)),
                    joint_limit_margin_rad=float(arm.get("joint_limit_margin_rad", 0.1)),
                    nominal_reference_from_base=(
                        None
                        if nominal is None
                        else urdf_origin_matrix(nominal["xyz"], nominal["rpy"])
                    ),
                )
            )
        return CalibrationRig(
            name=str(raw["rig"]),
            camera_frame=str(camera["frame"]),
            camera_serial=None if camera.get("serial") is None else str(camera["serial"]),
            board=CharucoBoardSpec(
                squares_x=int(board["squares_x"]),
                squares_y=int(board["squares_y"]),
                square_length_metres=float(board["square_length_metres"]),
                marker_length_metres=float(board["marker_length_metres"]),
                dictionary=str(board["dictionary"]),
            ),
            arms=tuple(arms),
            reference_frame=None if reference is None else str(reference),
        )
    except CalibrationProfileError:
        raise
    except (KeyError, TypeError, ValueError) as error:
        raise CalibrationProfileError(f"{path}: {error}") from error


__all__ = [
    "ArmProfile",
    "CalibrationProfileError",
    "CalibrationRig",
    "load_calibration_rig",
    "urdf_origin_matrix",
]
