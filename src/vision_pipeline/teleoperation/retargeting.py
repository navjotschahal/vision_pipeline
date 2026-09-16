"""Clutch-relative, position-only retargeting for a pair of robot arms."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vision_pipeline.contracts import ClockDomainMismatchError, FrameId, TimePoint
from vision_pipeline.geometry.spatial import Pose3D, Vector3
from vision_pipeline.perception.pose import HumanJoint, HumanKeypoint3D, HumanPose3D
from vision_pipeline.teleoperation.contracts import (
    ArmCartesianTarget,
    ArmSide,
    BimanualCartesianTarget,
    TeleopState,
)

Matrix3RowMajor = tuple[float, float, float, float, float, float, float, float, float]


def _finite_vector(value: tuple[float, ...], length: int, name: str) -> None:
    if not isinstance(value, tuple) or len(value) != length:
        raise TypeError(f"{name} must be a {length}-element tuple")
    if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
        raise TypeError(f"{name} elements must be numbers")
    if not all(math.isfinite(item) for item in value):
        raise ValueError(f"{name} elements must be finite")


def _subtract(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale_components(vector: Vector3, gain: Vector3) -> Vector3:
    return (vector[0] * gain[0], vector[1] * gain[1], vector[2] * gain[2])


def _matvec(matrix: Matrix3RowMajor, vector: Vector3) -> Vector3:
    return (
        matrix[0] * vector[0] + matrix[1] * vector[1] + matrix[2] * vector[2],
        matrix[3] * vector[0] + matrix[4] * vector[1] + matrix[5] * vector[2],
        matrix[6] * vector[0] + matrix[7] * vector[1] + matrix[8] * vector[2],
    )


def _determinant(matrix: Matrix3RowMajor) -> float:
    return (
        matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6])
    )


def _validate_rotation(matrix: Matrix3RowMajor) -> None:
    _finite_vector(matrix, 9, "operator_to_robot_rotation")
    rows = (matrix[0:3], matrix[3:6], matrix[6:9])
    for index, row in enumerate(rows):
        norm = sum(value * value for value in row)
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError(f"operator_to_robot_rotation row {index} must have unit length")
    for first, second in ((rows[0], rows[1]), (rows[0], rows[2]), (rows[1], rows[2])):
        if not math.isclose(
            sum(a * b for a, b in zip(first, second, strict=True)),
            0.0,
            abs_tol=1e-6,
        ):
            raise ValueError("operator_to_robot_rotation rows must be orthogonal")
    if not math.isclose(_determinant(matrix), 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise ValueError("operator_to_robot_rotation must be a proper rotation")


@dataclass(frozen=True, slots=True)
class AxisAlignedWorkspace:
    minimum_metres: Vector3
    maximum_metres: Vector3

    def __post_init__(self) -> None:
        _finite_vector(self.minimum_metres, 3, "minimum_metres")
        _finite_vector(self.maximum_metres, 3, "maximum_metres")
        if any(
            low >= high
            for low, high in zip(self.minimum_metres, self.maximum_metres, strict=True)
        ):
            raise ValueError("workspace minimums must be smaller than maximums")

    def clamp(self, point: Vector3) -> tuple[Vector3, bool]:
        bounded = tuple(
            min(max(value, low), high)
            for value, low, high in zip(
                point, self.minimum_metres, self.maximum_metres, strict=True
            )
        )
        result = (float(bounded[0]), float(bounded[1]), float(bounded[2]))
        return result, result != point


@dataclass(frozen=True, slots=True)
class ArmRetargetingConfig:
    side: ArmSide
    operator_frame: FrameId
    robot_base_frame: FrameId
    controlled_tool_frame: FrameId
    operator_to_robot_rotation: Matrix3RowMajor
    translation_gain: Vector3
    workspace: AxisAlignedWorkspace
    maximum_linear_speed_metres_per_second: float = 0.25
    minimum_landmark_confidence: float = 0.6

    def __post_init__(self) -> None:
        if not isinstance(self.side, ArmSide):
            raise TypeError("side must be an ArmSide")
        for name in ("operator_frame", "robot_base_frame", "controlled_tool_frame"):
            if not isinstance(getattr(self, name), FrameId):
                raise TypeError(f"{name} must be a FrameId")
        _validate_rotation(self.operator_to_robot_rotation)
        _finite_vector(self.translation_gain, 3, "translation_gain")
        if any(value <= 0 for value in self.translation_gain):
            raise ValueError("translation_gain elements must be positive")
        if not isinstance(self.workspace, AxisAlignedWorkspace):
            raise TypeError("workspace must be an AxisAlignedWorkspace")
        speed = self.maximum_linear_speed_metres_per_second
        if isinstance(speed, bool) or not isinstance(speed, int | float):
            raise TypeError("maximum_linear_speed_metres_per_second must be a number")
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("maximum_linear_speed_metres_per_second must be positive and finite")
        confidence = self.minimum_landmark_confidence
        if isinstance(confidence, bool) or not isinstance(confidence, int | float):
            raise TypeError("minimum_landmark_confidence must be a number")
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("minimum_landmark_confidence must be between zero and one")


@dataclass(frozen=True, slots=True)
class BimanualRetargetingConfig:
    left: ArmRetargetingConfig
    right: ArmRetargetingConfig
    maximum_observation_age_seconds: float = 0.15
    target_ttl_seconds: float = 0.10
    maximum_update_gap_seconds: float = 0.5

    def __post_init__(self) -> None:
        if self.left.side is not ArmSide.LEFT or self.right.side is not ArmSide.RIGHT:
            raise ValueError("left/right retargeting configs have incorrect sides")
        if self.left.operator_frame != self.right.operator_frame:
            raise ValueError("both arms must use one common operator frame")
        for name in (
            "maximum_observation_age_seconds",
            "target_ttl_seconds",
            "maximum_update_gap_seconds",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class _ArmAnchor:
    operator_wrist_from_shoulder: Vector3
    robot_pose: Pose3D


@dataclass(frozen=True, slots=True)
class _Calibration:
    operator_id: str
    left: _ArmAnchor
    right: _ArmAnchor


_ARM_JOINTS = {
    ArmSide.LEFT: (HumanJoint.LEFT_SHOULDER, HumanJoint.LEFT_ELBOW, HumanJoint.LEFT_WRIST),
    ArmSide.RIGHT: (
        HumanJoint.RIGHT_SHOULDER,
        HumanJoint.RIGHT_ELBOW,
        HumanJoint.RIGHT_WRIST,
    ),
}


class BimanualRetargeter:
    """Map torso-relative human wrist motion to bounded dual-arm targets.

    ``engage`` is the clutch edge: it atomically captures the operator's neutral pose
    and both robot end-effector poses. Every invalid update holds both arms together.
    """

    def __init__(self, config: BimanualRetargetingConfig) -> None:
        self._config = config
        self._calibration: _Calibration | None = None
        self._last_left: ArmCartesianTarget | None = None
        self._last_right: ArmCartesianTarget | None = None
        self._last_command_at: TimePoint | None = None
        self._last_observation_at: TimePoint | None = None
        self._sequence = 0

    @property
    def is_armed(self) -> bool:
        return self._calibration is not None

    def engage(
        self,
        pose: HumanPose3D,
        *,
        left_robot_pose: Pose3D,
        right_robot_pose: Pose3D,
        now: TimePoint,
    ) -> None:
        """Capture a jump-free neutral mapping for one explicitly identified operator."""

        if pose.reference_frame != self._config.left.operator_frame:
            raise ValueError("pose is not expressed in the configured operator frame")
        if left_robot_pose.reference_frame != self._config.left.robot_base_frame:
            raise ValueError("left robot anchor is not in the configured left base frame")
        if right_robot_pose.reference_frame != self._config.right.robot_base_frame:
            raise ValueError("right robot anchor is not in the configured right base frame")
        left_relative, _ = self._operator_arm(pose, self._config.left)
        right_relative, _ = self._operator_arm(pose, self._config.right)
        self._calibration = _Calibration(
            operator_id=pose.person_id,
            left=_ArmAnchor(left_relative, left_robot_pose),
            right=_ArmAnchor(right_relative, right_robot_pose),
        )
        self._last_left = self._anchor_target(self._config.left, left_robot_pose)
        self._last_right = self._anchor_target(self._config.right, right_robot_pose)
        self._last_command_at = now
        self._last_observation_at = None

    def disarm(self) -> None:
        self._calibration = None
        self._last_left = None
        self._last_right = None
        self._last_command_at = None
        self._last_observation_at = None

    def update(
        self,
        pose: HumanPose3D,
        *,
        now: TimePoint,
        deadman_pressed: bool,
    ) -> BimanualCartesianTarget:
        calibration = self._calibration
        if calibration is None:
            return self._emit(
                TeleopState.DISARMED,
                now,
                source_pose=pose,
                left=None,
                right=None,
                reason="clutch-not-engaged",
            )
        assert self._last_left is not None
        assert self._last_right is not None
        assert self._last_command_at is not None

        invalid_reason = self._validate_update(pose, now, deadman_pressed)
        if invalid_reason is not None:
            return self._hold(pose, now, invalid_reason)

        try:
            elapsed_seconds = (now - self._last_command_at) / 1_000_000_000
        except ClockDomainMismatchError:
            return self._hold(pose, now, "controller-clock-domain-mismatch")
        if elapsed_seconds <= 0:
            return self._hold(pose, now, "non-monotonic-controller-time")
        if elapsed_seconds > self._config.maximum_update_gap_seconds:
            return self._hold(pose, now, "controller-update-gap")

        left_relative, left_confidence = self._operator_arm(pose, self._config.left)
        right_relative, right_confidence = self._operator_arm(pose, self._config.right)
        left = self._retarget_arm(
            self._config.left,
            calibration.left,
            left_relative,
            left_confidence,
            self._last_left,
            elapsed_seconds,
        )
        right = self._retarget_arm(
            self._config.right,
            calibration.right,
            right_relative,
            right_confidence,
            self._last_right,
            elapsed_seconds,
        )
        self._last_left = left
        self._last_right = right
        self._last_command_at = now
        self._last_observation_at = pose.source_header.received_at
        return self._emit(
            TeleopState.ACTIVE,
            now,
            source_pose=pose,
            left=left,
            right=right,
            reason=None,
        )

    def _validate_update(
        self, pose: HumanPose3D, now: TimePoint, deadman_pressed: bool
    ) -> str | None:
        calibration = self._calibration
        assert calibration is not None
        if not isinstance(deadman_pressed, bool):
            raise TypeError("deadman_pressed must be a boolean")
        if not deadman_pressed:
            return "deadman-released"
        if pose.person_id != calibration.operator_id:
            return "operator-identity-changed"
        if pose.reference_frame != self._config.left.operator_frame:
            return "operator-frame-changed"
        received_at = pose.source_header.received_at
        try:
            age_seconds = (now - received_at) / 1_000_000_000
        except ClockDomainMismatchError:
            return "observation-clock-domain-mismatch"
        if age_seconds < 0:
            return "observation-from-future"
        if age_seconds > self._config.maximum_observation_age_seconds:
            return "stale-observation"
        if self._last_observation_at is not None:
            try:
                if received_at - self._last_observation_at <= 0:
                    return "duplicate-or-out-of-order-observation"
            except ClockDomainMismatchError:
                return "observation-clock-domain-changed"
        for arm in (self._config.left, self._config.right):
            try:
                self._operator_arm(pose, arm)
            except ValueError as error:
                return str(error)
        return None

    @staticmethod
    def _operator_arm(
        pose: HumanPose3D, config: ArmRetargetingConfig
    ) -> tuple[Vector3, float]:
        shoulder_joint, elbow_joint, wrist_joint = _ARM_JOINTS[config.side]
        landmarks: list[HumanKeypoint3D] = []
        for joint in (shoulder_joint, elbow_joint, wrist_joint):
            landmark = pose.get(joint)
            if landmark is None:
                raise ValueError(f"{config.side.value}-{joint.value}-missing")
            if landmark.confidence < config.minimum_landmark_confidence:
                raise ValueError(f"{config.side.value}-{joint.value}-low-confidence")
            landmarks.append(landmark)
        shoulder, _elbow, wrist = landmarks
        return (
            _subtract(wrist.position_metres, shoulder.position_metres),
            min(item.confidence for item in landmarks),
        )

    @staticmethod
    def _anchor_target(config: ArmRetargetingConfig, pose: Pose3D) -> ArmCartesianTarget:
        return ArmCartesianTarget(
            side=config.side,
            controlled_tool_frame=config.controlled_tool_frame,
            pose=pose,
            confidence=1.0,
        )

    @staticmethod
    def _rate_limit(
        previous: Vector3, candidate: Vector3, maximum_step_metres: float
    ) -> tuple[Vector3, bool]:
        delta = _subtract(candidate, previous)
        distance = math.sqrt(sum(value * value for value in delta))
        if distance <= maximum_step_metres:
            return candidate, False
        ratio = maximum_step_metres / distance
        limited_delta = (delta[0] * ratio, delta[1] * ratio, delta[2] * ratio)
        return _add(previous, limited_delta), True

    def _retarget_arm(
        self,
        config: ArmRetargetingConfig,
        anchor: _ArmAnchor,
        operator_relative: Vector3,
        confidence: float,
        previous: ArmCartesianTarget,
        elapsed_seconds: float,
    ) -> ArmCartesianTarget:
        human_delta = _subtract(operator_relative, anchor.operator_wrist_from_shoulder)
        robot_delta = _matvec(
            config.operator_to_robot_rotation,
            _scale_components(human_delta, config.translation_gain),
        )
        candidate = _add(anchor.robot_pose.position_metres, robot_delta)
        candidate, workspace_limited = config.workspace.clamp(candidate)
        candidate, rate_limited = self._rate_limit(
            previous.pose.position_metres,
            candidate,
            config.maximum_linear_speed_metres_per_second * elapsed_seconds,
        )
        return ArmCartesianTarget(
            side=config.side,
            controlled_tool_frame=config.controlled_tool_frame,
            pose=Pose3D(
                reference_frame=config.robot_base_frame,
                position_metres=candidate,
                orientation_xyzw=anchor.robot_pose.orientation_xyzw,
            ),
            confidence=confidence,
            limited=workspace_limited or rate_limited,
        )

    def _hold(self, pose: HumanPose3D, now: TimePoint, reason: str) -> BimanualCartesianTarget:
        assert self._last_left is not None
        assert self._last_right is not None
        return self._emit(
            TeleopState.HOLD,
            now,
            source_pose=pose,
            left=self._last_left,
            right=self._last_right,
            reason=reason,
        )

    def _emit(
        self,
        state: TeleopState,
        now: TimePoint,
        *,
        source_pose: HumanPose3D,
        left: ArmCartesianTarget | None,
        right: ArmCartesianTarget | None,
        reason: str | None,
    ) -> BimanualCartesianTarget:
        ttl_nanoseconds = round(self._config.target_ttl_seconds * 1_000_000_000)
        target = BimanualCartesianTarget(
            command_sequence=self._sequence,
            state=state,
            generated_at=now,
            expires_at=TimePoint(now.nanoseconds + ttl_nanoseconds, now.clock),
            source_header=source_pose.source_header,
            operator_id=source_pose.person_id,
            left=left,
            right=right,
            reason=reason,
        )
        self._sequence += 1
        return target


__all__ = [
    "ArmRetargetingConfig",
    "AxisAlignedWorkspace",
    "BimanualRetargeter",
    "BimanualRetargetingConfig",
    "Matrix3RowMajor",
]
