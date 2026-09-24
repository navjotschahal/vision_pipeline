"""Webcam-only, two-axis simulated arm retargeting.

The camera cannot observe metric depth. One robot axis stays at its clutch value.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from vision_pipeline.contracts import ClockDomainMismatchError, FrameId, TimePoint
from vision_pipeline.geometry.spatial import Pose3D
from vision_pipeline.perception.pose import HumanJoint, HumanPose2D
from vision_pipeline.teleoperation.contracts import ArmCartesianTarget, ArmSide, TeleopState
from vision_pipeline.teleoperation.retargeting import AxisAlignedWorkspace
from vision_pipeline.teleoperation.single_arm import SingleArmCommand


@dataclass(frozen=True, slots=True)
class WebcamPlanarConfig:
    side: ArmSide = ArmSide.RIGHT
    robot_base_frame: FrameId = FrameId("kuka/base")
    tool_frame: FrameId = FrameId("kuka/attachment_site")
    horizontal_robot_axis: int = 1
    vertical_robot_axis: int = 2
    horizontal_gain_metres: float = 0.5
    vertical_gain_metres: float = -0.5
    workspace: AxisAlignedWorkspace = AxisAlignedWorkspace((0.35, -0.30, 0.14), (0.82, 0.30, 0.65))
    maximum_speed_metres_per_second: float = 0.25
    minimum_confidence: float = 0.6
    maximum_observation_age_seconds: float = 0.25
    maximum_update_gap_seconds: float = 0.5
    target_ttl_seconds: float = 0.15

    def __post_init__(self) -> None:
        if self.horizontal_robot_axis not in (0, 1, 2) or self.vertical_robot_axis not in (0, 1, 2):
            raise ValueError("robot axes must be 0, 1, or 2")
        if self.horizontal_robot_axis == self.vertical_robot_axis:
            raise ValueError("horizontal and vertical robot axes must differ")
        for name in (
            "horizontal_gain_metres",
            "vertical_gain_metres",
            "maximum_speed_metres_per_second",
            "minimum_confidence",
            "maximum_observation_age_seconds",
            "maximum_update_gap_seconds",
            "target_ttl_seconds",
        ):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.maximum_speed_metres_per_second <= 0:
            raise ValueError("maximum speed must be positive")
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum confidence must be between zero and one")
        if (
            min(
                self.maximum_observation_age_seconds,
                self.maximum_update_gap_seconds,
                self.target_ttl_seconds,
            )
            <= 0
        ):
            raise ValueError("timing limits must be positive")


class WebcamPlanarRetargeter:
    def __init__(self, config: WebcamPlanarConfig | None = None) -> None:
        self.config = config or WebcamPlanarConfig()
        self._anchor_human: tuple[float, float] | None = None
        self._anchor_robot: Pose3D | None = None
        self._last: ArmCartesianTarget | None = None
        self._last_at: TimePoint | None = None
        self._last_frame_at: TimePoint | None = None
        self._person_id: str | None = None
        self._camera_frame: FrameId | None = None
        self._image_size: tuple[int, int] | None = None

    @property
    def is_armed(self) -> bool:
        return self._anchor_human is not None

    def disarm(self) -> None:
        self._anchor_human = None
        self._anchor_robot = None
        self._last = None
        self._last_at = None
        self._last_frame_at = None
        self._person_id = None
        self._camera_frame = None
        self._image_size = None

    def _human(self, pose: HumanPose2D, size: tuple[int, int]) -> tuple[tuple[float, float], float]:
        side = self.config.side.value
        joints = tuple(HumanJoint(f"{side}_{part}") for part in ("shoulder", "elbow", "wrist"))
        points = [pose.get(joint) for joint in joints]
        if any(
            point is None or point.confidence < self.config.minimum_confidence for point in points
        ):
            raise ValueError("selected arm landmarks are missing or uncertain")
        shoulder, _, wrist = points
        assert shoulder is not None and wrist is not None
        width, height = size
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive")
        return (
            (wrist.image_xy[0] - shoulder.image_xy[0]) / width,
            (wrist.image_xy[1] - shoulder.image_xy[1]) / height,
        ), min(point.confidence for point in points if point is not None)

    def engage(
        self, pose: HumanPose2D, *, image_size: tuple[int, int], robot_pose: Pose3D, now: TimePoint
    ) -> None:
        if robot_pose.reference_frame != self.config.robot_base_frame:
            raise ValueError("robot anchor must be in the configured base frame")
        try:
            age = (now - pose.source_header.received_at) / 1e9
        except ClockDomainMismatchError as error:
            raise ValueError("camera and controller clocks differ") from error
        if not 0 <= age <= self.config.maximum_observation_age_seconds:
            raise ValueError("camera observation is stale")
        human, _ = self._human(pose, image_size)
        self._anchor_human = human
        self._anchor_robot = robot_pose
        self._last = ArmCartesianTarget(self.config.side, self.config.tool_frame, robot_pose, 1.0)
        self._last_at = now
        self._last_frame_at = pose.source_header.received_at
        self._person_id = pose.person_id
        self._camera_frame = pose.source_header.frame_id
        self._image_size = image_size

    def update(
        self, pose: HumanPose2D | None, *, now: TimePoint, deadman_pressed: bool
    ) -> SingleArmCommand:
        last = self._last
        if last is None or self._anchor_human is None or self._anchor_robot is None:
            return self._emit(TeleopState.DISARMED, pose, now, None, "clutch-not-engaged")
        reason = None
        if not deadman_pressed:
            reason = "deadman-released"
        elif pose is None:
            reason = "operator-not-detected"
        elif pose.person_id != self._person_id or pose.source_header.frame_id != self._camera_frame:
            reason = "operator-or-camera-changed"
        elif self._image_size is None or self._last_at is None or self._last_frame_at is None:
            reason = "missing-anchor"
        else:
            try:
                age = (now - pose.source_header.received_at) / 1e9
                elapsed = (now - self._last_at) / 1e9
                new_frame = pose.source_header.received_at - self._last_frame_at > 0
                if not new_frame:
                    reason = "duplicate-observation"
                elif not 0 <= age <= self.config.maximum_observation_age_seconds:
                    reason = "stale-observation"
                elif not 0 < elapsed <= self.config.maximum_update_gap_seconds:
                    reason = "controller-update-gap"
            except ClockDomainMismatchError:
                reason = "clock-domain-mismatch"
        if reason is not None:
            return self._emit(TeleopState.HOLD, pose, now, last, reason)
        assert pose is not None and self._image_size is not None and self._last_at is not None
        try:
            human, confidence = self._human(pose, self._image_size)
        except ValueError as error:
            return self._emit(TeleopState.HOLD, pose, now, last, str(error))
        candidate = list(self._anchor_robot.position_metres)
        horizontal = self.config.horizontal_robot_axis
        vertical = self.config.vertical_robot_axis
        candidate[horizontal] += (
            human[0] - self._anchor_human[0]
        ) * self.config.horizontal_gain_metres
        candidate[vertical] += (human[1] - self._anchor_human[1]) * self.config.vertical_gain_metres
        bounded, clamped = self.config.workspace.clamp(tuple(candidate))
        delta = tuple(b - a for a, b in zip(last.pose.position_metres, bounded, strict=True))
        distance = math.sqrt(sum(value * value for value in delta))
        maximum_step = self.config.maximum_speed_metres_per_second * ((now - self._last_at) / 1e9)
        limited = distance > maximum_step
        if limited:
            ratio = maximum_step / distance
            bounded = tuple(
                a + ratio * d for a, d in zip(last.pose.position_metres, delta, strict=True)
            )
        target = ArmCartesianTarget(
            self.config.side,
            self.config.tool_frame,
            Pose3D(self.config.robot_base_frame, bounded, self._anchor_robot.orientation_xyzw),
            confidence,
            clamped or limited,
        )
        self._last = target
        self._last_at = now
        self._last_frame_at = pose.source_header.received_at
        return self._emit(TeleopState.ACTIVE, pose, now, target, None)

    def _emit(
        self,
        state: TeleopState,
        pose: HumanPose2D | None,
        now: TimePoint,
        arm: ArmCartesianTarget | None,
        reason: str | None,
    ) -> SingleArmCommand:
        expiry = TimePoint(now.nanoseconds + round(self.config.target_ttl_seconds * 1e9), now.clock)
        return SingleArmCommand(
            state,
            now,
            expiry,
            pose.source_header if pose else None,
            pose.person_id if pose else self._person_id,
            arm,
            reason,
        )


__all__ = ["WebcamPlanarConfig", "WebcamPlanarRetargeter"]
