"""Expiring, clutch-relative target generation for one selected manipulator arm."""

from __future__ import annotations

import math
from dataclasses import dataclass

from vision_pipeline.contracts import ClockDomainMismatchError, SampleHeader, TimePoint
from vision_pipeline.geometry.spatial import Pose3D
from vision_pipeline.perception.pose import HumanPose3D
from vision_pipeline.teleoperation.contracts import ArmCartesianTarget, TeleopState
from vision_pipeline.teleoperation.retargeting import (
    ArmRetargetingConfig,
    BimanualRetargeter,
    _ArmAnchor,
)


@dataclass(frozen=True, slots=True)
class SingleArmCommand:
    state: TeleopState
    generated_at: TimePoint
    expires_at: TimePoint
    source_header: SampleHeader | None
    operator_id: str | None
    arm: ArmCartesianTarget | None
    reason: str | None


class SingleArmRetargeter:
    """Select one human joint chain and produce bounded, position-only TCP targets."""

    def __init__(
        self,
        config: ArmRetargetingConfig,
        *,
        maximum_observation_age_seconds: float = 0.15,
        target_ttl_seconds: float = 0.10,
        maximum_update_gap_seconds: float = 0.5,
    ) -> None:
        for value in (
            maximum_observation_age_seconds,
            target_ttl_seconds,
            maximum_update_gap_seconds,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("timing limits must be positive")
        self.config = config
        self.maximum_observation_age_seconds = maximum_observation_age_seconds
        self.target_ttl_seconds = target_ttl_seconds
        self.maximum_update_gap_seconds = maximum_update_gap_seconds
        self._anchor: _ArmAnchor | None = None
        self._operator_id: str | None = None
        self._last_target: ArmCartesianTarget | None = None
        self._last_command_at: TimePoint | None = None
        self._last_observation_at: TimePoint | None = None

    def engage(self, pose: HumanPose3D, *, robot_pose: Pose3D, now: TimePoint) -> None:
        if pose.reference_frame != self.config.operator_frame:
            raise ValueError("operator frame does not match configuration")
        if robot_pose.reference_frame != self.config.robot_base_frame:
            raise ValueError("robot pose must use the configured base frame")
        try:
            age = (now - pose.source_header.received_at) / 1e9
        except ClockDomainMismatchError as error:
            raise ValueError("operator observation clock does not match controller") from error
        if age < 0 or age > self.maximum_observation_age_seconds:
            raise ValueError("operator observation is stale or from the future")
        relative, _ = BimanualRetargeter._operator_arm(pose, self.config)
        self._anchor = _ArmAnchor(relative, robot_pose)
        self._operator_id = pose.person_id
        self._last_target = BimanualRetargeter._anchor_target(self.config, robot_pose)
        self._last_command_at = now
        self._last_observation_at = None

    def disarm(self) -> None:
        self._anchor = None
        self._operator_id = None
        self._last_target = None
        self._last_command_at = None
        self._last_observation_at = None

    def update(
        self, pose: HumanPose3D, *, now: TimePoint, deadman_pressed: bool
    ) -> SingleArmCommand:
        if not isinstance(deadman_pressed, bool):
            raise TypeError("deadman_pressed must be a boolean")
        if self._anchor is None or self._last_target is None or self._last_command_at is None:
            return self._emit(TeleopState.DISARMED, pose, now, None, "clutch-not-engaged")
        reason: str | None = None
        if not deadman_pressed:
            reason = "deadman-released"
        elif pose.person_id != self._operator_id:
            reason = "operator-identity-changed"
        elif pose.reference_frame != self.config.operator_frame:
            reason = "operator-frame-changed"
        else:
            try:
                age = (now - pose.source_header.received_at) / 1e9
                elapsed = (now - self._last_command_at) / 1e9
                if self._last_observation_at is not None and (
                    pose.source_header.received_at - self._last_observation_at <= 0
                ):
                    reason = "duplicate-or-out-of-order-observation"
                elif age < 0 or age > self.maximum_observation_age_seconds:
                    reason = "stale-observation"
                elif elapsed <= 0 or elapsed > self.maximum_update_gap_seconds:
                    reason = "controller-update-gap"
            except ClockDomainMismatchError:
                reason = "clock-domain-mismatch"
        if reason is not None:
            return self._emit(TeleopState.HOLD, pose, now, self._last_target, reason)
        try:
            relative, confidence = BimanualRetargeter._operator_arm(pose, self.config)
        except ValueError as error:
            return self._emit(TeleopState.HOLD, pose, now, self._last_target, str(error))
        elapsed = (now - self._last_command_at) / 1e9
        target = BimanualRetargeter._retarget_arm(
            self.config,
            self._anchor,
            relative,
            confidence,
            self._last_target,
            elapsed,
        )
        self._last_target = target
        self._last_command_at = now
        self._last_observation_at = pose.source_header.received_at
        return self._emit(TeleopState.ACTIVE, pose, now, target, None)

    def _emit(
        self,
        state: TeleopState,
        pose: HumanPose3D,
        now: TimePoint,
        arm: ArmCartesianTarget | None,
        reason: str | None,
    ) -> SingleArmCommand:
        expiry = TimePoint(now.nanoseconds + round(self.target_ttl_seconds * 1e9), now.clock)
        return SingleArmCommand(state, now, expiry, pose.source_header, pose.person_id, arm, reason)


__all__ = ["SingleArmCommand", "SingleArmRetargeter"]
