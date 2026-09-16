"""Robot-independent, expiring bimanual teleoperation target contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from vision_pipeline.contracts import FrameId, SampleHeader, TimePoint
from vision_pipeline.geometry.spatial import Pose3D


class ArmSide(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class TeleopState(StrEnum):
    DISARMED = "disarmed"
    HOLD = "hold"
    ACTIVE = "active"


@dataclass(frozen=True, slots=True)
class ArmCartesianTarget:
    """A bounded position target for one named tool in one robot base frame.

    Orientation is present because the downstream Cartesian controller needs a full
    pose. The initial body-pose retargeter deliberately keeps it at the calibrated
    anchor orientation; it does not infer hand orientation from a wrist point.
    """

    side: ArmSide
    controlled_tool_frame: FrameId
    pose: Pose3D
    confidence: float
    limited: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.side, ArmSide):
            raise TypeError("side must be an ArmSide")
        if not isinstance(self.controlled_tool_frame, FrameId):
            raise TypeError("controlled_tool_frame must be a FrameId")
        if not isinstance(self.pose, Pose3D):
            raise TypeError("pose must be a Pose3D")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, int | float):
            raise TypeError("confidence must be a number")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and between zero and one")
        if not isinstance(self.limited, bool):
            raise TypeError("limited must be a boolean")


@dataclass(frozen=True, slots=True)
class BimanualCartesianTarget:
    """One atomic command decision for both arms.

    A robot-side controller must reject the target after ``expires_at`` and implement
    its own real-time interpolation, IK, collision checking, and stop behavior.
    """

    command_sequence: int
    state: TeleopState
    generated_at: TimePoint
    expires_at: TimePoint
    source_header: SampleHeader | None
    operator_id: str | None
    left: ArmCartesianTarget | None
    right: ArmCartesianTarget | None
    reason: str | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.command_sequence, bool)
            or not isinstance(self.command_sequence, int)
            or self.command_sequence < 0
        ):
            raise ValueError("command_sequence must be a non-negative integer")
        if not isinstance(self.state, TeleopState):
            raise TypeError("state must be a TeleopState")
        if not isinstance(self.generated_at, TimePoint) or not isinstance(
            self.expires_at, TimePoint
        ):
            raise TypeError("generated_at and expires_at must be TimePoint values")
        if self.expires_at.clock != self.generated_at.clock:
            raise ValueError("generated_at and expires_at must use the same clock")
        if self.expires_at.nanoseconds <= self.generated_at.nanoseconds:
            raise ValueError("expires_at must be after generated_at")
        if self.source_header is not None and not isinstance(self.source_header, SampleHeader):
            raise TypeError("source_header must be a SampleHeader or None")
        if self.operator_id is not None and (
            not isinstance(self.operator_id, str) or not self.operator_id.strip()
        ):
            raise ValueError("operator_id must be a non-empty string or None")
        if self.left is not None and self.left.side is not ArmSide.LEFT:
            raise ValueError("left target must have side=left")
        if self.right is not None and self.right.side is not ArmSide.RIGHT:
            raise ValueError("right target must have side=right")
        if self.state is TeleopState.DISARMED and (self.left is not None or self.right is not None):
            raise ValueError("a disarmed command cannot contain arm targets")
        if self.state is not TeleopState.DISARMED and (self.left is None or self.right is None):
            raise ValueError("active and hold commands require both arm targets")
        if self.state is TeleopState.ACTIVE and self.reason is not None:
            raise ValueError("an active command cannot have a hold reason")
        if self.state is not TeleopState.ACTIVE and (
            not isinstance(self.reason, str) or not self.reason.strip()
        ):
            raise ValueError("non-active commands require a reason")


class BimanualTargetSink(Protocol):
    """Non-real-time boundary to a robot-side safety/controller process."""

    def send(self, target: BimanualCartesianTarget) -> None:
        """Send one atomic, expiring target without blocking a control loop."""


__all__ = [
    "ArmCartesianTarget",
    "ArmSide",
    "BimanualCartesianTarget",
    "BimanualTargetSink",
    "TeleopState",
]
