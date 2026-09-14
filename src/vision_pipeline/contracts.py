"""Foundational contracts shared by sensor, model, storage, and runtime adapters.

This module deliberately contains no NumPy, camera SDK, GPU, or robotics middleware
dependency. It describes what a measurement means without prescribing how its payload
is represented or processed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from functools import total_ordering
from typing import Generic, TypeVar

_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
PayloadT = TypeVar("PayloadT")


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value or value != value.strip() or "\x00" in value:
        raise ValueError(
            f"{field_name} must be non-empty, have no surrounding whitespace, "
            "and contain no NUL characters"
        )


def _validate_int64(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer")
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise ValueError(f"{field_name} must fit in a signed 64-bit integer")


class ClockKind(StrEnum):
    """The semantics of a clock, separate from its unique domain identity."""

    DEVICE = "device"
    HOST_MONOTONIC = "host_monotonic"
    UTC = "utc"
    PTP = "ptp"
    SIMULATION = "simulation"


@dataclass(frozen=True, slots=True)
class ClockDomain:
    """One uninterrupted clock epoch.

    Two devices have different domains even if both use the same clock kind. A device
    restart or counter reset must also create a new domain identifier.
    """

    domain_id: str
    kind: ClockKind

    def __post_init__(self) -> None:
        _validate_identifier(self.domain_id, "domain_id")
        if not isinstance(self.kind, ClockKind):
            raise TypeError("kind must be a ClockKind")


class ClockDomainMismatchError(ValueError):
    """Raised when timestamps from clocks without a known mapping are compared."""


@total_ordering
@dataclass(frozen=True, slots=True)
class TimePoint:
    """A nanosecond timestamp meaningful only inside its clock domain."""

    nanoseconds: int
    clock: ClockDomain

    def __post_init__(self) -> None:
        _validate_int64(self.nanoseconds, "nanoseconds")
        if not isinstance(self.clock, ClockDomain):
            raise TypeError("clock must be a ClockDomain")

    def _require_same_clock(self, other: TimePoint) -> None:
        if not isinstance(other, TimePoint):
            raise TypeError("timestamps can only be compared with another TimePoint")
        if self.clock != other.clock:
            raise ClockDomainMismatchError(
                f"no clock mapping from {other.clock.domain_id!r} " f"to {self.clock.domain_id!r}"
            )

    def __lt__(self, other: TimePoint) -> bool:
        self._require_same_clock(other)
        return self.nanoseconds < other.nanoseconds

    def __sub__(self, other: TimePoint) -> int:
        """Return elapsed nanoseconds, rejecting unmapped clock domains."""

        self._require_same_clock(other)
        return self.nanoseconds - other.nanoseconds


@dataclass(frozen=True, slots=True)
class FrameId:
    """Opaque identifier for the coordinate frame in which a measurement is expressed."""

    value: str

    def __post_init__(self) -> None:
        _validate_identifier(self.value, "frame_id")


@dataclass(frozen=True, slots=True)
class CalibrationRef:
    """Reference to an immutable, versioned calibration bundle."""

    calibration_id: str
    revision: str

    def __post_init__(self) -> None:
        _validate_identifier(self.calibration_id, "calibration_id")
        _validate_identifier(self.revision, "revision")


class MeasurementKind(StrEnum):
    """Physical measurement families, excluding derived perception results."""

    RGB_IMAGE = "rgb_image"
    DEPTH_IMAGE = "depth_image"
    INFRARED_IMAGE = "infrared_image"
    LIDAR_SCAN = "lidar_scan"
    RADAR_SCAN = "radar_scan"
    IMU = "imu"
    GNSS = "gnss"
    FORCE_TORQUE = "force_torque"
    TACTILE = "tactile"
    JOINT_STATE = "joint_state"
    POSE = "pose"


class ComputeKind(StrEnum):
    """Where a result was computed; this is factual provenance, not a request."""

    UNKNOWN = "unknown"
    SENSOR_DEVICE = "sensor_device"
    HOST_CPU = "host_cpu"
    HOST_GPU = "host_gpu"
    DSP = "dsp"
    NPU = "npu"
    MCU = "mcu"
    REMOTE = "remote"


@dataclass(frozen=True, slots=True)
class ComputePlacement:
    """Compute resource that produced a measurement."""

    kind: ComputeKind
    device_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ComputeKind):
            raise TypeError("kind must be a ComputeKind")
        _validate_identifier(self.device_id, "device_id")


class MemoryKind(StrEnum):
    """Current memory domain of a payload."""

    HOST = "host"
    HOST_PINNED = "host_pinned"
    SHARED = "shared"
    DEVICE_LOCAL = "device_local"
    SENSOR_OWNED = "sensor_owned"
    REMOTE = "remote"


@dataclass(frozen=True, slots=True)
class MemoryPlacement:
    """Where payload bytes currently live, independent of where they were computed."""

    kind: MemoryKind
    device_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MemoryKind):
            raise TypeError("kind must be a MemoryKind")
        if self.device_id is not None:
            _validate_identifier(self.device_id, "device_id")
        if (
            self.kind in {MemoryKind.DEVICE_LOCAL, MemoryKind.SENSOR_OWNED}
            and self.device_id is None
        ):
            raise ValueError(f"{self.kind.value} memory requires a device_id")


@dataclass(frozen=True, slots=True)
class SampleHeader:
    """Immutable meaning and provenance attached to one physical measurement."""

    sample_id: str
    source_id: str
    sequence_number: int
    measurement_kind: MeasurementKind
    captured_at: TimePoint | None
    received_at: TimePoint
    frame_id: FrameId
    calibration: CalibrationRef | None
    producer: str
    produced_on: ComputePlacement

    def __post_init__(self) -> None:
        _validate_identifier(self.sample_id, "sample_id")
        _validate_identifier(self.source_id, "source_id")
        _validate_int64(self.sequence_number, "sequence_number")
        if self.sequence_number < 0:
            raise ValueError("sequence_number must be non-negative")
        if not isinstance(self.measurement_kind, MeasurementKind):
            raise TypeError("measurement_kind must be a MeasurementKind")
        if self.captured_at is not None and not isinstance(self.captured_at, TimePoint):
            raise TypeError("captured_at must be a TimePoint or None")
        if not isinstance(self.received_at, TimePoint):
            raise TypeError("received_at must be a TimePoint")
        if not isinstance(self.frame_id, FrameId):
            raise TypeError("frame_id must be a FrameId")
        if self.calibration is not None and not isinstance(self.calibration, CalibrationRef):
            raise TypeError("calibration must be a CalibrationRef or None")
        _validate_identifier(self.producer, "producer")
        if not isinstance(self.produced_on, ComputePlacement):
            raise TypeError("produced_on must be a ComputePlacement")

        if (
            self.captured_at is not None
            and self.captured_at.clock == self.received_at.clock
            and self.received_at < self.captured_at
        ):
            raise ValueError("received_at cannot precede captured_at in the same clock domain")


@dataclass(frozen=True, slots=True)
class SensorSample(Generic[PayloadT]):
    """A physical measurement with payload storage kept separate from its meaning.

    The envelope is immutable and construction never copies or inspects ``payload``.
    Payload ownership and synchronization will be represented by buffer adapters in a
    later lesson.
    """

    header: SampleHeader
    payload: PayloadT
    payload_memory: MemoryPlacement

    def __post_init__(self) -> None:
        if not isinstance(self.header, SampleHeader):
            raise TypeError("header must be a SampleHeader")
        if self.payload is None:
            raise ValueError("payload must not be None")
        if not isinstance(self.payload_memory, MemoryPlacement):
            raise TypeError("payload_memory must be a MemoryPlacement")
