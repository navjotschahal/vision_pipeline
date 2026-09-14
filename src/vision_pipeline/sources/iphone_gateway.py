"""Decode and align measurements from the native iPhone sensor gateway."""

from __future__ import annotations

import json
import math
import struct
import threading
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import BinaryIO, cast

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.geometry.camera import PinholeIntrinsics

Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]

_MAX_HEADER_BYTES = 1_048_576
_MAX_PAYLOAD_BYTES = 64 * 1_048_576


class IphoneGatewayProtocolError(ValueError):
    """The peer sent a malformed or unsupported sensor packet."""


@dataclass(frozen=True, slots=True)
class IphoneImuSample:
    """Raw Core Motion data in Apple's device-motion coordinate convention."""

    source_id: str
    session_id: str
    sequence_number: int
    device_timestamp_ns: int
    attitude_xyzw: tuple[float, float, float, float]
    gravity_compensated_acceleration_mps2: tuple[float, float, float]
    angular_velocity_radps: tuple[float, float, float]
    gravity_mps2: tuple[float, float, float]

    def to_wire_dict(self) -> dict[str, object]:
        return {
            "schema": "iphone_imu.v1",
            "source_id": self.source_id,
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "device_timestamp_ns": self.device_timestamp_ns,
            "attitude_quaternion": _xyzw_dict(self.attitude_xyzw),
            "gravity_compensated_acceleration_mps2": _xyz_dict(
                self.gravity_compensated_acceleration_mps2
            ),
            "angular_velocity_radps": _xyz_dict(self.angular_velocity_radps),
            "gravity_mps2": _xyz_dict(self.gravity_mps2),
        }


@dataclass(frozen=True, slots=True)
class IphoneRGBDFrame:
    """Synchronized RGB-D frame with ARKit's visual-inertial camera pose.

    ARKit camera coordinates are +X right, +Y up, with the camera looking
    along -Z. Depth uses the optical convention (+X right, +Y down, +Z
    forward), so optical_to_world includes the required Y and Z axis flips.
    """

    source_id: str
    session_id: str
    sequence_number: int
    device_timestamp_ns: int
    tracking_state: str
    rgb_orientation: str
    rgb_jpeg: bytes
    rgb_width: int
    rgb_height: int
    depth_metres: Float32Array
    confidence: UInt8Array | None
    camera_intrinsics_rgb: PinholeIntrinsics
    camera_to_world_arkit: Float32Array

    @property
    def depth_intrinsics(self) -> PinholeIntrinsics:
        """Scale RGB-resolution ARKit intrinsics to the aligned depth grid."""

        depth_height, depth_width = self.depth_metres.shape
        scale_x = depth_width / self.rgb_width
        scale_y = depth_height / self.rgb_height
        source = self.camera_intrinsics_rgb
        return PinholeIntrinsics(
            width=depth_width,
            height=depth_height,
            fx=source.fx * scale_x,
            fy=source.fy * scale_y,
            cx=source.cx * scale_x,
            cy=source.cy * scale_y,
            calibration_id="arkit-live-intrinsics",
        )

    @property
    def optical_to_world(self) -> Float32Array:
        optical_to_arkit_camera = np.diag((1.0, -1.0, -1.0, 1.0)).astype(np.float32)
        return cast(Float32Array, self.camera_to_world_arkit @ optical_to_arkit_camera)

    @property
    def camera_position_world_metres(self) -> tuple[float, float, float]:
        translation = self.camera_to_world_arkit[:3, 3]
        return (float(translation[0]), float(translation[1]), float(translation[2]))

    def world_points(
        self,
        *,
        sampling_stride: int = 2,
        min_depth_metres: float = 0.05,
        max_depth_metres: float = 8.0,
        minimum_confidence: int = 1,
    ) -> Float32Array:
        """Deproject depth and motion-compensate points into ARKit world."""

        optical_points = self.optical_points(
            sampling_stride=sampling_stride,
            min_depth_metres=min_depth_metres,
            max_depth_metres=max_depth_metres,
            minimum_confidence=minimum_confidence,
        )
        if len(optical_points) == 0:
            return optical_points
        optical_h = np.column_stack(
            (optical_points, np.ones(len(optical_points), dtype=np.float32))
        )
        world_h = optical_h @ self.optical_to_world.T
        return np.ascontiguousarray(world_h[:, :3], dtype=np.float32)

    def optical_points(
        self,
        *,
        sampling_stride: int = 2,
        min_depth_metres: float = 0.05,
        max_depth_metres: float = 8.0,
        minimum_confidence: int = 1,
    ) -> Float32Array:
        """Deproject one depth image in its unaccumulated camera optical frame."""

        if isinstance(sampling_stride, bool) or not isinstance(sampling_stride, int):
            raise TypeError("sampling_stride must be an integer")
        if sampling_stride <= 0:
            raise ValueError("sampling_stride must be positive")
        if isinstance(minimum_confidence, bool) or not isinstance(minimum_confidence, int):
            raise TypeError("minimum_confidence must be an integer")
        if not 0 <= minimum_confidence <= 2:
            raise ValueError("minimum_confidence must be between 0 and 2")
        if not (
            math.isfinite(min_depth_metres)
            and math.isfinite(max_depth_metres)
            and 0 <= min_depth_metres < max_depth_metres
        ):
            raise ValueError("depth limits must be finite, non-negative, and increasing")

        depth = self.depth_metres[::sampling_stride, ::sampling_stride]
        rows = np.arange(0, self.depth_metres.shape[0], sampling_stride, dtype=np.float32)
        columns = np.arange(0, self.depth_metres.shape[1], sampling_stride, dtype=np.float32)
        pixel_y, pixel_x = np.meshgrid(rows, columns, indexing="ij")
        valid = (
            np.isfinite(depth)
            & (depth >= min_depth_metres)
            & (depth <= max_depth_metres)
        )
        if self.confidence is not None:
            valid &= self.confidence[::sampling_stride, ::sampling_stride] >= minimum_confidence

        z = depth[valid]
        if z.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        intrinsics = self.depth_intrinsics
        x = (pixel_x[valid] - intrinsics.cx) * z / intrinsics.fx
        y = (pixel_y[valid] - intrinsics.cy) * z / intrinsics.fy
        return np.ascontiguousarray(np.column_stack((x, y, z)), dtype=np.float32)

    def metadata_dict(self) -> dict[str, object]:
        """Return metadata without duplicating image and depth payloads."""

        intrinsics = self.camera_intrinsics_rgb
        return {
            "schema": "iphone_rgbd.v1",
            "source_id": self.source_id,
            "session_id": self.session_id,
            "sequence_number": self.sequence_number,
            "device_timestamp_ns": self.device_timestamp_ns,
            "tracking_state": self.tracking_state,
            "rgb_orientation": self.rgb_orientation,
            "rgb_width": self.rgb_width,
            "rgb_height": self.rgb_height,
            "depth_width": int(self.depth_metres.shape[1]),
            "depth_height": int(self.depth_metres.shape[0]),
            "camera_intrinsics": {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
            },
            "camera_transform_column_major": self.camera_to_world_arkit.reshape(
                -1, order="F"
            ).tolist(),
        }


@dataclass(frozen=True, slots=True)
class ContinuityObservation:
    """Sequence and timestamp checks for one newly received sample."""

    new_session: bool
    dropped: int
    duplicate_or_reordered: bool
    timestamp_regressed: bool


class StreamContinuityMonitor:
    """Track gaps and monotonic device time within each session."""

    def __init__(self) -> None:
        self._session_id: str | None = None
        self._sequence_number: int | None = None
        self._timestamp_ns: int | None = None

    def observe(
        self, session_id: str, sequence_number: int, device_timestamp_ns: int
    ) -> ContinuityObservation:
        new_session = session_id != self._session_id
        if new_session:
            self._session_id = session_id
            self._sequence_number = sequence_number
            self._timestamp_ns = device_timestamp_ns
            return ContinuityObservation(True, 0, False, False)

        assert self._sequence_number is not None
        assert self._timestamp_ns is not None
        reordered = sequence_number <= self._sequence_number
        dropped = max(0, sequence_number - self._sequence_number - 1)
        regressed = device_timestamp_ns <= self._timestamp_ns
        if not reordered:
            self._sequence_number = sequence_number
        if device_timestamp_ns > self._timestamp_ns:
            self._timestamp_ns = device_timestamp_ns
        return ContinuityObservation(False, dropped, reordered, regressed)


class IphoneImuTimeline:
    """Thread-safe recent IMU buffer for camera timestamp association."""

    def __init__(self, capacity: int = 2_000) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self._samples: deque[IphoneImuSample] = deque(maxlen=capacity)
        self._lock = threading.Lock()

    def append(self, sample: IphoneImuSample) -> None:
        with self._lock:
            self._samples.append(sample)

    def nearest(self, frame: IphoneRGBDFrame) -> tuple[IphoneImuSample | None, int | None]:
        """Return nearest same-session IMU and signed IMU-minus-frame delta."""

        with self._lock:
            candidates = tuple(
                sample
                for sample in self._samples
                if sample.source_id == frame.source_id and sample.session_id == frame.session_id
            )
        if not candidates:
            return None, None
        nearest = min(
            candidates,
            key=lambda sample: abs(sample.device_timestamp_ns - frame.device_timestamp_ns),
        )
        return nearest, nearest.device_timestamp_ns - frame.device_timestamp_ns


def parse_imu_line(line: bytes | str) -> IphoneImuSample:
    """Parse one newline-delimited iphone_imu.v1 measurement."""

    try:
        value = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise IphoneGatewayProtocolError(f"invalid IMU JSON: {error}") from error
    document = _mapping(value, "IMU sample")
    if document.get("schema") != "iphone_imu.v1":
        raise IphoneGatewayProtocolError("expected iphone_imu.v1 object")
    try:
        return IphoneImuSample(
            source_id=_nonempty_string(document["source_id"], "source_id"),
            session_id=_nonempty_string(document["session_id"], "session_id"),
            sequence_number=_nonnegative_int(document["sequence_number"], "sequence_number"),
            device_timestamp_ns=_nonnegative_int(
                document["device_timestamp_ns"], "device_timestamp_ns"
            ),
            attitude_xyzw=_quaternion(document["attitude_quaternion"]),
            gravity_compensated_acceleration_mps2=_vector3(
                document["gravity_compensated_acceleration_mps2"],
                "gravity_compensated_acceleration_mps2",
            ),
            angular_velocity_radps=_vector3(
                document["angular_velocity_radps"], "angular_velocity_radps"
            ),
            gravity_mps2=_vector3(document["gravity_mps2"], "gravity_mps2"),
        )
    except KeyError as error:
        raise IphoneGatewayProtocolError(f"missing IMU field: {error.args[0]}") from error
    except (TypeError, ValueError) as error:
        raise IphoneGatewayProtocolError(f"invalid IMU field: {error}") from error


def read_rgbd_frame(stream: BinaryIO) -> IphoneRGBDFrame:
    """Read one length-prefixed iphone_rgbd.v1 packet from a byte stream."""

    (header_length,) = struct.unpack(">I", _read_exact(stream, 4))
    if not 0 < header_length <= _MAX_HEADER_BYTES:
        raise IphoneGatewayProtocolError(f"invalid RGB-D header length: {header_length}")
    try:
        value = json.loads(_read_exact(stream, header_length))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise IphoneGatewayProtocolError(f"invalid RGB-D header JSON: {error}") from error
    header = _mapping(value, "RGB-D header")
    if header.get("schema") != "iphone_rgbd.v1":
        raise IphoneGatewayProtocolError("expected iphone_rgbd.v1 header")

    try:
        if header["rgb_encoding"] != "jpeg":
            raise IphoneGatewayProtocolError("only JPEG RGB payloads are supported")
        if header["depth_encoding"] != "float32_le_metres":
            raise IphoneGatewayProtocolError("only Float32 little-endian depth is supported")
        rgb_width = _positive_int(header["rgb_width"], "rgb_width")
        rgb_height = _positive_int(header["rgb_height"], "rgb_height")
        depth_width = _positive_int(header["depth_width"], "depth_width")
        depth_height = _positive_int(header["depth_height"], "depth_height")
        payload = _mapping(header["payload"], "payload")
        rgb_count = _nonnegative_int(payload["rgb_bytes"], "rgb_bytes")
        depth_count = _nonnegative_int(payload["depth_bytes"], "depth_bytes")
        confidence_count = _nonnegative_int(
            payload["confidence_bytes"], "confidence_bytes"
        )
        if depth_count != depth_width * depth_height * 4:
            raise IphoneGatewayProtocolError("depth byte count does not match dimensions")
        if confidence_count not in (0, depth_width * depth_height):
            raise IphoneGatewayProtocolError("confidence byte count does not match dimensions")
        if rgb_count + depth_count + confidence_count > _MAX_PAYLOAD_BYTES:
            raise IphoneGatewayProtocolError("RGB-D payload exceeds the safety limit")

        rgb_jpeg = _read_exact(stream, rgb_count)
        depth = np.frombuffer(_read_exact(stream, depth_count), dtype="<f4").reshape(
            depth_height, depth_width
        )
        confidence: UInt8Array | None = None
        if confidence_count:
            confidence = np.frombuffer(
                _read_exact(stream, confidence_count), dtype=np.uint8
            ).reshape(depth_height, depth_width)

        intrinsics_values = _finite_float_list(
            header["camera_intrinsics_column_major"], 9, "camera intrinsics"
        )
        intrinsics = np.asarray(intrinsics_values, dtype=np.float32).reshape(
            (3, 3), order="F"
        )
        transform_values = _finite_float_list(
            header["camera_transform_column_major"], 16, "camera transform"
        )
        camera_to_world = np.asarray(transform_values, dtype=np.float32).reshape(
            (4, 4), order="F"
        )
        return IphoneRGBDFrame(
            source_id=_nonempty_string(header["source_id"], "source_id"),
            session_id=_nonempty_string(header["session_id"], "session_id"),
            sequence_number=_nonnegative_int(header["sequence_number"], "sequence_number"),
            device_timestamp_ns=_nonnegative_int(
                header["device_timestamp_ns"], "device_timestamp_ns"
            ),
            tracking_state=_nonempty_string(header["tracking_state"], "tracking_state"),
            rgb_orientation=_nonempty_string(
                header.get("rgb_orientation", "camera_buffer_native"), "rgb_orientation"
            ),
            rgb_jpeg=rgb_jpeg,
            rgb_width=rgb_width,
            rgb_height=rgb_height,
            depth_metres=np.array(depth, dtype=np.float32, copy=True),
            confidence=(
                np.array(confidence, dtype=np.uint8, copy=True)
                if confidence is not None
                else None
            ),
            camera_intrinsics_rgb=PinholeIntrinsics(
                width=rgb_width,
                height=rgb_height,
                fx=float(intrinsics[0, 0]),
                fy=float(intrinsics[1, 1]),
                cx=float(intrinsics[0, 2]),
                cy=float(intrinsics[1, 2]),
                calibration_id="arkit-live-intrinsics",
            ),
            camera_to_world_arkit=np.array(camera_to_world, dtype=np.float32, copy=True),
        )
    except KeyError as error:
        raise IphoneGatewayProtocolError(
            f"missing RGB-D header field: {error.args[0]}"
        ) from error
    except (TypeError, ValueError) as error:
        if isinstance(error, IphoneGatewayProtocolError):
            raise
        raise IphoneGatewayProtocolError(f"invalid RGB-D header field: {error}") from error


def _read_exact(stream: BinaryIO, byte_count: int) -> bytes:
    chunks: list[bytes] = []
    remaining = byte_count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("RGB-D stream ended mid-packet")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _finite_float_list(value: object, length: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"{name} must contain {length} values")
    return tuple(_finite_float(item, name) for item in value)


def _vector3(value: object, name: str) -> tuple[float, float, float]:
    document = _mapping(value, name)
    result = tuple(_finite_float(document[axis], name) for axis in ("x", "y", "z"))
    return cast(tuple[float, float, float], result)


def _quaternion(value: object) -> tuple[float, float, float, float]:
    document = _mapping(value, "attitude_quaternion")
    result = tuple(
        _finite_float(document[axis], "attitude_quaternion")
        for axis in ("x", "y", "z", "w")
    )
    return cast(tuple[float, float, float, float], result)


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} values must be numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} values must be finite")
    return result


def _xyz_dict(value: tuple[float, float, float]) -> dict[str, float]:
    return {"x": value[0], "y": value[1], "z": value[2]}


def _xyzw_dict(value: tuple[float, float, float, float]) -> dict[str, float]:
    return {"x": value[0], "y": value[1], "z": value[2], "w": value[3]}


__all__ = [
    "ContinuityObservation",
    "IphoneGatewayProtocolError",
    "IphoneImuSample",
    "IphoneImuTimeline",
    "IphoneRGBDFrame",
    "StreamContinuityMonitor",
    "parse_imu_line",
    "read_rgbd_frame",
]
