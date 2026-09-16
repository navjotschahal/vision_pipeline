"""Optional Intel RealSense RGB-D source backed by ``pyrealsense2``.

The vendor SDK is imported only when :meth:`RealSenseSource.open` is called. Frames are
copied out of SDK-owned buffers before this adapter advertises host-owned memory.
"""

from __future__ import annotations

import importlib
import math
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self, cast

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import (
    CalibrationRef,
    ClockDomain,
    ClockKind,
    ComputeKind,
    ComputePlacement,
    FrameId,
    MeasurementKind,
    MemoryKind,
    MemoryPlacement,
    SampleHeader,
    SensorSample,
    TimePoint,
)
from vision_pipeline.geometry.camera import PinholeIntrinsics
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.rgbd import AlignedRgbdFrame, DepthFrame

SdkLoader = Callable[[], Any]
ClockReader = Callable[[], int]


class RealSenseError(RuntimeError):
    """Base class for RealSense adapter failures."""


class RealSenseDependencyError(RealSenseError):
    """The optional Python SDK is not installed."""


class RealSenseOpenError(RealSenseError):
    """The configured RealSense device or stream profile could not be opened."""


class RealSenseReadError(RealSenseError):
    """An open RealSense source failed to return a valid synchronized frame."""


@dataclass(frozen=True, slots=True)
class RealSenseStreamMode:
    width: int
    height: int
    fps: int
    pixel_format: str

    def __post_init__(self) -> None:
        for name in ("width", "height", "fps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(self.pixel_format, str) or not self.pixel_format.strip():
            raise ValueError("pixel_format must be a non-empty string")


@dataclass(frozen=True, slots=True)
class RealSenseIntrinsics:
    """SDK-reported intrinsics, retaining distortion omitted by the pinhole view."""

    pinhole: PinholeIntrinsics
    distortion_model: str
    distortion_coefficients: tuple[float, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pinhole, PinholeIntrinsics):
            raise TypeError("pinhole must be PinholeIntrinsics")
        if not isinstance(self.distortion_model, str) or not self.distortion_model.strip():
            raise ValueError("distortion_model must be a non-empty string")
        if not isinstance(self.distortion_coefficients, tuple) or not all(
            isinstance(value, int | float) and math.isfinite(value)
            for value in self.distortion_coefficients
        ):
            raise ValueError("distortion_coefficients must be a tuple of finite numbers")


@dataclass(frozen=True, slots=True)
class RealSenseStreamInfo:
    mode: RealSenseStreamMode
    intrinsics: RealSenseIntrinsics


@dataclass(frozen=True, slots=True)
class RealSenseInfo:
    """Requested and active device facts captured after profile negotiation."""

    sdk_version: str
    device_name: str
    serial_number: str
    firmware_version: str | None
    usb_type_descriptor: str | None
    requested_color: RealSenseStreamMode
    requested_depth: RealSenseStreamMode
    actual_color: RealSenseStreamInfo
    actual_depth: RealSenseStreamInfo
    depth_scale_metres: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.sdk_version, "sdk_version"),
            (self.device_name, "device_name"),
            (self.serial_number, "serial_number"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not math.isfinite(self.depth_scale_metres) or self.depth_scale_metres <= 0:
            raise ValueError("depth_scale_metres must be positive and finite")


@dataclass(frozen=True, slots=True)
class RealSenseConfig:
    """Raw RealSense profile request and stable output identities."""

    serial_number: str | None = None
    color_width: int = 640
    color_height: int = 480
    color_fps: int = 30
    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30
    color_source_id: str | None = None
    aligned_depth_source_id: str | None = None
    color_frame_id: FrameId = FrameId("realsense_color_optical")
    calibration: CalibrationRef | None = None
    timeout_ms: int = 5_000

    def __post_init__(self) -> None:
        if self.serial_number is not None and (
            not isinstance(self.serial_number, str) or not self.serial_number.strip()
        ):
            raise ValueError("serial_number must be a non-empty string or None")
        for name in (
            "color_width",
            "color_height",
            "color_fps",
            "depth_width",
            "depth_height",
            "depth_fps",
            "timeout_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("color_source_id", "aligned_depth_source_id"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be a non-empty string or None")
        if not isinstance(self.color_frame_id, FrameId):
            raise TypeError("color_frame_id must be a FrameId")
        if self.calibration is not None and not isinstance(self.calibration, CalibrationRef):
            raise TypeError("calibration must be a CalibrationRef or None")


def _load_sdk() -> Any:
    try:
        return importlib.import_module("pyrealsense2")
    except ModuleNotFoundError as error:
        raise RealSenseDependencyError(
            "pyrealsense2 is required for RealSense capture; install the optional "
            "RealSense dependencies for this host platform"
        ) from error


def _enum_name(value: object) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name
    rendered = str(value)
    return rendered.rsplit(".", maxsplit=1)[-1]


def _video_profile(profile: Any) -> Any:
    converter = getattr(profile, "as_video_stream_profile", None)
    return converter() if callable(converter) else profile


def _intrinsics_object(profile: Any) -> Any:
    video = _video_profile(profile)
    getter = getattr(video, "get_intrinsics", None)
    if callable(getter):
        return getter()
    return video.intrinsics


def _reported_intrinsics(profile: Any) -> RealSenseIntrinsics:
    values = _intrinsics_object(profile)
    coefficients = tuple(float(value) for value in values.coeffs)
    return RealSenseIntrinsics(
        pinhole=PinholeIntrinsics(
            width=int(values.width),
            height=int(values.height),
            fx=float(values.fx),
            fy=float(values.fy),
            cx=float(values.ppx),
            cy=float(values.ppy),
            calibration_id=None,
        ),
        distortion_model=_enum_name(values.model),
        distortion_coefficients=coefficients,
    )


def _stream_info(profile: Any) -> RealSenseStreamInfo:
    video = _video_profile(profile)
    mode = RealSenseStreamMode(
        width=int(video.width()),
        height=int(video.height()),
        fps=int(video.fps()),
        pixel_format=_enum_name(video.format()),
    )
    intrinsics = _reported_intrinsics(video)
    if (mode.width, mode.height) != (intrinsics.pinhole.width, intrinsics.pinhole.height):
        raise ValueError("stream profile dimensions and reported intrinsics differ")
    return RealSenseStreamInfo(mode=mode, intrinsics=intrinsics)


def _frame_profile(frame: Any) -> Any:
    profile = getattr(frame, "profile", None)
    if profile is not None:
        return profile
    return frame.get_profile()


def _device_info(device: Any, sdk: Any, field: str) -> str | None:
    key = getattr(sdk.camera_info, field, None)
    if key is None:
        return None
    supports = getattr(device, "supports", None)
    if callable(supports) and not supports(key):
        return None
    try:
        value = str(device.get_info(key))
    except Exception:
        return None
    return value if value.strip() else None


class RealSenseSource:
    """Pull synchronized BGR and metric depth registered to the color frame."""

    def __init__(
        self,
        config: RealSenseConfig = RealSenseConfig(),
        *,
        sdk_loader: SdkLoader = _load_sdk,
        clock_ns: ClockReader = time.monotonic_ns,
        host_clock: ClockDomain | None = None,
        run_id: str | None = None,
    ) -> None:
        if not isinstance(config, RealSenseConfig):
            raise TypeError("config must be a RealSenseConfig")
        resolved_run_id = run_id or uuid.uuid4().hex
        if not isinstance(resolved_run_id, str) or not resolved_run_id.strip():
            raise ValueError("run_id must be a non-empty string")
        self._config = config
        self._sdk_loader = sdk_loader
        self._clock_ns = clock_ns
        self._run_id = resolved_run_id
        self._host_clock = host_clock or ClockDomain(
            f"host/monotonic/process-{os.getpid()}/run-{resolved_run_id}",
            ClockKind.HOST_MONOTONIC,
        )
        self._sdk: Any | None = None
        self._pipeline: Any | None = None
        self._align: Any | None = None
        self._info: RealSenseInfo | None = None
        self._device_clocks: dict[str, ClockDomain] = {}
        self._frameset_sequence = 0

    @property
    def is_open(self) -> bool:
        return self._pipeline is not None

    @property
    def info(self) -> RealSenseInfo:
        if self._info is None:
            raise RealSenseOpenError("RealSense information is unavailable before open()")
        return self._info

    def open(self) -> None:
        if self.is_open:
            raise RealSenseOpenError("RealSense source is already open")
        sdk = self._sdk_loader()
        pipeline = sdk.pipeline()
        request = sdk.config()
        if self._config.serial_number is not None:
            request.enable_device(self._config.serial_number)
        request.enable_stream(
            sdk.stream.color,
            self._config.color_width,
            self._config.color_height,
            sdk.format.bgr8,
            self._config.color_fps,
        )
        request.enable_stream(
            sdk.stream.depth,
            self._config.depth_width,
            self._config.depth_height,
            sdk.format.z16,
            self._config.depth_fps,
        )

        started = False
        try:
            active_profile = pipeline.start(request)
            started = True
            device = active_profile.get_device()
            serial = _device_info(device, sdk, "serial_number") or self._config.serial_number
            if serial is None:
                raise ValueError("device did not report a serial number")
            device_name = _device_info(device, sdk, "name") or "Intel RealSense"
            depth_scale = float(device.first_depth_sensor().get_depth_scale())
            if not math.isfinite(depth_scale) or depth_scale <= 0:
                raise ValueError("device reported an invalid depth scale")
            color_profile = active_profile.get_stream(sdk.stream.color)
            depth_profile = active_profile.get_stream(sdk.stream.depth)
            sdk_version = str(getattr(sdk, "__version__", "unknown"))
            if not sdk_version.strip():
                sdk_version = "unknown"
            info = RealSenseInfo(
                sdk_version=sdk_version,
                device_name=device_name,
                serial_number=serial,
                firmware_version=_device_info(device, sdk, "firmware_version"),
                usb_type_descriptor=_device_info(device, sdk, "usb_type_descriptor"),
                requested_color=RealSenseStreamMode(
                    self._config.color_width,
                    self._config.color_height,
                    self._config.color_fps,
                    "bgr8",
                ),
                requested_depth=RealSenseStreamMode(
                    self._config.depth_width,
                    self._config.depth_height,
                    self._config.depth_fps,
                    "z16",
                ),
                actual_color=_stream_info(color_profile),
                actual_depth=_stream_info(depth_profile),
                depth_scale_metres=depth_scale,
            )
            align = sdk.align(sdk.stream.color)
        except Exception as error:
            if started:
                try:
                    pipeline.stop()
                except Exception:
                    pass
            raise RealSenseOpenError(
                "could not start the configured RealSense RGB-D streams"
            ) from error

        self._sdk = sdk
        self._pipeline = pipeline
        self._align = align
        self._info = info
        self._device_clocks.clear()
        self._frameset_sequence = 0

    def read(self) -> AlignedRgbdFrame:
        pipeline = self._pipeline
        align = self._align
        if pipeline is None or align is None:
            raise RealSenseReadError("open() must be called before read()")

        try:
            frames = pipeline.wait_for_frames(self._config.timeout_ms)
            received_at = TimePoint(self._clock_ns(), self._host_clock)
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                raise RealSenseReadError(
                    "aligned frameset did not contain both color and depth"
                )

            color_view = np.asanyarray(color_frame.get_data())
            if color_view.dtype != np.dtype(np.uint8):
                raise RealSenseReadError("BGR color frame did not use uint8 elements")
            if color_view.ndim != 3 or color_view.shape[2] != 3:
                raise RealSenseReadError("BGR color frame did not have HxWx3 shape")
            color_pixels = cast(
                NDArray[np.uint8],
                np.array(color_view, dtype=np.uint8, order="C", copy=True),
            )

            depth_view = np.asanyarray(depth_frame.get_data())
            if depth_view.dtype != np.dtype(np.uint16) or depth_view.ndim != 2:
                raise RealSenseReadError("aligned Z16 depth frame did not have uint16 HxW data")
            raw_depth = cast(
                NDArray[np.uint16],
                np.array(depth_view, dtype=np.uint16, order="C", copy=True),
            )
            depth_scale = self._depth_units(depth_frame)
            depth_metres = np.empty(raw_depth.shape, dtype=np.float32)
            np.multiply(raw_depth, np.float32(depth_scale), out=depth_metres)

            color_intrinsics = _reported_intrinsics(_frame_profile(color_frame)).pinhole
            depth_intrinsics = _reported_intrinsics(_frame_profile(depth_frame)).pinhole
            color_number = self._frame_number(color_frame, "color")
            depth_number = self._frame_number(depth_frame, "depth")
            color_source_id, depth_source_id = self._source_ids()
            frame_id = self._config.color_frame_id
            info = self.info
            color_header = SampleHeader(
                sample_id=f"{self._run_id}/{color_source_id}/{color_number}",
                source_id=color_source_id,
                sequence_number=color_number,
                measurement_kind=MeasurementKind.RGB_IMAGE,
                captured_at=self._captured_at(color_frame),
                received_at=received_at,
                frame_id=frame_id,
                calibration=self._config.calibration,
                producer=f"librealsense-python/{info.sdk_version}/color-stream",
                produced_on=ComputePlacement(
                    ComputeKind.UNKNOWN,
                    f"realsense:{info.serial_number}:color-path",
                ),
            )
            depth_header = SampleHeader(
                sample_id=f"{self._run_id}/{depth_source_id}/{depth_number}",
                source_id=depth_source_id,
                sequence_number=depth_number,
                measurement_kind=MeasurementKind.DEPTH_IMAGE,
                captured_at=self._captured_at(depth_frame),
                received_at=received_at,
                frame_id=frame_id,
                calibration=self._config.calibration,
                producer=f"librealsense-python/{info.sdk_version}/align-depth-to-color",
                produced_on=ComputePlacement(
                    ComputeKind.HOST_CPU,
                    f"host-process-{os.getpid()}/librealsense-align",
                ),
            )
            color_sample = SensorSample(
                header=color_header,
                payload=ImageFrame(color_pixels, PixelFormat.BGR8),
                payload_memory=MemoryPlacement(MemoryKind.HOST),
            )
            depth_sample = SensorSample(
                header=depth_header,
                payload=DepthFrame(depth_metres),
                payload_memory=MemoryPlacement(MemoryKind.HOST),
            )
            result = AlignedRgbdFrame(
                frameset_id=f"{self._run_id}/realsense-frameset/{self._frameset_sequence}",
                color=color_sample,
                depth=depth_sample,
                color_intrinsics=color_intrinsics,
                aligned_depth_intrinsics=depth_intrinsics,
                synchronization_method="librealsense-composite-frameset",
                alignment_method="librealsense-align-depth-to-color",
            )
        except RealSenseReadError:
            raise
        except Exception as error:
            raise RealSenseReadError("failed to acquire a valid RealSense RGB-D frame") from error

        self._frameset_sequence += 1
        return result

    def _source_ids(self) -> tuple[str, str]:
        serial = self.info.serial_number
        color = self._config.color_source_id or f"sensors/realsense-{serial}/color"
        depth = (
            self._config.aligned_depth_source_id
            or f"sensors/realsense-{serial}/depth-aligned-to-color"
        )
        return color, depth

    def _depth_units(self, depth_frame: Any) -> float:
        getter = getattr(depth_frame, "get_units", None)
        units = float(getter()) if callable(getter) else self.info.depth_scale_metres
        if not math.isfinite(units) or units <= 0:
            raise RealSenseReadError("depth frame reported invalid metric units")
        return units

    def _frame_number(self, frame: Any, stream_name: str) -> int:
        number = int(frame.get_frame_number())
        if number < 0:
            raise RealSenseReadError(f"{stream_name} frame number must be non-negative")
        return number

    def _captured_at(self, frame: Any) -> TimePoint | None:
        try:
            milliseconds = float(frame.get_timestamp())
            domain_name = _enum_name(frame.get_frame_timestamp_domain())
        except (AttributeError, TypeError, ValueError):
            return None
        if not math.isfinite(milliseconds):
            return None
        nanoseconds = round(milliseconds * 1_000_000)
        domain = self._device_clocks.get(domain_name)
        if domain is None:
            domain = ClockDomain(
                f"realsense/{self.info.serial_number}/{domain_name}/run-{self._run_id}",
                ClockKind.DEVICE,
            )
            self._device_clocks[domain_name] = domain
        try:
            return TimePoint(nanoseconds, domain)
        except (TypeError, ValueError):
            return None

    def close(self) -> None:
        pipeline = self._pipeline
        self._sdk = None
        self._pipeline = None
        self._align = None
        self._info = None
        self._device_clocks.clear()
        if pipeline is not None:
            try:
                pipeline.stop()
            except Exception as error:
                raise RealSenseError("failed to stop the RealSense pipeline") from error

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "RealSenseConfig",
    "RealSenseDependencyError",
    "RealSenseError",
    "RealSenseInfo",
    "RealSenseIntrinsics",
    "RealSenseOpenError",
    "RealSenseReadError",
    "RealSenseSource",
    "RealSenseStreamInfo",
    "RealSenseStreamMode",
]
