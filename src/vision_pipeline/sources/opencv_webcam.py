"""Laptop webcam source backed by OpenCV VideoCapture."""

from __future__ import annotations

import math
import os
import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, Self, cast

import cv2
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
from vision_pipeline.image import ImageFrame, PixelFormat


class WebcamError(RuntimeError):
    """Base class for laptop webcam failures."""


class WebcamOpenError(WebcamError):
    """The configured webcam could not be opened."""


class WebcamReadError(WebcamError):
    """The webcam failed to produce a valid frame."""


class WebcamBackend(StrEnum):
    AUTO = "auto"
    AVFOUNDATION = "avfoundation"
    V4L2 = "v4l2"
    MSMF = "msmf"


@dataclass(frozen=True, slots=True)
class WebcamConfig:
    device_index: int = 0
    backend: WebcamBackend = WebcamBackend.AUTO
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    source_id: str | None = None
    frame_id: FrameId = FrameId("laptop_webcam_optical")
    calibration: CalibrationRef | None = None

    def __post_init__(self) -> None:
        if isinstance(self.device_index, bool) or not isinstance(self.device_index, int):
            raise TypeError("device_index must be an integer")
        if self.device_index < 0:
            raise ValueError("device_index must be non-negative")
        if not isinstance(self.backend, WebcamBackend):
            raise TypeError("backend must be a WebcamBackend")
        for field_name, value in (("width", self.width), ("height", self.height)):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise TypeError(f"{field_name} must be an integer or None")
                if value <= 0:
                    raise ValueError(f"{field_name} must be positive")
        if self.fps is not None:
            if isinstance(self.fps, bool) or not isinstance(self.fps, int | float):
                raise TypeError("fps must be a number or None")
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise ValueError("fps must be positive and finite")
        if self.source_id is not None and not self.source_id.strip():
            raise ValueError("source_id must be non-empty when provided")
        if not isinstance(self.frame_id, FrameId):
            raise TypeError("frame_id must be a FrameId")
        if self.calibration is not None and not isinstance(self.calibration, CalibrationRef):
            raise TypeError("calibration must be a CalibrationRef or None")


@dataclass(frozen=True, slots=True)
class CameraMode:
    width: int | None
    height: int | None
    fps: float | None


@dataclass(frozen=True, slots=True)
class PropertyAcceptance:
    width: bool | None
    height: bool | None
    fps: bool | None


@dataclass(frozen=True, slots=True)
class WebcamInfo:
    backend: str
    requested: CameraMode
    actual: CameraMode
    accepted: PropertyAcceptance


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]: ...

    def set(self, property_id: int, value: float) -> bool: ...

    def get(self, property_id: int) -> float: ...

    def getBackendName(self) -> str: ...

    def release(self) -> None: ...


CaptureFactory = Callable[[int, int], VideoCaptureLike]
ClockReader = Callable[[], int]


def _default_capture_factory(device_index: int, backend_id: int) -> VideoCaptureLike:
    return cast(VideoCaptureLike, cv2.VideoCapture(device_index, backend_id))


def _resolve_backend(backend: WebcamBackend) -> int:
    if backend is WebcamBackend.AUTO:
        return cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    mapping = {
        WebcamBackend.AVFOUNDATION: cv2.CAP_AVFOUNDATION,
        WebcamBackend.V4L2: cv2.CAP_V4L2,
        WebcamBackend.MSMF: cv2.CAP_MSMF,
    }
    return mapping[backend]


def _reported_int(capture: VideoCaptureLike, property_id: int) -> int:
    value = float(capture.get(property_id))
    return int(round(value)) if math.isfinite(value) and value > 0 else 0


def _reported_float(capture: VideoCaptureLike, property_id: int) -> float:
    value = float(capture.get(property_id))
    return value if math.isfinite(value) and value > 0 else 0.0


def _set_if_requested(
    capture: VideoCaptureLike,
    property_id: int,
    value: int | float | None,
) -> bool | None:
    return None if value is None else bool(capture.set(property_id, float(value)))


class OpenCvWebcam:
    """Capture honest, host-resident BGR measurements from a laptop webcam.

    OpenCV does not expose a synchronization-grade live-camera acquisition timestamp,
    so emitted samples keep ``captured_at=None`` and record host receipt immediately
    after ``VideoCapture.read`` succeeds.
    """

    def __init__(
        self,
        config: WebcamConfig,
        *,
        capture_factory: CaptureFactory = _default_capture_factory,
        clock_ns: ClockReader = time.monotonic_ns,
        host_clock: ClockDomain | None = None,
        run_id: str | None = None,
    ) -> None:
        self._config = config
        self._capture_factory = capture_factory
        self._clock_ns = clock_ns
        self._run_id = run_id or uuid.uuid4().hex
        self._host_clock = host_clock or ClockDomain(
            f"host/monotonic/process-{os.getpid()}/run-{self._run_id}",
            ClockKind.HOST_MONOTONIC,
        )
        self._capture: VideoCaptureLike | None = None
        self._info: WebcamInfo | None = None
        self._sequence_number = 0

    @property
    def is_open(self) -> bool:
        return self._capture is not None

    @property
    def info(self) -> WebcamInfo:
        if self._info is None:
            raise WebcamOpenError("webcam information is unavailable before open()")
        return self._info

    def open(self) -> None:
        if self.is_open:
            raise WebcamOpenError("webcam is already open")

        backend_id = _resolve_backend(self._config.backend)
        capture = self._capture_factory(self._config.device_index, backend_id)
        if not capture.isOpened():
            capture.release()
            raise WebcamOpenError(
                f"could not open webcam index {self._config.device_index} "
                f"with backend {self._config.backend.value}"
            )

        try:
            accepted = PropertyAcceptance(
                width=_set_if_requested(
                    capture,
                    cv2.CAP_PROP_FRAME_WIDTH,
                    self._config.width,
                ),
                height=_set_if_requested(
                    capture,
                    cv2.CAP_PROP_FRAME_HEIGHT,
                    self._config.height,
                ),
                fps=_set_if_requested(capture, cv2.CAP_PROP_FPS, self._config.fps),
            )
            try:
                backend_name = capture.getBackendName()
            except (AttributeError, cv2.error):
                backend_name = self._config.backend.value

            actual = CameraMode(
                width=_reported_int(capture, cv2.CAP_PROP_FRAME_WIDTH),
                height=_reported_int(capture, cv2.CAP_PROP_FRAME_HEIGHT),
                fps=_reported_float(capture, cv2.CAP_PROP_FPS),
            )
        except Exception as error:
            capture.release()
            raise WebcamOpenError("webcam opened but profile negotiation failed") from error

        self._capture = capture
        self._sequence_number = 0
        self._info = WebcamInfo(
            backend=backend_name,
            requested=CameraMode(self._config.width, self._config.height, self._config.fps),
            actual=actual,
            accepted=accepted,
        )

    def read(self) -> SensorSample[ImageFrame]:
        capture = self._capture
        if capture is None:
            raise WebcamReadError("open() must be called before read()")

        success, pixels = capture.read()
        if not success or pixels is None:
            raise WebcamReadError("webcam returned no frame")

        received_at = TimePoint(self._clock_ns(), self._host_clock)
        try:
            image = ImageFrame(pixels, PixelFormat.BGR8)
        except (TypeError, ValueError) as error:
            raise WebcamReadError(f"webcam returned an invalid BGR frame: {error}") from error

        info = self.info
        if image.width != info.actual.width or image.height != info.actual.height:
            self._info = replace(
                info,
                actual=replace(info.actual, width=image.width, height=image.height),
            )

        sequence_number = self._sequence_number
        source_id = self._config.source_id or f"sensors/webcam-{self._config.device_index}/rgb"
        header = SampleHeader(
            sample_id=f"{self._run_id}/{source_id}/{sequence_number}",
            source_id=source_id,
            sequence_number=sequence_number,
            measurement_kind=MeasurementKind.RGB_IMAGE,
            captured_at=None,
            received_at=received_at,
            frame_id=self._config.frame_id,
            calibration=self._config.calibration,
            producer=f"opencv-videocapture/{info.backend.lower()}",
            produced_on=ComputePlacement(ComputeKind.HOST_CPU, "opencv-videocapture"),
        )
        sample = SensorSample(
            header=header,
            payload=image,
            payload_memory=MemoryPlacement(MemoryKind.HOST),
        )
        self._sequence_number += 1
        return sample

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        self._info = None
        if capture is not None:
            capture.release()

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
