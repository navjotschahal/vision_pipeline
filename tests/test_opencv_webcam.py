from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from vision_pipeline.contracts import (
    ClockDomain,
    ClockKind,
    ComputeKind,
    FrameId,
    MeasurementKind,
    MemoryKind,
)
from vision_pipeline.image import PixelFormat
from vision_pipeline.sources.opencv_webcam import (
    OpenCvWebcam,
    VideoCaptureLike,
    WebcamConfig,
    WebcamOpenError,
    WebcamReadError,
)


class FakeCapture:
    def __init__(
        self,
        reads: list[tuple[bool, NDArray[np.uint8] | None]],
        *,
        opened: bool = True,
        events: list[str] | None = None,
    ) -> None:
        self.opened = opened
        self.reads = reads
        self.events = events if events is not None else []
        self.release_count = 0
        self.settings: list[tuple[int, float]] = []
        self.reported = {
            cv2.CAP_PROP_FRAME_WIDTH: 640.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 480.0,
            cv2.CAP_PROP_FPS: 29.97,
        }

    def isOpened(self) -> bool:
        return self.opened

    def read(self) -> tuple[bool, NDArray[np.uint8] | None]:
        self.events.append("read")
        return self.reads.pop(0)

    def set(self, property_id: int, value: float) -> bool:
        self.settings.append((property_id, value))
        return True

    def get(self, property_id: int) -> float:
        return self.reported[property_id]

    def getBackendName(self) -> str:
        return "AVFOUNDATION"

    def release(self) -> None:
        self.release_count += 1


def factory_for(capture: FakeCapture) -> Callable[[int, int], VideoCaptureLike]:
    def factory(_device_index: int, _backend_id: int) -> VideoCaptureLike:
        return capture

    return factory


def test_read_returns_truthful_host_bgr_measurement() -> None:
    events: list[str] = []
    pixels = np.array([[[1, 2, 3]]], dtype=np.uint8)
    capture = FakeCapture([(True, pixels)], events=events)
    host_clock = ClockDomain("host/test-run", ClockKind.HOST_MONOTONIC)

    def clock_ns() -> int:
        events.append("clock")
        return 123_456

    camera = OpenCvWebcam(
        WebcamConfig(source_id="sensors/laptop/rgb", frame_id=FrameId("laptop_optical")),
        capture_factory=factory_for(capture),
        clock_ns=clock_ns,
        host_clock=host_clock,
        run_id="test-run",
    )

    with camera:
        sample = camera.read()

    assert events == ["read", "clock"]
    assert sample.header.captured_at is None
    assert sample.header.received_at.nanoseconds == 123_456
    assert sample.header.received_at.clock == host_clock
    assert sample.header.measurement_kind is MeasurementKind.RGB_IMAGE
    assert sample.header.source_id == "sensors/laptop/rgb"
    assert sample.header.frame_id == FrameId("laptop_optical")
    assert sample.header.sequence_number == 0
    assert sample.header.sample_id == "test-run/sensors/laptop/rgb/0"
    assert sample.header.calibration is None
    assert sample.header.produced_on.kind is ComputeKind.HOST_CPU
    assert sample.payload.pixel_format is PixelFormat.BGR8
    assert sample.payload.data is pixels
    assert sample.payload.data[0, 0].tolist() == [1, 2, 3]
    assert sample.payload_memory.kind is MemoryKind.HOST


def test_failed_read_does_not_stamp_time_or_consume_sequence() -> None:
    events: list[str] = []
    pixels = np.zeros((2, 3, 3), dtype=np.uint8)
    capture = FakeCapture([(False, None), (True, pixels)], events=events)

    def clock_ns() -> int:
        events.append("clock")
        return 100

    camera = OpenCvWebcam(
        WebcamConfig(),
        capture_factory=factory_for(capture),
        clock_ns=clock_ns,
        run_id="run",
    )
    camera.open()

    with pytest.raises(WebcamReadError, match="no frame"):
        camera.read()
    sample = camera.read()
    camera.close()

    assert events == ["read", "read", "clock"]
    assert sample.header.sequence_number == 0


def test_requested_mode_is_distinct_from_reported_mode() -> None:
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = FakeCapture([(True, pixels)])
    camera = OpenCvWebcam(
        WebcamConfig(width=1280, height=720, fps=60.0),
        capture_factory=factory_for(capture),
        run_id="run",
    )

    with camera:
        assert camera.info.requested.width == 1280
        assert camera.info.requested.height == 720
        assert camera.info.requested.fps == 60.0
        assert camera.info.actual.width == 640
        assert camera.info.actual.height == 480
        assert camera.info.actual.fps == pytest.approx(29.97)

    assert capture.settings == [
        (cv2.CAP_PROP_FRAME_WIDTH, 1280),
        (cv2.CAP_PROP_FRAME_HEIGHT, 720),
        (cv2.CAP_PROP_FPS, 60.0),
    ]


def test_default_config_does_not_guess_camera_properties() -> None:
    capture = FakeCapture([])
    camera = OpenCvWebcam(
        WebcamConfig(),
        capture_factory=factory_for(capture),
        run_id="run",
    )

    with camera:
        assert camera.info.requested.width is None
        assert camera.info.requested.height is None
        assert camera.info.requested.fps is None
        assert camera.info.accepted.width is None
        assert camera.info.accepted.height is None
        assert camera.info.accepted.fps is None

    assert capture.settings == []


def test_unopened_camera_is_released_and_reported() -> None:
    capture = FakeCapture([], opened=False)
    camera = OpenCvWebcam(
        WebcamConfig(),
        capture_factory=factory_for(capture),
        run_id="run",
    )

    with pytest.raises(WebcamOpenError, match="could not open"):
        camera.open()

    assert capture.release_count == 1
    assert not camera.is_open


def test_context_manager_releases_once_after_error() -> None:
    capture = FakeCapture([(False, None)])
    camera = OpenCvWebcam(
        WebcamConfig(),
        capture_factory=factory_for(capture),
        run_id="run",
    )

    with pytest.raises(WebcamReadError), camera:
        camera.read()

    camera.close()
    assert capture.release_count == 1
