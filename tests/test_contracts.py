from dataclasses import FrozenInstanceError

import pytest

from vision_pipeline.contracts import (
    CalibrationRef,
    ClockDomain,
    ClockDomainMismatchError,
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

HOST_CLOCK = ClockDomain("host/boot-42", ClockKind.HOST_MONOTONIC)
CAMERA_CLOCK = ClockDomain("realsense/front/epoch-7", ClockKind.DEVICE)


def make_header(**overrides: object) -> SampleHeader:
    values = {
        "sample_id": "episode-1/front-depth/17",
        "source_id": "sensors/front_realsense/depth",
        "sequence_number": 17,
        "measurement_kind": MeasurementKind.DEPTH_IMAGE,
        "captured_at": TimePoint(1_000_000, CAMERA_CLOCK),
        "received_at": TimePoint(2_000_000, HOST_CLOCK),
        "frame_id": FrameId("front_depth_optical"),
        "calibration": CalibrationRef("front-realsense-640x480", "sha256:abc123"),
        "producer": "realsense-depth-engine",
        "produced_on": ComputePlacement(
            ComputeKind.SENSOR_DEVICE,
            "realsense:front:depth-asic",
        ),
    }
    values.update(overrides)
    return SampleHeader(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("factory", "expected_exception"),
    [
        (lambda: ClockDomain(" ", ClockKind.DEVICE), ValueError),
        (lambda: FrameId("bad\x00frame"), ValueError),
        (lambda: CalibrationRef("camera", ""), ValueError),
    ],
)
def test_identifiers_are_explicit_and_valid(
    factory: object,
    expected_exception: type[Exception],
) -> None:
    with pytest.raises(expected_exception):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize("invalid", [True, 1.25, -(2**63) - 1, 2**63])
def test_timepoint_requires_signed_int64(invalid: object) -> None:
    expected = TypeError if isinstance(invalid, bool | float) else ValueError
    with pytest.raises(expected):
        TimePoint(invalid, HOST_CLOCK)  # type: ignore[arg-type]


def test_elapsed_time_is_only_valid_inside_one_clock_domain() -> None:
    start = TimePoint(100, HOST_CLOCK)
    end = TimePoint(145, HOST_CLOCK)

    assert end - start == 45
    assert start < end


def test_different_clock_domains_cannot_be_compared() -> None:
    device_time = TimePoint(100, CAMERA_CLOCK)
    host_time = TimePoint(100, HOST_CLOCK)

    with pytest.raises(ClockDomainMismatchError):
        _ = host_time - device_time
    with pytest.raises(ClockDomainMismatchError):
        _ = host_time < device_time


def test_same_clock_kind_does_not_make_devices_comparable() -> None:
    left = TimePoint(100, ClockDomain("camera/left/epoch-1", ClockKind.DEVICE))
    right = TimePoint(100, ClockDomain("camera/right/epoch-1", ClockKind.DEVICE))

    with pytest.raises(ClockDomainMismatchError):
        _ = left - right


def test_clock_reset_creates_a_new_incomparable_epoch() -> None:
    before_reset = TimePoint(99, ClockDomain("camera/left/epoch-1", ClockKind.DEVICE))
    after_reset = TimePoint(1, ClockDomain("camera/left/epoch-2", ClockKind.DEVICE))

    with pytest.raises(ClockDomainMismatchError):
        _ = after_reset - before_reset


def test_sample_preserves_capture_and_receive_clock_domains() -> None:
    header = make_header()

    assert header.captured_at is not None
    assert header.captured_at.clock == CAMERA_CLOCK
    assert header.received_at.clock == HOST_CLOCK


def test_missing_capture_time_and_calibration_are_not_fabricated() -> None:
    header = make_header(captured_at=None, calibration=None)

    assert header.captured_at is None
    assert header.calibration is None


def test_receive_cannot_precede_capture_in_same_domain() -> None:
    with pytest.raises(ValueError, match="cannot precede"):
        make_header(
            captured_at=TimePoint(101, HOST_CLOCK),
            received_at=TimePoint(100, HOST_CLOCK),
        )


def test_sequence_number_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        make_header(sequence_number=-1)


def test_on_sensor_compute_and_host_memory_are_independent() -> None:
    payload = bytearray(b"depth-frame")
    sample = SensorSample(
        header=make_header(),
        payload=payload,
        payload_memory=MemoryPlacement(MemoryKind.HOST),
    )

    assert sample.header.produced_on.kind is ComputeKind.SENSOR_DEVICE
    assert sample.payload_memory.kind is MemoryKind.HOST
    assert sample.payload is payload


def test_gpu_compute_and_device_memory_can_be_recorded() -> None:
    sample = SensorSample(
        header=make_header(
            producer="gpu-depth-registration",
            produced_on=ComputePlacement(ComputeKind.HOST_GPU, "metal:0"),
        ),
        payload=object(),
        payload_memory=MemoryPlacement(MemoryKind.DEVICE_LOCAL, "metal:0"),
    )

    assert sample.header.produced_on.device_id == sample.payload_memory.device_id


def test_sample_envelope_is_immutable() -> None:
    sample = SensorSample(
        header=make_header(),
        payload=object(),
        payload_memory=MemoryPlacement(MemoryKind.HOST),
    )

    with pytest.raises(FrozenInstanceError):
        sample.payload = object()  # type: ignore[misc]
