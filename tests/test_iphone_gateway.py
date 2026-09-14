from __future__ import annotations

import io
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from vision_pipeline.apps.iphone_receiver import IphoneDatasetRecorder
from vision_pipeline.sources.iphone_gateway import (
    IphoneGatewayProtocolError,
    IphoneImuTimeline,
    StreamContinuityMonitor,
    parse_imu_line,
    read_rgbd_frame,
)


def _imu_document(
    *, sequence: int = 7, timestamp_ns: int = 123_000_000, session: str = "run-a"
) -> dict[str, object]:
    return {
        "schema": "iphone_imu.v1",
        "source_id": "iphone-test",
        "session_id": session,
        "sequence_number": sequence,
        "device_timestamp_ns": timestamp_ns,
        "attitude_quaternion": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        "gravity_compensated_acceleration_mps2": {"x": 1.0, "y": 2.0, "z": 3.0},
        "angular_velocity_radps": {"x": 0.1, "y": 0.2, "z": 0.3},
        "gravity_mps2": {"x": 0.0, "y": 0.0, "z": -9.80665},
    }


def _rgbd_packet(
    *,
    sequence: int = 3,
    timestamp_ns: int = 120_000_000,
    session: str = "run-a",
    depth_byte_count: int | None = None,
) -> bytes:
    rgb = b"fake-jpeg"
    depth = np.array([[1.0, 2.0]], dtype="<f4").tobytes()
    confidence = np.array([[2, 0]], dtype=np.uint8).tobytes()
    header = {
        "schema": "iphone_rgbd.v1",
        "source_id": "iphone-test",
        "session_id": session,
        "sequence_number": sequence,
        "device_timestamp_ns": timestamp_ns,
        "tracking_state": "normal",
        "rgb_encoding": "jpeg",
        "rgb_width": 4,
        "rgb_height": 2,
        "depth_encoding": "float32_le_metres",
        "depth_width": 2,
        "depth_height": 1,
        "confidence_encoding": "arkit_uint8_0_low_1_medium_2_high",
        "rgb_orientation": "camera_buffer_native",
        "camera_intrinsics_column_major": [
            4.0,
            0.0,
            0.0,
            0.0,
            4.0,
            0.0,
            1.5,
            0.5,
            1.0,
        ],
        "camera_transform_column_major": [
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0,
        ],
        "payload": {
            "rgb_bytes": len(rgb),
            "depth_bytes": len(depth) if depth_byte_count is None else depth_byte_count,
            "confidence_bytes": len(confidence),
        },
    }
    encoded_header = json.dumps(header).encode()
    return struct.pack(">I", len(encoded_header)) + encoded_header + rgb + depth + confidence


def test_parse_imu_and_associate_same_device_clock() -> None:
    sample = parse_imu_line(json.dumps(_imu_document()).encode())
    frame = read_rgbd_frame(io.BytesIO(_rgbd_packet()))
    timeline = IphoneImuTimeline()
    timeline.append(sample)

    nearest, delta_ns = timeline.nearest(frame)

    assert nearest == sample
    assert delta_ns == 3_000_000
    assert sample.gravity_compensated_acceleration_mps2 == (1.0, 2.0, 3.0)


def test_decode_rgbd_and_transform_optical_points_to_arkit_world() -> None:
    frame = read_rgbd_frame(io.BytesIO(_rgbd_packet()))

    assert frame.depth_metres.shape == (1, 2)
    assert frame.depth_intrinsics.fx == pytest.approx(2.0)
    assert frame.depth_intrinsics.fy == pytest.approx(2.0)
    points = frame.world_points(sampling_stride=1, minimum_confidence=1)

    # The second depth pixel has low confidence and is rejected.
    assert points.shape == (1, 3)
    np.testing.assert_allclose(points[0], (-0.375, 0.125, -1.0))


def test_rgbd_rejects_inconsistent_payload_dimensions() -> None:
    with pytest.raises(IphoneGatewayProtocolError, match="depth byte count"):
        read_rgbd_frame(io.BytesIO(_rgbd_packet(depth_byte_count=4)))


def test_continuity_monitor_detects_gap_order_and_new_session() -> None:
    monitor = StreamContinuityMonitor()
    assert monitor.observe("run-a", 4, 100).new_session
    gap = monitor.observe("run-a", 7, 200)
    assert gap.dropped == 2
    reordered = monitor.observe("run-a", 7, 190)
    assert reordered.duplicate_or_reordered
    assert reordered.timestamp_regressed
    assert monitor.observe("run-b", 0, 10).new_session


def test_dataset_recorder_keeps_binary_payloads_and_time_semantics(tmp_path: Path) -> None:
    output = tmp_path / "capture"
    sample = parse_imu_line(json.dumps(_imu_document()))
    frame = read_rgbd_frame(io.BytesIO(_rgbd_packet()))
    with IphoneDatasetRecorder(output) as recorder:
        recorder.record_imu(sample, host_received_ns=1_000)
        recorder.record_frame(frame, host_received_ns=2_000, nearest_imu_delta_ns=3_000_000)

    manifest = json.loads((output / "manifest.json").read_text())
    imu = json.loads((output / "imu.ndjson").read_text().strip())
    metadata = json.loads((output / "frames.ndjson").read_text().strip())
    assert manifest["schema"] == "vision_pipeline.iphone_dataset.v1"
    assert imu["host_received_monotonic_ns"] == 1_000
    assert metadata["nearest_imu_delta_ns"] == 3_000_000
    assert np.load(output / metadata["depth_path"], allow_pickle=False).shape == (1, 2)
    assert (output / metadata["rgb_path"]).read_bytes() == b"fake-jpeg"
