import json
from datetime import UTC, datetime

import pytest

from vision_pipeline.sources.camera_profiles import (
    CameraProbeError,
    camera_by_index,
    parse_camera_probe_json,
    render_macos_camera_report,
)


def probe_output() -> str:
    return json.dumps(
        [
            {
                "index": 0,
                "name": "Test Camera",
                "unique_id": "camera-123",
                "position": "front",
                "is_connected": True,
                "is_suspended": False,
                "profiles": [
                    {
                        "width": 3840,
                        "height": 2160,
                        "min_fps": 15.0,
                        "max_fps": 30.0,
                        "pixel_format": "420v",
                    },
                    {
                        "width": 1920,
                        "height": 1080,
                        "min_fps": 15.0,
                        "max_fps": 60.0,
                        "pixel_format": "420v",
                    },
                    {
                        "width": 1280,
                        "height": 720,
                        "min_fps": 15.0,
                        "max_fps": 120.0,
                        "pixel_format": "420v",
                    },
                ],
            }
        ]
    )


def test_native_probe_profiles_are_parsed() -> None:
    (device,) = parse_camera_probe_json(probe_output())

    assert device.index == 0
    assert device.name == "Test Camera"
    assert device.unique_id == "camera-123"
    assert len(device.profiles) == 3
    assert device.profiles[0].width == 3840


def test_probe_report_preserves_raw_modes_without_recommending_one() -> None:
    devices = parse_camera_probe_json(probe_output())

    report = render_macos_camera_report(
        devices,
        generated_at=datetime(2026, 9, 11, 12, 0, tzinfo=UTC),
    )

    assert "Generated: `2026-09-11T12:00:00+00:00`" in report
    assert "Stable native ID: `camera-123`" in report
    assert "| 1280 | 720 | 15 | 120 | `420v` |" in report
    assert "This is an inventory, not a recommendation" in report


def test_camera_index_must_come_from_probe_results() -> None:
    devices = parse_camera_probe_json(probe_output())

    assert camera_by_index(devices, 0).unique_id == "camera-123"
    with pytest.raises(CameraProbeError, match="available indexes: 0"):
        camera_by_index(devices, 4)


def test_invalid_probe_output_is_rejected() -> None:
    with pytest.raises(CameraProbeError, match="invalid native camera probe output"):
        parse_camera_probe_json('{"not": "a device list"}')
