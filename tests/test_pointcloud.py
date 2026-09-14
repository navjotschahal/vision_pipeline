from __future__ import annotations

import numpy as np
import pytest
import torch

from vision_pipeline.contracts import (
    ClockDomain,
    ClockKind,
    ComputeKind,
    ComputePlacement,
    FrameId,
    MeasurementKind,
    MemoryKind,
    MemoryPlacement,
    SampleHeader,
    TimePoint,
)
from vision_pipeline.geometry import (
    GeometryKind,
    PinholeIntrinsics,
    depth_to_point_cloud,
)
from vision_pipeline.geometry.visualization import (
    PointCloudRenderer,
    PointCloudViewConfig,
)
from vision_pipeline.image import ImageFrame, PixelFormat


def source_header() -> SampleHeader:
    clock = ClockDomain("host/test", ClockKind.HOST_MONOTONIC)
    return SampleHeader(
        sample_id="camera/1",
        source_id="camera",
        sequence_number=1,
        measurement_kind=MeasurementKind.RGB_IMAGE,
        captured_at=None,
        received_at=TimePoint(100, clock),
        frame_id=FrameId("camera_optical"),
        calibration=None,
        producer="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
    )


def test_intrinsics_from_horizontal_fov_are_explicitly_uncalibrated() -> None:
    intrinsics = PinholeIntrinsics.from_horizontal_fov(4, 2, 90)

    assert intrinsics.fx == pytest.approx(2.0)
    assert intrinsics.fy == pytest.approx(2.0)
    assert intrinsics.cx == pytest.approx(1.5)
    assert intrinsics.cy == pytest.approx(0.5)
    assert intrinsics.calibration_id is None


def test_depth_deprojection_produces_xyz_and_rgb_on_the_tensor_device() -> None:
    depth = torch.full((1, 1, 2, 2), 2.0)
    bgr = np.array(
        [
            [[30, 20, 10], [60, 50, 40]],
            [[90, 80, 70], [120, 110, 100]],
        ],
        dtype=np.uint8,
    )
    image = ImageFrame(bgr, PixelFormat.BGR8)
    compute = ComputePlacement(ComputeKind.HOST_CPU, "cpu")
    memory = MemoryPlacement(MemoryKind.HOST)

    cloud = depth_to_point_cloud(
        depth,
        image,
        source_header(),
        PinholeIntrinsics(2, 2, fx=1, fy=1, cx=0.5, cy=0.5, calibration_id="test"),
        min_depth_metres=0.1,
        max_depth_metres=3,
        sampling_stride=1,
        geometry_kind=GeometryKind.MEASURED_METRIC,
        produced_on=compute,
        memory=memory,
    )

    torch.testing.assert_close(
        cloud.xyz,
        torch.tensor(
            [[-1, -1, 2], [1, -1, 2], [-1, 1, 2], [1, 1, 2]],
            dtype=torch.float32,
        ),
    )
    assert cloud.colors_rgb is not None
    assert cloud.colors_rgb.tolist() == [
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
        [100, 110, 120],
    ]
    assert cloud.xyz.device == depth.device
    assert cloud.geometry_kind is GeometryKind.MEASURED_METRIC


def test_depth_deprojection_filters_invalid_and_out_of_range_points() -> None:
    depth = torch.tensor([[float("nan"), 0.05], [1.0, 4.0]])
    image = ImageFrame(np.zeros((2, 2, 3), dtype=np.uint8), PixelFormat.RGB8)

    cloud = depth_to_point_cloud(
        depth,
        image,
        source_header(),
        PinholeIntrinsics(2, 2, fx=1, fy=1, cx=0, cy=0),
        min_depth_metres=0.1,
        max_depth_metres=3,
        sampling_stride=1,
        geometry_kind=GeometryKind.ESTIMATED_METRIC,
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
        memory=MemoryPlacement(MemoryKind.HOST),
    )

    assert cloud.xyz.shape == (1, 3)
    assert cloud.xyz[0].tolist() == pytest.approx([0.0, 1.0, 1.0])


def test_point_cloud_renderer_produces_fixed_size_view() -> None:
    depth = torch.full((2, 2), 2.0)
    image = ImageFrame(np.full((2, 2, 3), 200, dtype=np.uint8), PixelFormat.RGB8)
    cloud = depth_to_point_cloud(
        depth,
        image,
        source_header(),
        PinholeIntrinsics(2, 2, fx=1, fy=1, cx=0.5, cy=0.5),
        min_depth_metres=0.1,
        max_depth_metres=3,
        sampling_stride=1,
        geometry_kind=GeometryKind.ESTIMATED_METRIC,
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
        memory=MemoryPlacement(MemoryKind.HOST),
    )
    renderer = PointCloudRenderer(PointCloudViewConfig(width=320, height=240, point_size=1))

    view = renderer.render(cloud)
    renderer.rotate(yaw_degrees=10, pitch_degrees=-5, roll_degrees=365)
    renderer.change_zoom(1.2)

    assert view.shape == (240, 320, 3)
    assert view.dtype == np.uint8
    assert renderer.zoom == pytest.approx(1.2)
    assert renderer.roll_radians == pytest.approx(np.deg2rad(5))
