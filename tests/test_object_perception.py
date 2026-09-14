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
from vision_pipeline.geometry import GeometryKind, PinholeIntrinsics, PointCloud, Twist3D
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception import BoundingBox2D
from vision_pipeline.perception.objects import (
    ObjectTrack3D,
    SelectionKind,
    TrackStatus,
    estimate_object_observation,
    estimate_observed_oriented_box,
    extract_object_point_cloud,
)


def source_header() -> SampleHeader:
    clock = ClockDomain("host/object-test", ClockKind.HOST_MONOTONIC)
    return SampleHeader(
        sample_id="camera/5",
        source_id="camera",
        sequence_number=5,
        measurement_kind=MeasurementKind.DEPTH_IMAGE,
        captured_at=TimePoint(100, clock),
        received_at=TimePoint(110, clock),
        frame_id=FrameId("camera_optical"),
        calibration=None,
        producer="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
    )


def placement() -> tuple[ComputePlacement, MemoryPlacement]:
    return (
        ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
        MemoryPlacement(MemoryKind.HOST),
    )


def test_segmentation_mask_extracts_only_selected_depth_pixels() -> None:
    depth = torch.full((4, 4), 2.0)
    image = ImageFrame(np.zeros((4, 4, 3), dtype=np.uint8), PixelFormat.RGB8)
    mask = torch.zeros((4, 4), dtype=torch.bool)
    mask[1:3, 1:3] = True
    compute, memory = placement()

    cloud, selection_kind = extract_object_point_cloud(
        depth,
        image,
        source_header(),
        PinholeIntrinsics(4, 4, fx=2, fy=2, cx=1.5, cy=1.5),
        segmentation_mask=mask,
        min_depth_metres=0.1,
        max_depth_metres=3,
        sampling_stride=1,
        geometry_kind=GeometryKind.MEASURED_METRIC,
        produced_on=compute,
        memory=memory,
    )

    assert selection_kind is SelectionKind.SEGMENTATION_MASK
    assert cloud.xyz.shape == (4, 3)
    assert torch.all(cloud.xyz[:, 2] == 2)


def test_bounding_box_is_an_explicit_coarse_selection_fallback() -> None:
    depth = torch.full((4, 4), 1.0)
    image = ImageFrame(np.zeros((4, 4, 3), dtype=np.uint8), PixelFormat.RGB8)
    box = BoundingBox2D(1, 1, 3, 3)
    compute, memory = placement()

    cloud, selection_kind = extract_object_point_cloud(
        depth,
        image,
        source_header(),
        PinholeIntrinsics(4, 4, fx=2, fy=2, cx=1.5, cy=1.5),
        image_box=box,
        min_depth_metres=0.1,
        max_depth_metres=3,
        sampling_stride=1,
        geometry_kind=GeometryKind.ESTIMATED_METRIC,
        produced_on=compute,
        memory=memory,
    )

    assert selection_kind is SelectionKind.BOUNDING_BOX
    assert cloud.xyz.shape == (4, 3)


def test_object_cloud_requires_exactly_one_pixel_selection() -> None:
    depth = torch.ones((2, 2))
    image = ImageFrame(np.zeros((2, 2, 3), dtype=np.uint8), PixelFormat.RGB8)
    compute, memory = placement()
    arguments = {
        "min_depth_metres": 0.1,
        "max_depth_metres": 2,
        "sampling_stride": 1,
        "geometry_kind": GeometryKind.MEASURED_METRIC,
        "produced_on": compute,
        "memory": memory,
    }

    with pytest.raises(ValueError, match="exactly one"):
        extract_object_point_cloud(
            depth,
            image,
            source_header(),
            PinholeIntrinsics(2, 2, 1, 1, 0.5, 0.5),
            **arguments,
        )


def test_pca_box_recovers_axis_aligned_cuboid_pose_and_dimensions() -> None:
    center = torch.tensor((1.0, 2.0, 3.0))
    half_size = torch.tensor((2.0, 1.0, 0.5))
    signs = torch.tensor(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=torch.float32,
    )
    compute, memory = placement()
    cloud = PointCloud(
        source=source_header(),
        xyz=center + signs * half_size,
        colors_rgb=None,
        intrinsics=PinholeIntrinsics(4, 4, 2, 2, 1.5, 1.5),
        geometry_kind=GeometryKind.MEASURED_METRIC,
        sampling_stride=1,
        produced_on=compute,
        memory=memory,
    )

    bounds = estimate_observed_oriented_box(cloud, minimum_points=8)

    assert bounds.pose.position_metres == pytest.approx((1, 2, 3))
    assert bounds.size_metres == pytest.approx((4, 2, 1))
    assert bounds.pose.orientation_xyzw == pytest.approx((0, 0, 0, 1))


def test_observation_and_track_preserve_frame_time_and_unknown_velocity() -> None:
    compute, memory = placement()
    xyz = torch.tensor(
        [
            [-2, -1, -0.5],
            [-2, -1, 0.5],
            [-2, 1, -0.5],
            [-2, 1, 0.5],
            [2, -1, -0.5],
            [2, -1, 0.5],
            [2, 1, -0.5],
            [2, 1, 0.5],
        ],
        dtype=torch.float32,
    )
    cloud = PointCloud(
        source=source_header(),
        xyz=xyz,
        colors_rgb=None,
        intrinsics=PinholeIntrinsics(4, 4, 2, 2, 1.5, 1.5),
        geometry_kind=GeometryKind.MEASURED_METRIC,
        sampling_stride=1,
        produced_on=compute,
        memory=memory,
    )
    observation = estimate_object_observation(
        cloud,
        label="cardboard box",
        confidence=0.9,
        selection_kind=SelectionKind.SEGMENTATION_MASK,
        minimum_points=8,
    )
    assert observation.source.captured_at is not None
    track = ObjectTrack3D(
        track_id=7,
        observation=observation,
        state_time=observation.source.captured_at,
        status=TrackStatus.TENTATIVE,
    )

    assert track.observation.bounds.pose.reference_frame == FrameId("camera_optical")
    assert track.twist is None

    with pytest.raises(ValueError, match="same reference frame"):
        ObjectTrack3D(
            track_id=7,
            observation=observation,
            state_time=observation.source.captured_at,
            status=TrackStatus.TRACKED,
            twist=Twist3D(FrameId("world"), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        )
