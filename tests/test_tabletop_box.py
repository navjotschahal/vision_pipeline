from __future__ import annotations

import math

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
from vision_pipeline.geometry import GeometryKind, PinholeIntrinsics, PointCloud
from vision_pipeline.geometry.spatial import RigidTransform3D
from vision_pipeline.perception.objects import (
    SelectionKind,
    TabletopSegmentationError,
    WorkspaceBounds,
    crop_to_workspace,
    estimate_tabletop_box,
    fit_support_plane,
    transform_point_cloud,
)

OPTICAL = FrameId("realsense_color_optical")
BASE = FrameId("panda_link0")
CLOCK = ClockDomain("host/tabletop-test", ClockKind.HOST_MONOTONIC)

# The synthetic scene is a fronto-parallel table at y = 0.5 m with one box resting on
# it, a much smaller second object, and clutter outside the workspace. Box side lengths
# are deliberately distinct so the PCA axes are not degenerate.
TABLE_Y = 0.5
BOX_EXTENTS = (0.20, 0.16, 0.30)
BOX_CENTRE = (0.0, 0.40, 1.05)


def source_header() -> SampleHeader:
    return SampleHeader(
        sample_id="realsense/17",
        source_id="sensors/realsense-146322071961/depth-aligned-to-color",
        sequence_number=17,
        measurement_kind=MeasurementKind.DEPTH_IMAGE,
        captured_at=TimePoint(1_700_000_000_000, CLOCK),
        received_at=TimePoint(1_700_000_020_000, CLOCK),
        frame_id=OPTICAL,
        calibration=None,
        producer="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
    )


def grid(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    mesh = torch.meshgrid(x, y, z, indexing="ij")
    return torch.stack([axis.reshape(-1) for axis in mesh], dim=1)


def table_points() -> torch.Tensor:
    return grid(
        torch.linspace(-0.5, 0.5, 101),
        torch.tensor([TABLE_Y]),
        torch.linspace(0.5, 1.5, 101),
    )


def box_points() -> torch.Tensor:
    return grid(
        torch.linspace(-0.10, 0.10, 21),
        torch.linspace(0.32, 0.48, 17),
        torch.linspace(0.90, 1.20, 31),
    )


def small_object_points() -> torch.Tensor:
    return grid(
        torch.linspace(0.30, 0.34, 5),
        torch.linspace(0.44, 0.48, 5),
        torch.linspace(0.60, 0.64, 5),
    )


def tiny_object_points() -> torch.Tensor:
    return grid(
        torch.linspace(0.30, 0.34, 3),
        torch.linspace(0.44, 0.48, 3),
        torch.linspace(0.60, 0.64, 3),
    )


def outside_workspace_points() -> torch.Tensor:
    return grid(
        torch.linspace(-0.2, 0.2, 5),
        torch.linspace(0.3, 0.5, 5),
        torch.linspace(2.0, 2.2, 5),
    )


def cloud_from(xyz: torch.Tensor, *, frame: FrameId = OPTICAL) -> PointCloud[torch.Tensor]:
    header = source_header()
    if frame != OPTICAL:
        header = SampleHeader(
            sample_id=header.sample_id,
            source_id=header.source_id,
            sequence_number=header.sequence_number,
            measurement_kind=header.measurement_kind,
            captured_at=header.captured_at,
            received_at=header.received_at,
            frame_id=frame,
            calibration=header.calibration,
            producer=header.producer,
            produced_on=header.produced_on,
        )
    return PointCloud(
        source=header,
        xyz=xyz.to(dtype=torch.float32),
        colors_rgb=None,
        intrinsics=PinholeIntrinsics(640, 480, fx=600, fy=600, cx=319.5, cy=239.5),
        geometry_kind=GeometryKind.MEASURED_METRIC,
        sampling_stride=1,
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
        memory=MemoryPlacement(MemoryKind.HOST),
    )


def scene() -> PointCloud[torch.Tensor]:
    return cloud_from(
        torch.cat(
            (
                table_points(),
                box_points(),
                small_object_points(),
                outside_workspace_points(),
            )
        )
    )


def workspace() -> WorkspaceBounds:
    return WorkspaceBounds(OPTICAL, (-0.6, -0.5, 0.4), (0.6, 0.6, 1.6))


def test_workspace_crop_keeps_only_points_inside_the_declared_region() -> None:
    cropped = crop_to_workspace(scene(), workspace())

    expected = len(table_points()) + len(box_points()) + len(small_object_points())
    assert int(cropped.xyz.shape[0]) == expected
    assert float(cropped.xyz[:, 2].max()) <= 1.6
    assert cropped.source.captured_at == scene().source.captured_at


def test_workspace_crop_rejects_bounds_declared_in_another_frame() -> None:
    bounds = WorkspaceBounds(BASE, (-0.6, -0.5, 0.4), (0.6, 0.6, 1.6))

    with pytest.raises(ValueError, match="does not match workspace frame"):
        crop_to_workspace(scene(), bounds)


def test_support_plane_recovers_the_table_not_a_box_face() -> None:
    cropped = crop_to_workspace(scene(), workspace())

    plane = fit_support_plane(cropped.xyz, distance_threshold_metres=0.01)

    assert abs(abs(plane.normal[1]) - 1.0) < 1e-6
    assert abs(plane.normal[0]) < 1e-6
    assert abs(plane.normal[2]) < 1e-6
    # normal . p + offset = 0 on the table, so |offset| is the table's y coordinate.
    assert abs(abs(plane.offset_metres) - TABLE_Y) < 1e-6
    assert plane.inlier_count == len(table_points())
    assert plane.rms_residual_metres < 1e-6


def test_largest_cluster_selects_the_box_over_the_smaller_object() -> None:
    result = estimate_tabletop_box(scene(), workspace())

    assert result.cluster_point_count == len(box_points())
    assert result.above_plane_point_count == len(box_points()) + len(small_object_points())
    assert result.workspace_point_count == (
        len(table_points()) + len(box_points()) + len(small_object_points())
    )


def test_estimated_box_recovers_extents_and_centre_and_keeps_capture_time() -> None:
    result = estimate_tabletop_box(scene(), workspace())
    observation = result.observation

    assert sorted(observation.bounds.size_metres, reverse=True) == pytest.approx(
        sorted(BOX_EXTENTS, reverse=True), abs=1e-6
    )
    assert observation.bounds.pose.position_metres == pytest.approx(BOX_CENTRE, abs=1e-6)
    assert observation.bounds.pose.reference_frame == OPTICAL
    assert observation.selection_kind is SelectionKind.FULL_CLOUD
    assert observation.point_count == len(box_points())
    assert "ransac-plane" in observation.method
    # The capture timestamp must survive the whole pipeline untouched.
    assert observation.source.captured_at == source_header().captured_at


def test_plane_normal_is_oriented_towards_the_object_side() -> None:
    result = estimate_tabletop_box(scene(), workspace())

    # Objects sit at smaller y than the table, so the object side is -y.
    assert result.plane.normal[1] < 0
    assert result.plane.signed_distance(torch.tensor([[0.0, 0.40, 1.05]])).item() > 0


def test_segmentation_is_deterministic_for_one_seed() -> None:
    first = estimate_tabletop_box(scene(), workspace(), seed=7)
    second = estimate_tabletop_box(scene(), workspace(), seed=7)

    assert first.observation.bounds == second.observation.bounds
    assert first.cluster_point_count == second.cluster_point_count


def test_a_scene_without_an_object_is_reported_rather_than_extrapolated() -> None:
    bare_table = cloud_from(table_points())

    with pytest.raises(TabletopSegmentationError):
        estimate_tabletop_box(bare_table, workspace())


def test_too_few_workspace_points_is_reported_rather_than_extrapolated() -> None:
    sparse = cloud_from(tiny_object_points())

    with pytest.raises(TabletopSegmentationError, match="workspace"):
        estimate_tabletop_box(sparse, workspace())


def test_transform_point_cloud_matches_the_scalar_rigid_transform() -> None:
    half = math.sqrt(0.5)
    transform = RigidTransform3D(
        source_frame=OPTICAL,
        target_frame=BASE,
        translation_metres=(0.4, -0.1, 0.8),
        rotation_xyzw=(0.0, 0.0, half, half),
    )
    points = torch.tensor(
        [[0.0, 0.40, 1.05], [-0.1, 0.32, 0.90], [0.1, 0.48, 1.20]],
        dtype=torch.float32,
    )

    moved = transform_point_cloud(cloud_from(points), transform)

    assert moved.source.frame_id == BASE
    assert moved.source.captured_at == source_header().captured_at
    for index in range(points.shape[0]):
        original = tuple(float(value) for value in points[index])
        assert tuple(float(value) for value in moved.xyz[index]) == pytest.approx(
            transform.apply_point(original), abs=1e-6
        )


def test_transform_point_cloud_rejects_a_mismatched_source_frame() -> None:
    transform = RigidTransform3D(
        source_frame=BASE,
        target_frame=OPTICAL,
        translation_metres=(0.0, 0.0, 0.0),
        rotation_xyzw=(0.0, 0.0, 0.0, 1.0),
    )

    with pytest.raises(ValueError, match="does not match transform source frame"):
        transform_point_cloud(scene(), transform)
