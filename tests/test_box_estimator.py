from __future__ import annotations

import numpy as np
import pytest

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
    SensorSample,
    TimePoint,
)
from vision_pipeline.geometry.camera import PinholeIntrinsics
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception.objects import (
    TabletopBoxEstimator,
    TabletopBoxEstimatorConfig,
    WorkspaceBounds,
)
from vision_pipeline.rgbd import AlignedRgbdFrame, DepthFrame

OPTICAL = FrameId("realsense_color_optical")
CLOCK = ClockDomain("host/box-estimator-test", ClockKind.HOST_MONOTONIC)
WIDTH, HEIGHT = 160, 120
INTRINSICS = PinholeIntrinsics(WIDTH, HEIGHT, fx=200.0, fy=200.0, cx=79.5, cy=59.5)
TABLE_Z = 1.0
BOX_ROWS = slice(40, 80)
BOX_COLS = slice(60, 100)


def _header(measurement_kind: MeasurementKind) -> SampleHeader:
    return SampleHeader(
        sample_id="realsense/1",
        source_id="sensors/realsense-test/stream",
        sequence_number=1,
        measurement_kind=measurement_kind,
        captured_at=TimePoint(1_700_000_000_000, CLOCK),
        received_at=TimePoint(1_700_000_020_000, CLOCK),
        frame_id=OPTICAL,
        calibration=None,
        producer="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
    )


def _frame(*, with_box: bool) -> AlignedRgbdFrame:
    depth = np.full((HEIGHT, WIDTH), TABLE_Z, dtype=np.float32)
    color = np.full((HEIGHT, WIDTH, 3), 180, dtype=np.uint8)
    if with_box:
        rows = np.arange(HEIGHT)[BOX_ROWS]
        # A slight tilt keeps the visible top face from being perfectly flat, which
        # would make the PCA oriented-box degenerate along the viewing axis -- a real
        # depth camera almost never sees a perfectly fronto-parallel top face either.
        tilt = 0.001 * (rows - rows[0])
        depth[BOX_ROWS, BOX_COLS] = (0.85 - tilt)[:, None]
        color[BOX_ROWS, BOX_COLS] = (40, 40, 200)
    color_sample = SensorSample(
        header=_header(MeasurementKind.RGB_IMAGE),
        payload=ImageFrame(color, PixelFormat.BGR8),
        payload_memory=MemoryPlacement(MemoryKind.HOST),
    )
    depth_sample = SensorSample(
        header=_header(MeasurementKind.DEPTH_IMAGE),
        payload=DepthFrame(depth),
        payload_memory=MemoryPlacement(MemoryKind.HOST),
    )
    return AlignedRgbdFrame(
        frameset_id="test/frameset/0",
        color=color_sample,
        depth=depth_sample,
        color_intrinsics=INTRINSICS,
        aligned_depth_intrinsics=INTRINSICS,
        synchronization_method="test",
        alignment_method="test",
    )


def _config(**overrides: object) -> TabletopBoxEstimatorConfig:
    workspace = WorkspaceBounds(OPTICAL, (-0.5, -0.5, 0.3), (0.5, 0.5, 1.5))
    fields: dict[str, object] = {
        "workspace": workspace,
        "sampling_stride": 1,
        "minimum_cluster_points": 200,
        "minimum_plane_inliers": 200,
    }
    fields.update(overrides)
    return TabletopBoxEstimatorConfig(**fields)  # type: ignore[arg-type]


def test_estimate_returns_a_box_observation_and_matching_diagnostics() -> None:
    estimator = TabletopBoxEstimator(_config())

    observation = estimator.estimate(_frame(with_box=True))

    assert observation is not None
    assert observation.bounds.pose.reference_frame == OPTICAL
    assert observation.point_count > 0
    assert "ransac-plane" in observation.method
    assert observation.source.captured_at == _header(MeasurementKind.DEPTH_IMAGE).captured_at

    diagnostics = estimator.last_diagnostics
    assert diagnostics is not None
    full_points = int(diagnostics.full_cloud.xyz.shape[0])
    assert diagnostics.workspace_mask.shape == (full_points,)
    assert diagnostics.above_plane_mask.shape == (full_points,)
    assert diagnostics.cluster_mask.shape == (full_points,)
    assert int(diagnostics.workspace_mask.sum()) == diagnostics.segmentation.workspace_point_count
    assert (
        int(diagnostics.above_plane_mask.sum()) == diagnostics.segmentation.above_plane_point_count
    )
    assert int(diagnostics.cluster_mask.sum()) == diagnostics.segmentation.cluster_point_count
    assert diagnostics.segmentation.cluster_point_count == observation.point_count


def test_estimate_returns_none_but_keeps_the_scene_cloud_without_an_object() -> None:
    estimator = TabletopBoxEstimator(_config())
    estimator.estimate(_frame(with_box=True))
    assert estimator.last_diagnostics is not None

    observation = estimator.estimate(_frame(with_box=False))

    assert observation is None
    diagnostics = estimator.last_diagnostics
    assert diagnostics is not None
    assert diagnostics.segmentation is None
    assert diagnostics.above_plane_mask is None
    assert diagnostics.cluster_mask is None
    assert int(diagnostics.full_cloud.xyz.shape[0]) > 0
    assert diagnostics.workspace_mask.shape == (int(diagnostics.full_cloud.xyz.shape[0]),)


def test_config_rejects_an_inverted_depth_range() -> None:
    with pytest.raises(ValueError, match="max_depth_metres"):
        _config(min_depth_metres=1.0, max_depth_metres=0.5)


def test_config_rejects_a_non_positive_sampling_stride() -> None:
    with pytest.raises(ValueError, match="sampling_stride"):
        _config(sampling_stride=0)
