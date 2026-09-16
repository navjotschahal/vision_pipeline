from __future__ import annotations

import numpy as np
import pytest
from rgbd_fixtures import (
    BOX_RELIEF,
    BOX_TOP_DEPTH,
    FRAME_ID,
    HEIGHT,
    INTRINSICS,
    ROD,
    WIDTH,
    make_frame,
    tabletop_scene,
)

from vision_pipeline.contracts import FrameId
from vision_pipeline.perception.objects import WorkspaceBounds
from vision_pipeline.perception.objects.selected_geometry import (
    GeometryRejection,
    SelectedGeometryConfig,
    SelectedGeometryError,
    SelectedGeometryEstimator,
    SupportPlaneStatus,
    depth_continuous_components,
    neighbour_counts,
)
from vision_pipeline.perception.objects.selection import SegmentationResult
from vision_pipeline.rgbd import AlignedRgbdFrame


def _result(frame: AlignedRgbdFrame, mask: np.ndarray, score: float = 0.9) -> SegmentationResult:
    return SegmentationResult(frame.frameset_id, frame.color.header, mask, score, "test")


def _config(**overrides: object) -> SelectedGeometryConfig:
    values: dict[str, object] = {
        "workspace": WorkspaceBounds(FRAME_ID, (-1.0, -1.0, 0.2), (1.0, 1.0, 2.0)),
        "min_depth_metres": 0.2,
        "max_depth_metres": 2.0,
        "minimum_points": 20,
    }
    values.update(overrides)
    return SelectedGeometryConfig(**values)  # type: ignore[arg-type]


def _leaky_mask(scene_mask: np.ndarray) -> np.ndarray:
    """The box plus a two-pixel ring that bleeds onto the table around it."""

    leaked = scene_mask.copy()
    rows, columns = np.nonzero(scene_mask)
    leaked[rows.min() - 2 : rows.max() + 3, columns.min() - 2 : columns.max() + 3] = True
    return leaked


def test_mask_and_same_frame_depth_give_timestamped_metric_geometry() -> None:
    scene = tabletop_scene(sequence=4)
    mask = _leaky_mask(scene.box_mask) | scene.rod_mask | scene.far_patch_mask | scene.flying_mask
    estimator = SelectedGeometryEstimator(_config())

    estimate = estimator.estimate(scene.frame, _result(scene.frame, mask), selection_id=3)

    # Provenance: the aligned depth header, the exact color sample, the frameset.
    assert estimate.observation.source == scene.frame.depth.header
    assert estimate.depth_source.captured_at == scene.frame.depth.header.captured_at
    assert estimate.color_source == scene.frame.color.header
    assert estimate.frameset_id == scene.frame.frameset_id
    assert estimate.selection_id == 3 and estimate.mask_backend == "test"
    assert estimate.observation.bounds.pose.reference_frame == FRAME_ID
    # Table leakage, the far patch, and flying pixels are gone; only the box top and rod remain.
    xyz = estimate.cloud.xyz.numpy()
    assert xyz[:, 2].min() >= BOX_TOP_DEPTH - 1e-6
    assert xyz[:, 2].max() <= BOX_TOP_DEPTH + BOX_RELIEF + 1e-6
    assert estimate.support_plane is not None
    assert estimate.quality.support_plane_status is SupportPlaneStatus.FITTED
    assert estimate.support_plane.normal == pytest.approx((0.0, 0.0, -1.0), abs=1e-6)
    assert estimate.support_plane.offset_metres == pytest.approx(1.0, abs=1e-6)
    assert estimate.quality.valid_depth_fraction < 1.0  # zero-depth pixels were rejected
    assert estimate.quality.component_points < estimate.quality.valid_depth_points
    assert estimate.quality.above_support_points < estimate.quality.workspace_points
    assert estimate.centroid_metres[2] == pytest.approx(BOX_TOP_DEPTH + BOX_RELIEF / 2, abs=2e-3)


def test_thin_structure_survives_outlier_cleanup() -> None:
    scene = tabletop_scene()
    mask = scene.box_mask | scene.rod_mask
    estimate = SelectedGeometryEstimator(_config()).estimate(
        scene.frame, _result(scene.frame, mask), selection_id=1
    )

    x0, y0, x1, _ = ROD
    depth = scene.frame.depth.payload.data
    rod_columns = np.arange(x0, x1, 2)
    rod_depth = depth[y0, rod_columns]
    expected_x = (rod_columns - INTRINSICS.cx) * rod_depth / INTRINSICS.fx
    xyz = estimate.cloud.xyz.numpy()
    expected_y = (y0 - INTRINSICS.cy) * rod_depth / INTRINSICS.fy
    for x, y, z in zip(expected_x, expected_y, rod_depth, strict=True):
        assert np.any(np.all(np.isclose(xyz, (x, y, z), atol=1e-5), axis=1))
    assert estimate.quality.isolated_points_removed == 0
    size = sorted(estimate.observation.bounds.size_metres)
    assert size[-1] > 0.4  # the rod lengthens the observed bounds (box alone is 0.25 m)


def test_mask_without_valid_depth_is_rejected() -> None:
    scene = tabletop_scene()
    depth = scene.frame.depth.payload.data.copy()
    depth[scene.box_mask] = 0.0
    frame = make_frame(0, scene.frame.color.payload.data, depth)

    with pytest.raises(SelectedGeometryError) as error:
        SelectedGeometryEstimator(_config()).estimate(
            frame, _result(frame, scene.box_mask), selection_id=1
        )
    assert error.value.reason is GeometryRejection.NO_VALID_DEPTH


def test_workspace_and_table_constraints_reject_out_of_bounds_selections() -> None:
    scene = tabletop_scene()
    far_workspace = _config(workspace=WorkspaceBounds(FRAME_ID, (-1, -1, 1.2), (1, 1, 2)))
    with pytest.raises(SelectedGeometryError) as outside:
        SelectedGeometryEstimator(far_workspace).estimate(
            scene.frame, _result(scene.frame, scene.box_mask), selection_id=1
        )
    assert outside.value.reason is GeometryRejection.OUTSIDE_WORKSPACE

    table_only = np.zeros((HEIGHT, WIDTH), np.bool_)
    table_only[90:110, 10:40] = True
    with pytest.raises(SelectedGeometryError) as flat:
        SelectedGeometryEstimator(_config()).estimate(
            scene.frame, _result(scene.frame, table_only), selection_id=1
        )
    assert flat.value.reason is GeometryRejection.AT_OR_BELOW_SUPPORT


def test_mask_from_another_frame_is_refused() -> None:
    scene = tabletop_scene(sequence=1)
    other = tabletop_scene(sequence=2).frame
    with pytest.raises(ValueError, match="does not belong"):
        SelectedGeometryEstimator(_config()).estimate(
            scene.frame, _result(other, scene.box_mask), selection_id=1
        )


def test_workspace_frame_must_match_depth_frame() -> None:
    scene = tabletop_scene()
    config = _config(workspace=WorkspaceBounds(FrameId("robot_base"), (-1, -1, 0), (1, 1, 2)))
    with pytest.raises(ValueError, match="does not match workspace"):
        SelectedGeometryEstimator(config).estimate(
            scene.frame, _result(scene.frame, scene.box_mask), selection_id=1
        )


def test_plane_above_part_of_the_object_is_not_used_for_clearance() -> None:
    # A shelf-like surface at 0.85 m fills the search ring; the object reaches 0.95 m.
    depth = np.full((HEIGHT, WIDTH), 0.85, dtype=np.float32)
    mask = np.zeros((HEIGHT, WIDTH), np.bool_)
    mask[40:80, 60:100] = True
    depth[40:80, 60:100] = np.linspace(0.90, 0.95, 40, dtype=np.float32)[None, :]
    frame = make_frame(0, depth=depth)

    estimate = SelectedGeometryEstimator(_config()).estimate(
        frame, _result(frame, mask), selection_id=1
    )

    assert estimate.support_plane is None
    assert estimate.quality.support_plane_status is SupportPlaneStatus.ABOVE_OBJECT
    assert estimate.quality.final_points == estimate.quality.workspace_points


def test_depth_continuous_components_split_on_depth_jumps_only() -> None:
    depth = np.array([[1.0, 1.01, 1.5, 1.5], [1.0, 1.0, 1.5, 1.5]], dtype=np.float32)
    valid = np.ones_like(depth, dtype=np.bool_)
    valid[1, 3] = False

    count, labels = depth_continuous_components(
        depth, valid, step_fraction=0.03, step_floor_metres=0.01
    )

    assert count == 3  # background label plus two surfaces
    assert labels[0, 0] == labels[0, 1] == labels[1, 0] == labels[1, 1] != 0
    assert labels[0, 2] == labels[0, 3] == labels[1, 2] != labels[0, 0]
    assert labels[1, 3] == 0


def test_neighbour_counts_keep_lines_and_isolate_strays() -> None:
    line = np.stack((np.arange(10) * 0.004, np.zeros(10), np.ones(10)), axis=1)
    stray = np.array([[0.5, 0.5, 1.5]])

    counts = neighbour_counts(np.vstack((line, stray)), 0.005)

    assert counts[-1] == 0
    assert counts[1:9].min() >= 2
