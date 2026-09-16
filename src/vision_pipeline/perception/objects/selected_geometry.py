"""Metric 3D geometry of a click-selected object from its mask and the same frame's depth.

The mask and the depth plane must come from one :class:`~vision_pipeline.rgbd.AlignedRgbdFrame`;
that is checked, not assumed. Stages run in this order, and each one's surviving point
count is reported in :class:`SelectedGeometryQuality`:

1. **Valid depth.** Mask pixels with finite depth inside the configured range, sampled
   on the deprojection grid. Zero depth is RealSense's "no measurement".
2. **Depth-continuous components.** Mask pixels are grouped by 4-neighbour continuity
   whose allowed depth step grows with range. The largest group is kept, together with
   any comparably large group at a similar depth (an object split by a thin occluder).
   Mask leakage onto a surface far behind the object is discarded. No pixel is removed
   merely for lying on a boundary, so thin parts survive.
3. **Workspace.** Points outside explicit bounds, in the same optical frame, are dropped.
4. **Support clearance.** A plane is fitted by RANSAC to *non-mask* depth in a ring
   around the object's image region (so neither the object's own face nor a distant
   parallel surface such as a shelf or another object's top wins the fit), oriented
   toward the camera, and rejected if it passes above part of the object (a clearance
   cut would then erase real object points). Points within ``table_clearance_metres``
   of an accepted plane, or behind it, are dropped. Without an accepted plane the stage
   is skipped and reported as such.
5. **Isolated points.** Points with too few 3D neighbours inside a range-adaptive radius
   are dropped. Connected thin structures keep neighbours along their length; flying
   pixels at depth edges do not.

Everything is expressed in the color camera's optical frame. The oriented box is PCA of
the *visible* surface: it is not an object pose, and its axes are ambiguous for
symmetric objects. ``object_score`` is the segmenter's presence probability, not a
geometric confidence.
"""

from __future__ import annotations

import dataclasses
import math
import os
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from vision_pipeline.contracts import (
    ComputeKind,
    ComputePlacement,
    MemoryKind,
    MemoryPlacement,
    SampleHeader,
)
from vision_pipeline.geometry.pointcloud import GeometryKind, PointCloud, depth_to_point_cloud
from vision_pipeline.geometry.spatial import Vector3
from vision_pipeline.rgbd import AlignedRgbdFrame

from .contracts import ObjectObservation3D, SelectionKind
from .geometry import estimate_object_observation, extract_object_point_cloud
from .selection import SegmentationResult, require_matching_frame
from .tabletop import (
    SupportPlane,
    TabletopSegmentationError,
    WorkspaceBounds,
    crop_to_workspace,
    fit_support_plane,
)

METHOD = (
    "click-mask+valid-depth+depth-continuous-components+workspace+support-clearance"
    "+neighbour-count+pca-observed-oriented-box"
)


class SupportPlaneStatus(StrEnum):
    FITTED = "fitted"
    CACHED = "cached"
    NOT_FOUND = "not_found"
    ABOVE_OBJECT = "above_object"


class GeometryRejection(StrEnum):
    NO_VALID_DEPTH = "no_valid_depth"
    OUTSIDE_WORKSPACE = "outside_workspace"
    NO_SUPPORT_PLANE = "no_support_plane"
    AT_OR_BELOW_SUPPORT = "at_or_below_support"
    TOO_FEW_POINTS = "too_few_points"
    DEGENERATE = "degenerate"


class SelectedGeometryError(ValueError):
    """The selected mask did not yield trustworthy 3D geometry on this frame."""

    def __init__(self, reason: GeometryRejection, message: str) -> None:
        super().__init__(f"{reason.value}: {message}")
        self.reason = reason


def _require_positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


def _require_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class SelectedGeometryConfig:
    """Filtering policy; distances are metres in the color optical frame.

    ``workspace`` of ``None`` explicitly disables the workspace constraint. With
    ``require_support_plane`` false, a frame without an accepted plane is still accepted
    and reports ``support_plane=None`` and its reason instead of silently skipping the
    clearance check. The plane search ring extends ``plane_search_radius_scale`` times
    the square root of the mask area (clamped to the pixel limits) beyond the mask. A
    plane is rejected when the 5th percentile of the object's signed distances to it is
    below ``-maximum_plane_intrusion_metres``, i.e. the plane lies above part of the
    object. An accepted plane is reused for up to ``plane_refit_interval_frames`` frames
    while that check still passes, which assumes the camera does not move relative to
    the table in between.
    """

    workspace: WorkspaceBounds | None
    min_depth_metres: float = 0.1
    max_depth_metres: float = 2.0
    sampling_stride: int = 2
    depth_step_fraction: float = 0.03
    depth_step_floor_metres: float = 0.01
    component_min_fraction: float = 0.15
    component_depth_tolerance_metres: float = 0.10
    require_support_plane: bool = False
    table_clearance_metres: float = 0.01
    plane_distance_threshold_metres: float = 0.008
    plane_sampling_stride: int = 4
    plane_iterations: int = 60
    minimum_plane_inliers: int = 150
    plane_search_radius_scale: float = 1.0
    plane_search_min_radius_pixels: int = 30
    plane_search_max_radius_pixels: int = 200
    maximum_plane_intrusion_metres: float = 0.05
    plane_refit_interval_frames: int = 10
    neighbour_radius_spacings: float = 4.0
    minimum_neighbour_radius_metres: float = 0.005
    minimum_neighbours: int = 3
    minimum_points: int = 30
    label: str = "click-selected-object"

    def __post_init__(self) -> None:
        if self.workspace is not None and not isinstance(self.workspace, WorkspaceBounds):
            raise TypeError("workspace must be WorkspaceBounds or None")
        for name in (
            "min_depth_metres",
            "max_depth_metres",
            "depth_step_fraction",
            "depth_step_floor_metres",
            "component_min_fraction",
            "component_depth_tolerance_metres",
            "table_clearance_metres",
            "plane_distance_threshold_metres",
            "plane_search_radius_scale",
            "maximum_plane_intrusion_metres",
            "neighbour_radius_spacings",
            "minimum_neighbour_radius_metres",
        ):
            _require_positive_finite(getattr(self, name), name)
        if self.max_depth_metres <= self.min_depth_metres:
            raise ValueError("max_depth_metres must exceed min_depth_metres")
        if self.plane_search_max_radius_pixels < self.plane_search_min_radius_pixels:
            raise ValueError("plane search radius limits are inverted")
        if self.component_min_fraction > 1:
            raise ValueError("component_min_fraction must not exceed one")
        for name in (
            "sampling_stride",
            "plane_sampling_stride",
            "plane_iterations",
            "minimum_plane_inliers",
            "plane_search_min_radius_pixels",
            "plane_search_max_radius_pixels",
            "plane_refit_interval_frames",
            "minimum_neighbours",
            "minimum_points",
        ):
            _require_positive_int(getattr(self, name), name)
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")


@dataclass(frozen=True, slots=True)
class SelectedGeometryQuality:
    """Measured evidence for one estimate; counts are on the sampled deprojection grid."""

    mask_pixels: int
    sampled_mask_pixels: int
    valid_depth_points: int
    valid_depth_fraction: float
    component_points: int
    components_found: int
    workspace_points: int
    above_support_points: int
    final_points: int
    isolated_points_removed: int
    depth_median_metres: float
    depth_mad_metres: float
    neighbour_radius_metres: float
    support_plane_status: SupportPlaneStatus
    support_plane_rms_metres: float | None
    support_plane_inliers: int | None
    support_plane_age_frames: int | None


@dataclass(frozen=True, slots=True)
class SelectedObjectGeometry:
    """Timestamped metric geometry of the selected object in the color optical frame.

    ``observation.source`` is the aligned depth sample header (capture time, frame ID,
    calibration); ``color_source`` is the color sample the mask was computed from. Both
    belong to ``frameset_id``.
    """

    observation: ObjectObservation3D
    cloud: PointCloud[torch.Tensor]
    frameset_id: str
    color_source: SampleHeader
    selection_id: int
    mask_backend: str
    object_score: float
    centroid_metres: Vector3
    covariance_metres2: tuple[float, ...]
    support_plane: SupportPlane | None
    quality: SelectedGeometryQuality
    method: str = METHOD

    def __post_init__(self) -> None:
        if self.observation.source != self.cloud.source:
            raise ValueError("observation and cloud must share one depth source header")
        if self.observation.source.frame_id != self.color_source.frame_id:
            raise ValueError("depth and color sources must share one optical frame")
        if len(self.covariance_metres2) != 9 or not all(
            math.isfinite(value) for value in self.covariance_metres2
        ):
            raise ValueError("covariance_metres2 must be 9 finite row-major values")
        if not all(math.isfinite(value) for value in self.centroid_metres):
            raise ValueError("centroid_metres must be finite")

    @property
    def depth_source(self) -> SampleHeader:
        return self.observation.source


@dataclass(frozen=True, slots=True)
class SelectedGeometryDiagnostics:
    """Point-level evidence for a viewer; not part of the estimate contract.

    ``stage_cloud`` holds every valid-depth mask point, and ``stage`` gives the last
    stage each point survived (0 valid depth, 1 component, 2 workspace, 3 above
    support, 4 final). ``background_xyz``/``plane_inliers`` show the support fit sample.
    """

    frameset_id: str
    stage_cloud: NDArray[np.float32]
    stage: NDArray[np.uint8]
    background_xyz: NDArray[np.float32] | None
    plane_inliers: NDArray[np.bool_] | None


def _placement() -> tuple[ComputePlacement, MemoryPlacement]:
    return (
        ComputePlacement(ComputeKind.HOST_CPU, f"host-process-{os.getpid()}/selected-geometry"),
        MemoryPlacement(MemoryKind.HOST),
    )


def depth_continuous_components(
    depth_metres: NDArray[np.float32],
    valid: NDArray[np.bool_],
    *,
    step_fraction: float,
    step_floor_metres: float,
) -> tuple[int, NDArray[np.int32]]:
    """Label 4-connected valid pixels whose neighbouring depth step is small enough.

    The allowed step is ``max(step_floor_metres, step_fraction * depth)``. Labels are
    computed on a 2x lattice whose in-between cells encode connectivity, so boundary
    pixels are never discarded to separate regions. Returns ``(count, labels)`` with 0
    meaning invalid, following OpenCV.
    """

    if depth_metres.shape != valid.shape or depth_metres.ndim != 2:
        raise ValueError("depth and valid mask must be matching HxW arrays")
    height, width = valid.shape
    depth = np.where(valid, depth_metres, np.float32(0))
    allowed = np.maximum(step_floor_metres, step_fraction * depth)
    right = (
        valid[:, 1:]
        & valid[:, :-1]
        & (np.abs(depth[:, 1:] - depth[:, :-1]) <= np.minimum(allowed[:, 1:], allowed[:, :-1]))
    )
    down = (
        valid[1:, :]
        & valid[:-1, :]
        & (np.abs(depth[1:, :] - depth[:-1, :]) <= np.minimum(allowed[1:, :], allowed[:-1, :]))
    )
    lattice = np.zeros((2 * height - 1, 2 * width - 1), dtype=np.uint8)
    lattice[::2, ::2] = valid
    lattice[::2, 1::2] = right
    lattice[1::2, ::2] = down
    count, labels = cv2.connectedComponents(lattice, connectivity=4)
    return int(count), cast(NDArray[np.int32], labels[::2, ::2].astype(np.int32))


def neighbour_counts(xyz: NDArray[np.float64], radius_metres: float) -> NDArray[np.int64]:
    """Count other points in the 3x3x3 block of ``radius_metres`` voxels around each point.

    This deterministic grid approximation of a radius query needs no KD-tree dependency:
    every true neighbour within ``radius_metres`` is counted, plus some up to about
    ``2*sqrt(3)`` radii away.
    """

    _require_positive_finite(radius_metres, "radius_metres")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have [N, 3] shape")
    if xyz.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    cells = np.floor(xyz / radius_metres).astype(np.int64)
    cells -= cells.min(axis=0) - 1
    span = int(cells.max()) + 2
    keys = (cells[:, 0] * span + cells[:, 1]) * span + cells[:, 2]
    unique, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    totals = np.zeros(unique.shape[0], dtype=np.int64)
    unique_cells = np.stack(
        (unique // (span * span), (unique // span) % span, unique % span), axis=1
    )
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                neighbour = unique_cells + (dx, dy, dz)
                neighbour_keys = (neighbour[:, 0] * span + neighbour[:, 1]) * span + neighbour[:, 2]
                position = np.searchsorted(unique, neighbour_keys)
                position = np.minimum(position, unique.shape[0] - 1)
                found = unique[position] == neighbour_keys
                totals += np.where(found, counts[position], 0)
    return cast(NDArray[np.int64], totals[inverse.reshape(-1)] - 1)


def _select(cloud: PointCloud[torch.Tensor], keep: NDArray[np.bool_]) -> PointCloud[torch.Tensor]:
    mask = torch.from_numpy(keep)
    colors = cloud.colors_rgb
    return dataclasses.replace(
        cloud, xyz=cloud.xyz[mask], colors_rgb=None if colors is None else colors[mask]
    )


class SelectedGeometryEstimator:
    """Stateful only for the cached support plane; call :meth:`reset` on reselection."""

    def __init__(self, config: SelectedGeometryConfig) -> None:
        if not isinstance(config, SelectedGeometryConfig):
            raise TypeError("config must be a SelectedGeometryConfig")
        self._config = config
        self._plane: SupportPlane | None = None
        self._plane_age = 0
        self._plane_sample: tuple[NDArray[np.float32], NDArray[np.bool_]] | None = None
        self._last_diagnostics: SelectedGeometryDiagnostics | None = None

    @property
    def config(self) -> SelectedGeometryConfig:
        return self._config

    @property
    def last_diagnostics(self) -> SelectedGeometryDiagnostics | None:
        return self._last_diagnostics

    def reset(self) -> None:
        self._plane = None
        self._plane_age = 0
        self._plane_sample = None

    def estimate(
        self,
        frame: AlignedRgbdFrame,
        segmentation: SegmentationResult,
        *,
        selection_id: int,
    ) -> SelectedObjectGeometry:
        require_matching_frame(segmentation, frame)
        config = self._config
        workspace = config.workspace
        if workspace is not None and frame.depth.header.frame_id != workspace.reference_frame:
            raise ValueError(
                f"depth frame {frame.depth.header.frame_id.value!r} does not match workspace "
                f"frame {workspace.reference_frame.value!r}"
            )
        stride = config.sampling_stride
        depth = frame.depth.payload.data
        mask = segmentation.mask
        grid_depth = depth[::stride, ::stride]
        grid_mask = mask[::stride, ::stride]
        grid_valid = (
            grid_mask
            & np.isfinite(grid_depth)
            & (grid_depth >= config.min_depth_metres)
            & (grid_depth <= config.max_depth_metres)
        )
        sampled_mask_pixels = int(grid_mask.sum())
        valid_points = int(grid_valid.sum())
        if valid_points == 0:
            self._last_diagnostics = None
            raise SelectedGeometryError(
                GeometryRejection.NO_VALID_DEPTH,
                f"none of {sampled_mask_pixels} sampled mask pixels has valid depth",
            )

        count, labels = depth_continuous_components(
            grid_depth,
            grid_valid,
            step_fraction=config.depth_step_fraction,
            step_floor_metres=config.depth_step_floor_metres,
        )
        sizes = np.bincount(labels.reshape(-1), minlength=count)
        sizes[0] = 0
        largest = int(np.argmax(sizes))
        largest_depth = float(np.median(grid_depth[labels == largest]))
        kept_labels = [largest]
        for label in np.flatnonzero(sizes >= config.component_min_fraction * sizes[largest]):
            if label == largest:
                continue
            component_depth = float(np.median(grid_depth[labels == label]))
            if abs(component_depth - largest_depth) <= config.component_depth_tolerance_metres:
                kept_labels.append(int(label))
        grid_kept = np.isin(labels, kept_labels) & grid_valid

        compute, memory = _placement()
        depth_tensor = torch.from_numpy(depth)
        stage_cloud, _ = extract_object_point_cloud(
            depth_tensor,
            frame.color.payload,
            frame.depth.header,
            frame.aligned_depth_intrinsics,
            segmentation_mask=torch.from_numpy(mask),
            min_depth_metres=config.min_depth_metres,
            max_depth_metres=config.max_depth_metres,
            sampling_stride=stride,
            geometry_kind=GeometryKind.MEASURED_METRIC,
            produced_on=compute,
            memory=memory,
        )
        if stage_cloud.xyz.shape[0] != valid_points:
            raise RuntimeError("deprojection and grid validity disagree on valid depth pixels")
        # Stage bookkeeping aligned to the valid-depth cloud's point order (row-major).
        stage = np.zeros(stage_cloud.xyz.shape[0], dtype=np.uint8)
        in_component = grid_kept[grid_valid]
        stage[in_component] = 1
        cloud = _select(stage_cloud, in_component)
        component_points = int(in_component.sum())

        if workspace is not None:
            inside = workspace.contains(cloud.xyz).numpy()
            cloud = _select(cloud, inside)
            stage[np.flatnonzero(in_component)[inside]] = 2
        else:
            stage[in_component] = 2
        workspace_points = int(cloud.xyz.shape[0])
        if workspace_points == 0:
            self._record(frame, stage_cloud, stage)
            raise SelectedGeometryError(
                GeometryRejection.OUTSIDE_WORKSPACE, "no selected points inside the workspace"
            )
        surviving = np.flatnonzero(stage == 2)

        plane, plane_status = self._support_plane(frame, mask, depth_tensor, cloud.xyz)
        if plane is None:
            if config.require_support_plane:
                self._record(frame, stage_cloud, stage)
                raise SelectedGeometryError(
                    GeometryRejection.NO_SUPPORT_PLANE,
                    f"no support plane under the object ({plane_status.value})",
                )
        else:
            above = (plane.signed_distance(cloud.xyz) > config.table_clearance_metres).numpy()
            cloud = _select(cloud, above)
            surviving = surviving[above]
        stage[surviving] = 3
        above_points = int(cloud.xyz.shape[0])
        if above_points == 0:
            self._record(frame, stage_cloud, stage)
            raise SelectedGeometryError(
                GeometryRejection.AT_OR_BELOW_SUPPORT,
                "every selected point lies within the table clearance or behind the table",
            )

        xyz = cloud.xyz.numpy().astype(np.float64)
        depth_median = float(np.median(xyz[:, 2]))
        depth_mad = float(np.median(np.abs(xyz[:, 2] - depth_median)))
        intrinsics = frame.aligned_depth_intrinsics
        spacing = depth_median * stride / min(intrinsics.fx, intrinsics.fy)
        radius = max(
            config.minimum_neighbour_radius_metres, config.neighbour_radius_spacings * spacing
        )
        connected = neighbour_counts(xyz, radius) >= config.minimum_neighbours
        cloud = _select(cloud, connected)
        surviving = surviving[connected]
        stage[surviving] = 4
        final_points = int(cloud.xyz.shape[0])
        self._record(frame, stage_cloud, stage)
        if final_points < config.minimum_points:
            raise SelectedGeometryError(
                GeometryRejection.TOO_FEW_POINTS,
                f"{final_points} points survived filtering; {config.minimum_points} required",
            )
        try:
            observation = estimate_object_observation(
                cloud,
                label=config.label,
                confidence=segmentation.object_score,
                selection_kind=SelectionKind.SEGMENTATION_MASK,
                minimum_points=config.minimum_points,
            )
        except ValueError as error:
            raise SelectedGeometryError(GeometryRejection.DEGENERATE, str(error)) from error

        final_xyz = xyz[connected]
        centroid = final_xyz.mean(axis=0)
        covariance = np.cov(final_xyz.T) if final_points > 1 else np.zeros((3, 3))
        return SelectedObjectGeometry(
            observation=observation,
            cloud=cloud,
            frameset_id=frame.frameset_id,
            color_source=frame.color.header,
            selection_id=selection_id,
            mask_backend=segmentation.backend,
            object_score=segmentation.object_score,
            centroid_metres=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
            covariance_metres2=tuple(float(value) for value in covariance.reshape(-1)),
            support_plane=plane,
            quality=SelectedGeometryQuality(
                mask_pixels=segmentation.mask_pixels,
                sampled_mask_pixels=sampled_mask_pixels,
                valid_depth_points=valid_points,
                valid_depth_fraction=valid_points / max(1, sampled_mask_pixels),
                component_points=component_points,
                components_found=int(np.count_nonzero(sizes)),
                workspace_points=workspace_points,
                above_support_points=above_points,
                final_points=final_points,
                isolated_points_removed=above_points - final_points,
                depth_median_metres=depth_median,
                depth_mad_metres=depth_mad,
                neighbour_radius_metres=radius,
                support_plane_status=plane_status,
                support_plane_rms_metres=None if plane is None else plane.rms_residual_metres,
                support_plane_inliers=None if plane is None else plane.inlier_count,
                support_plane_age_frames=None if plane is None else self._plane_age,
            ),
        )

    def _record(
        self,
        frame: AlignedRgbdFrame,
        stage_cloud: PointCloud[torch.Tensor],
        stage: NDArray[np.uint8],
    ) -> None:
        background, inliers = self._plane_sample or (None, None)
        self._last_diagnostics = SelectedGeometryDiagnostics(
            frameset_id=frame.frameset_id,
            stage_cloud=stage_cloud.xyz.numpy().astype(np.float32),
            stage=stage.copy(),
            background_xyz=background,
            plane_inliers=inliers,
        )

    def _supports(self, plane: SupportPlane, object_xyz: torch.Tensor) -> bool:
        """False when the plane passes above more than a sliver of the object."""

        distances = plane.signed_distance(object_xyz.to(dtype=torch.float64))
        lowest = float(torch.quantile(distances, 0.05))
        return lowest >= -self._config.maximum_plane_intrusion_metres

    def _support_plane(
        self,
        frame: AlignedRgbdFrame,
        mask: NDArray[np.bool_],
        depth_tensor: torch.Tensor,
        object_xyz: torch.Tensor,
    ) -> tuple[SupportPlane | None, SupportPlaneStatus]:
        config = self._config
        cached = self._plane
        if (
            cached is not None
            and self._plane_age + 1 < config.plane_refit_interval_frames
            and self._supports(cached, object_xyz)
        ):
            self._plane_age += 1
            return cached, SupportPlaneStatus.CACHED
        self._plane = None
        self._plane_sample = None
        stride = config.plane_sampling_stride
        grid_mask = mask[::stride, ::stride].astype(np.uint8)
        radius = math.sqrt(float(np.count_nonzero(mask))) * config.plane_search_radius_scale
        radius = min(
            max(radius, config.plane_search_min_radius_pixels),
            config.plane_search_max_radius_pixels,
        )
        outer = max(1, round(radius / stride))
        # A two-cell margin keeps mask-boundary (mixed depth) pixels out of the fit.
        ring = (cv2.dilate(grid_mask, np.ones((2 * outer + 1, 2 * outer + 1), np.uint8)) > 0) & ~(
            cv2.dilate(grid_mask, np.ones((5, 5), np.uint8)) > 0
        )
        selection = np.zeros(mask.shape, dtype=np.bool_)
        selection[::stride, ::stride] = ring
        compute, memory = _placement()
        background = depth_to_point_cloud(
            depth_tensor,
            frame.color.payload,
            frame.depth.header,
            frame.aligned_depth_intrinsics,
            min_depth_metres=config.min_depth_metres,
            max_depth_metres=config.max_depth_metres,
            sampling_stride=stride,
            geometry_kind=GeometryKind.MEASURED_METRIC,
            produced_on=compute,
            memory=memory,
            selection_mask=torch.from_numpy(selection),
        )
        if config.workspace is not None:
            background = crop_to_workspace(background, config.workspace)
        try:
            plane = fit_support_plane(
                background.xyz,
                distance_threshold_metres=config.plane_distance_threshold_metres,
                iterations=config.plane_iterations,
                minimum_inliers=config.minimum_plane_inliers,
                maximum_scoring_points=4_000,
            )
        except TabletopSegmentationError:
            return None, SupportPlaneStatus.NOT_FOUND
        if plane.offset_metres < 0:
            # The camera origin must lie on the positive (free-space) side of the support.
            plane = dataclasses.replace(
                plane,
                normal=(-plane.normal[0], -plane.normal[1], -plane.normal[2]),
                offset_metres=-plane.offset_metres,
            )
        background_xyz = background.xyz.numpy().astype(np.float32)
        inliers = (
            np.abs(plane.signed_distance(background.xyz).numpy())
            <= config.plane_distance_threshold_metres
        )
        self._plane_sample = (background_xyz, inliers)
        if not self._supports(plane, object_xyz):
            return None, SupportPlaneStatus.ABOVE_OBJECT
        self._plane = plane
        self._plane_age = 0
        return plane, SupportPlaneStatus.FITTED


__all__ = [
    "METHOD",
    "GeometryRejection",
    "SelectedGeometryConfig",
    "SelectedGeometryDiagnostics",
    "SelectedGeometryError",
    "SelectedGeometryEstimator",
    "SelectedGeometryQuality",
    "SelectedObjectGeometry",
    "SupportPlaneStatus",
    "depth_continuous_components",
    "neighbour_counts",
]
