"""The pose-estimation seam: today's geometric box finder behind a swappable contract.

The viewer, the shared-memory transport, and (in a later phase) the ROS node must
depend only on :class:`BoxEstimator` and the :class:`~.contracts.ObjectObservation3D`
it returns, never on how a concrete implementation got there. Today that implementation
is workspace-crop + RANSAC plane removal + grid clustering (`tabletop.py`); tomorrow it
could be a DNN. Swapping it should touch only this file and its config.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from vision_pipeline.contracts import ComputeKind, ComputePlacement, MemoryKind, MemoryPlacement
from vision_pipeline.geometry.pointcloud import GeometryKind, PointCloud, depth_to_point_cloud
from vision_pipeline.rgbd import AlignedRgbdFrame

from .contracts import ObjectObservation3D
from .tabletop import (
    TabletopSegmentation,
    TabletopSegmentationError,
    WorkspaceBounds,
    estimate_tabletop_box_with_points,
)


@runtime_checkable
class BoxEstimator(Protocol):
    """Produces one box pose from one aligned RGB-D frame, or ``None`` if not visible."""

    def estimate(self, frame: AlignedRgbdFrame) -> ObjectObservation3D | None: ...


def _require_positive_finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class TabletopBoxEstimatorConfig:
    """Tuning for the geometric estimator; distances are metres."""

    workspace: WorkspaceBounds
    label: str = "box"
    confidence: float = 1.0
    min_depth_metres: float = 0.1
    max_depth_metres: float = 2.0
    sampling_stride: int = 2
    plane_distance_threshold_metres: float = 0.01
    clearance_metres: float = 0.01
    cell_size_metres: float = 0.02
    minimum_cluster_points: int = 100
    minimum_plane_inliers: int = 100
    seed: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.workspace, WorkspaceBounds):
            raise TypeError("workspace must be a WorkspaceBounds")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("label must be a non-empty string")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be finite and between zero and one")
        _require_positive_finite(self.min_depth_metres, "min_depth_metres")
        _require_positive_finite(self.max_depth_metres, "max_depth_metres")
        if self.max_depth_metres <= self.min_depth_metres:
            raise ValueError("max_depth_metres must exceed min_depth_metres")
        if isinstance(self.sampling_stride, bool) or not isinstance(self.sampling_stride, int):
            raise TypeError("sampling_stride must be an integer")
        if self.sampling_stride <= 0:
            raise ValueError("sampling_stride must be positive")
        for name in (
            "plane_distance_threshold_metres",
            "clearance_metres",
            "cell_size_metres",
        ):
            _require_positive_finite(getattr(self, name), name)
        for name in ("minimum_cluster_points", "minimum_plane_inliers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class TabletopDiagnostics:
    """Point-level evidence behind one estimate, for a viewer only.

    Not part of the ``BoxEstimator`` contract: a future non-geometric estimator has no
    obligation to produce this, and consumers that only need the pose never look at it.

    ``full_cloud`` and ``workspace_mask`` are always populated once depth deprojection
    succeeds, even when segmentation fails -- seeing the whole scene, not just the
    workspace crop, is what makes a bad crop or a bad plane fit visible rather than
    inferred. ``above_plane_mask`` and ``cluster_mask`` are ``None`` on failure. All
    three masks are boolean and aligned to ``full_cloud``'s point order.
    """

    full_cloud: PointCloud[torch.Tensor]
    segmentation: TabletopSegmentation | None
    workspace_mask: torch.Tensor
    above_plane_mask: torch.Tensor | None
    cluster_mask: torch.Tensor | None


class TabletopBoxEstimator:
    """The geometric ``BoxEstimator``: workspace crop, RANSAC plane, grid clustering."""

    def __init__(self, config: TabletopBoxEstimatorConfig) -> None:
        if not isinstance(config, TabletopBoxEstimatorConfig):
            raise TypeError("config must be a TabletopBoxEstimatorConfig")
        self._config = config
        self._last_diagnostics: TabletopDiagnostics | None = None

    @property
    def last_diagnostics(self) -> TabletopDiagnostics | None:
        """Evidence behind the most recent successful :meth:`estimate`, if any."""

        return self._last_diagnostics

    def estimate(self, frame: AlignedRgbdFrame) -> ObjectObservation3D | None:
        if not isinstance(frame, AlignedRgbdFrame):
            raise TypeError("frame must be an AlignedRgbdFrame")
        config = self._config
        if frame.depth.header.frame_id != config.workspace.reference_frame:
            raise ValueError(
                f"frame {frame.depth.header.frame_id.value!r} does not match workspace "
                f"frame {config.workspace.reference_frame.value!r}"
            )
        depth = torch.from_numpy(frame.depth.payload.data)
        cloud = depth_to_point_cloud(
            depth,
            frame.color.payload,
            frame.depth.header,
            frame.aligned_depth_intrinsics,
            min_depth_metres=config.min_depth_metres,
            max_depth_metres=config.max_depth_metres,
            sampling_stride=config.sampling_stride,
            geometry_kind=GeometryKind.MEASURED_METRIC,
            produced_on=ComputePlacement(
                ComputeKind.HOST_CPU,
                f"host-process-{os.getpid()}/tabletop-box-estimator",
            ),
            memory=MemoryPlacement(MemoryKind.HOST),
        )
        workspace_mask = config.workspace.contains(cloud.xyz)
        try:
            segmentation, _workspace_cloud, above_mask_in_workspace, cluster_mask_in_workspace = (
                estimate_tabletop_box_with_points(
                    cloud,
                    config.workspace,
                    label=config.label,
                    confidence=config.confidence,
                    plane_distance_threshold_metres=config.plane_distance_threshold_metres,
                    clearance_metres=config.clearance_metres,
                    cell_size_metres=config.cell_size_metres,
                    minimum_cluster_points=config.minimum_cluster_points,
                    minimum_plane_inliers=config.minimum_plane_inliers,
                    seed=config.seed,
                )
            )
        except TabletopSegmentationError:
            self._last_diagnostics = TabletopDiagnostics(
                full_cloud=cloud,
                segmentation=None,
                workspace_mask=workspace_mask,
                above_plane_mask=None,
                cluster_mask=None,
            )
            return None
        above_mask = torch.zeros(cloud.xyz.shape[0], dtype=torch.bool)
        above_mask[workspace_mask] = above_mask_in_workspace
        cluster_mask = torch.zeros(cloud.xyz.shape[0], dtype=torch.bool)
        cluster_mask[workspace_mask] = cluster_mask_in_workspace
        self._last_diagnostics = TabletopDiagnostics(
            full_cloud=cloud,
            segmentation=segmentation,
            workspace_mask=workspace_mask,
            above_plane_mask=above_mask,
            cluster_mask=cluster_mask,
        )
        return segmentation.observation


__all__ = [
    "BoxEstimator",
    "TabletopBoxEstimator",
    "TabletopBoxEstimatorConfig",
    "TabletopDiagnostics",
]
