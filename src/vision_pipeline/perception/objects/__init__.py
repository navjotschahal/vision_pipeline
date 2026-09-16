"""Model-independent 3D object perception building blocks."""

from .box_estimator import (
    BoxEstimator,
    TabletopBoxEstimator,
    TabletopBoxEstimatorConfig,
    TabletopDiagnostics,
)
from .contracts import (
    GraspCandidate3D,
    ObjectObservation3D,
    ObjectTrack3D,
    PoseCovariance,
    SelectionKind,
    TrackStatus,
)
from .geometry import (
    bounding_box_mask,
    estimate_object_observation,
    estimate_observed_oriented_box,
    extract_object_point_cloud,
)
from .tabletop import (
    SupportPlane,
    TabletopSegmentation,
    TabletopSegmentationError,
    WorkspaceBounds,
    crop_to_workspace,
    estimate_tabletop_box,
    estimate_tabletop_box_with_points,
    fit_support_plane,
    largest_planar_cluster,
    largest_planar_cluster_mask,
    points_above_plane,
    transform_point_cloud,
)

__all__ = [
    "BoxEstimator",
    "GraspCandidate3D",
    "ObjectObservation3D",
    "ObjectTrack3D",
    "PoseCovariance",
    "SelectionKind",
    "SupportPlane",
    "TabletopBoxEstimator",
    "TabletopBoxEstimatorConfig",
    "TabletopDiagnostics",
    "TabletopSegmentation",
    "TabletopSegmentationError",
    "TrackStatus",
    "WorkspaceBounds",
    "bounding_box_mask",
    "crop_to_workspace",
    "estimate_object_observation",
    "estimate_observed_oriented_box",
    "estimate_tabletop_box",
    "estimate_tabletop_box_with_points",
    "extract_object_point_cloud",
    "fit_support_plane",
    "largest_planar_cluster",
    "largest_planar_cluster_mask",
    "points_above_plane",
    "transform_point_cloud",
]
