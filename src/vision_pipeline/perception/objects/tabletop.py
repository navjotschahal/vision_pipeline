"""Detector-free box selection for a single object on a known support plane.

The bimanual box-grasp experiment observes exactly one box on a table, so an image
detector is unnecessary evidence. This module selects the object geometrically:

1. crop the cloud to an explicit workspace box,
2. fit the dominant support plane with RANSAC and drop its inliers,
3. cluster what remains on a grid in the plane's own basis,
4. keep the largest cluster and reuse the existing PCA oriented-box estimator.

Every stage reports what it measured, so a caller can publish the numbers rather than
assert that the pipeline worked. Nothing here invents geometry for unobserved surfaces,
and the source measurement's capture timestamp is carried through untouched.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import torch

from vision_pipeline.contracts import FrameId
from vision_pipeline.geometry.pointcloud import PointCloud
from vision_pipeline.geometry.spatial import (
    OrientedBox3D,
    QuaternionXyzw,
    RigidTransform3D,
    Vector3,
)

from .contracts import ObjectObservation3D, SelectionKind
from .geometry import estimate_observed_oriented_box

_METHOD = "workspace-crop+ransac-plane+grid-cluster+pca-observed-oriented-box"


class TabletopSegmentationError(ValueError):
    """The scene did not yield a usable single-object cluster.

    Callers should mark the track ``LOST`` instead of extrapolating a stale pose.
    """


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
class WorkspaceBounds:
    """An axis-aligned region of interest expressed in one named frame.

    The bounds are a statement about where the experiment allows an object to be, so
    they carry their frame and are checked against the cloud rather than assumed.
    """

    reference_frame: FrameId
    minimum_metres: Vector3
    maximum_metres: Vector3

    def __post_init__(self) -> None:
        if not isinstance(self.reference_frame, FrameId):
            raise TypeError("reference_frame must be a FrameId")
        for value, name in (
            (self.minimum_metres, "minimum_metres"),
            (self.maximum_metres, "maximum_metres"),
        ):
            if not isinstance(value, tuple) or len(value) != 3:
                raise TypeError(f"{name} must be a 3-element tuple")
            if any(isinstance(item, bool) or not isinstance(item, int | float) for item in value):
                raise TypeError(f"{name} elements must be numbers")
            if not all(math.isfinite(item) for item in value):
                raise ValueError(f"{name} elements must be finite")
        if any(
            low >= high
            for low, high in zip(self.minimum_metres, self.maximum_metres, strict=True)
        ):
            raise ValueError("workspace minimums must be smaller than maximums")

    def contains(self, xyz: torch.Tensor) -> torch.Tensor:
        """Return the boolean mask of ``[N, 3]`` points inside these bounds."""

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have [N, 3] shape")
        low = torch.tensor(self.minimum_metres, dtype=xyz.dtype, device=xyz.device)
        high = torch.tensor(self.maximum_metres, dtype=xyz.dtype, device=xyz.device)
        return torch.isfinite(xyz).all(dim=1) & (xyz >= low).all(dim=1) & (xyz <= high).all(dim=1)


@dataclass(frozen=True, slots=True)
class SupportPlane:
    """A fitted plane satisfying ``normal . point + offset = 0``."""

    normal: Vector3
    offset_metres: float
    inlier_count: int
    rms_residual_metres: float

    def __post_init__(self) -> None:
        if not isinstance(self.normal, tuple) or len(self.normal) != 3:
            raise TypeError("normal must be a 3-element tuple")
        if not all(math.isfinite(value) for value in self.normal):
            raise ValueError("normal elements must be finite")
        norm = math.sqrt(sum(value * value for value in self.normal))
        if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
            raise ValueError("normal must be a unit vector")
        if not math.isfinite(self.offset_metres):
            raise ValueError("offset_metres must be finite")
        _require_positive_int(self.inlier_count, "inlier_count")
        if not math.isfinite(self.rms_residual_metres) or self.rms_residual_metres < 0:
            raise ValueError("rms_residual_metres must be non-negative and finite")

    def signed_distance(self, xyz: torch.Tensor) -> torch.Tensor:
        """Signed metres from the plane, positive along ``normal``."""

        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have [N, 3] shape")
        normal = torch.tensor(self.normal, dtype=xyz.dtype, device=xyz.device)
        return xyz @ normal + self.offset_metres


@dataclass(frozen=True, slots=True)
class TabletopSegmentation:
    """One box observation plus the measured evidence each stage produced."""

    observation: ObjectObservation3D
    plane: SupportPlane
    workspace_point_count: int
    above_plane_point_count: int
    cluster_point_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.observation, ObjectObservation3D):
            raise TypeError("observation must be an ObjectObservation3D")
        if not isinstance(self.plane, SupportPlane):
            raise TypeError("plane must be a SupportPlane")
        for name in (
            "workspace_point_count",
            "above_plane_point_count",
            "cluster_point_count",
        ):
            _require_positive_int(getattr(self, name), name)


def _rotation_matrix(quaternion: QuaternionXyzw) -> list[list[float]]:
    x, y, z, w = quaternion
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def transform_point_cloud(
    cloud: PointCloud[torch.Tensor],
    transform: RigidTransform3D,
) -> PointCloud[torch.Tensor]:
    """Re-express a cloud in ``transform.target_frame`` without touching its timestamps.

    This is the vectorized counterpart of :meth:`RigidTransform3D.apply_point`, which is
    scalar and too slow for a full frame. The returned header keeps ``captured_at`` and
    only its ``frame_id`` changes, because the measurement is the same measurement.
    """

    if not isinstance(transform, RigidTransform3D):
        raise TypeError("transform must be a RigidTransform3D")
    if cloud.source.frame_id != transform.source_frame:
        raise ValueError(
            f"cloud frame {cloud.source.frame_id.value!r} does not match transform "
            f"source frame {transform.source_frame.value!r}"
        )
    xyz = cloud.xyz
    rotation = torch.tensor(
        _rotation_matrix(transform.rotation_xyzw),
        dtype=xyz.dtype,
        device=xyz.device,
    )
    translation = torch.tensor(
        transform.translation_metres,
        dtype=xyz.dtype,
        device=xyz.device,
    )
    return dataclasses.replace(
        cloud,
        source=dataclasses.replace(cloud.source, frame_id=transform.target_frame),
        xyz=xyz @ rotation.T + translation,
    )


def _select(cloud: PointCloud[torch.Tensor], mask: torch.Tensor) -> PointCloud[torch.Tensor]:
    colors = cloud.colors_rgb
    return dataclasses.replace(
        cloud,
        xyz=cloud.xyz[mask],
        colors_rgb=None if colors is None else colors[mask],
    )


def crop_to_workspace(
    cloud: PointCloud[torch.Tensor],
    bounds: WorkspaceBounds,
) -> PointCloud[torch.Tensor]:
    """Keep only the points inside an explicit region of interest."""

    if not isinstance(bounds, WorkspaceBounds):
        raise TypeError("bounds must be WorkspaceBounds")
    if cloud.source.frame_id != bounds.reference_frame:
        raise ValueError(
            f"cloud frame {cloud.source.frame_id.value!r} does not match workspace frame "
            f"{bounds.reference_frame.value!r}"
        )
    return _select(cloud, bounds.contains(cloud.xyz))


def _refine_plane(points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Total-least-squares plane through ``points``; returns (unit normal, offset)."""

    centroid = points.mean(dim=0)
    centered = points - centroid
    covariance = centered.T @ centered / max(1, points.shape[0] - 1)
    _, eigenvectors = torch.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    normal = normal / torch.linalg.norm(normal)
    return normal, -(normal @ centroid)


def fit_support_plane(
    xyz: torch.Tensor,
    *,
    distance_threshold_metres: float = 0.01,
    iterations: int = 200,
    minimum_inliers: int = 100,
    maximum_scoring_points: int = 20_000,
    seed: int = 0,
) -> SupportPlane:
    """Fit the dominant plane by RANSAC, then refine on its inliers.

    Hypotheses are scored against a bounded random subsample so the cost does not grow
    with image resolution; the refit and the reported inlier count use every point. The
    sampling is seeded, so the same cloud always yields the same plane.
    """

    _require_positive_finite(distance_threshold_metres, "distance_threshold_metres")
    _require_positive_int(iterations, "iterations")
    _require_positive_int(minimum_inliers, "minimum_inliers")
    _require_positive_int(maximum_scoring_points, "maximum_scoring_points")
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have [N, 3] shape")
    points = xyz[torch.isfinite(xyz).all(dim=1)].to(dtype=torch.float64)
    total = int(points.shape[0])
    if total < minimum_inliers or total < 3:
        raise TabletopSegmentationError(
            f"plane fitting needs at least {max(3, minimum_inliers)} finite points; got {total}"
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    scoring = points
    if total > maximum_scoring_points:
        chosen = torch.randperm(total, generator=generator)[:maximum_scoring_points]
        scoring = points[chosen]

    best_normal: torch.Tensor | None = None
    best_offset = 0.0
    best_score = -1
    for _ in range(iterations):
        sample = torch.randperm(total, generator=generator)[:3]
        a, b, c = points[sample[0]], points[sample[1]], points[sample[2]]
        normal = torch.linalg.cross(b - a, c - a)
        magnitude = float(torch.linalg.norm(normal))
        if magnitude < 1e-9:
            continue
        normal = normal / magnitude
        offset = float(-(normal @ a))
        score = int(
            (torch.abs(scoring @ normal + offset) <= distance_threshold_metres).sum()
        )
        if score > best_score:
            best_score, best_normal, best_offset = score, normal, offset
    if best_normal is None:
        raise TabletopSegmentationError("no non-degenerate plane hypothesis was found")

    inliers = torch.abs(points @ best_normal + best_offset) <= distance_threshold_metres
    if int(inliers.sum()) < minimum_inliers:
        raise TabletopSegmentationError(
            f"best plane had {int(inliers.sum())} inliers; at least {minimum_inliers} required"
        )
    plane_normal, plane_offset = _refine_plane(points[inliers])
    residual = points @ plane_normal + plane_offset
    refined = torch.abs(residual) <= distance_threshold_metres
    count = int(refined.sum())
    if count < minimum_inliers:
        raise TabletopSegmentationError(
            f"refined plane had {count} inliers; at least {minimum_inliers} required"
        )
    rms = float(torch.sqrt((residual[refined] ** 2).mean()))
    return SupportPlane(
        normal=(float(plane_normal[0]), float(plane_normal[1]), float(plane_normal[2])),
        offset_metres=float(plane_offset),
        inlier_count=count,
        rms_residual_metres=rms,
    )


def points_above_plane(
    cloud: PointCloud[torch.Tensor],
    plane: SupportPlane,
    *,
    clearance_metres: float = 0.01,
) -> tuple[PointCloud[torch.Tensor], SupportPlane]:
    """Drop the support surface, keeping only what stands off it.

    A fitted normal has an arbitrary sign, so the side holding more off-plane points is
    taken to be the object side and the returned plane is flipped to point that way.
    """

    _require_positive_finite(clearance_metres, "clearance_metres")
    distance = plane.signed_distance(cloud.xyz)
    positive = int((distance > clearance_metres).sum())
    negative = int((distance < -clearance_metres).sum())
    oriented = plane
    if negative > positive:
        oriented = dataclasses.replace(
            plane,
            normal=(-plane.normal[0], -plane.normal[1], -plane.normal[2]),
            offset_metres=-plane.offset_metres,
        )
        distance = -distance
    return _select(cloud, distance > clearance_metres), oriented


def _plane_basis(normal: Vector3) -> tuple[torch.Tensor, torch.Tensor]:
    axis = torch.tensor(normal, dtype=torch.float64)
    seed = torch.tensor(
        (1.0, 0.0, 0.0) if abs(axis[0]) < 0.9 else (0.0, 1.0, 0.0),
        dtype=torch.float64,
    )
    first = torch.linalg.cross(axis, seed)
    first = first / torch.linalg.norm(first)
    second = torch.linalg.cross(axis, first)
    return first, second / torch.linalg.norm(second)


def largest_planar_cluster_mask(
    cloud: PointCloud[torch.Tensor],
    plane: SupportPlane,
    *,
    cell_size_metres: float = 0.02,
    minimum_cluster_points: int = 100,
) -> torch.Tensor:
    """Boolean mask, aligned to ``cloud``, selecting its biggest connected group.

    Objects resting on a table separate cleanly in the plane's own 2D basis, so the
    points are binned onto a grid there and grouped with OpenCV's connected-component
    labelling rather than a hand-written clustering loop. OpenCV is imported lazily, the
    way the RealSense SDK is, so importing this package never requires it.

    ``cell_size_metres`` must comfortably exceed the cloud's point spacing at the working
    distance, or the occupancy grid develops holes and splits one object into several
    clusters. Spacing is about ``depth / fx``: roughly 1.7 mm at 1 m for a 640x480 stream
    with fx 600, so the 2 cm default holds on the order of a hundred points per cell.
    Raise the sampling stride and this value together, never one alone.
    """

    _require_positive_finite(cell_size_metres, "cell_size_metres")
    _require_positive_int(minimum_cluster_points, "minimum_cluster_points")
    try:
        import cv2
    except ModuleNotFoundError as error:  # pragma: no cover - exercised by environment
        raise TabletopSegmentationError(
            "opencv-python is required to cluster tabletop points; install the optional "
            "webcam dependencies for this host"
        ) from error

    xyz = cloud.xyz.to(dtype=torch.float64)
    if int(xyz.shape[0]) < minimum_cluster_points:
        raise TabletopSegmentationError(
            f"{int(xyz.shape[0])} points remain above the plane; at least "
            f"{minimum_cluster_points} required"
        )
    first, second = _plane_basis(plane.normal)
    projected = torch.stack((xyz @ first, xyz @ second), dim=1)
    origin = projected.amin(dim=0)
    cells = torch.floor((projected - origin) / cell_size_metres).to(dtype=torch.int64)
    height = int(cells[:, 1].max()) + 1
    width = int(cells[:, 0].max()) + 1
    occupancy = torch.zeros((height, width), dtype=torch.uint8)
    occupancy[cells[:, 1], cells[:, 0]] = 1

    count, labels = cv2.connectedComponents(occupancy.numpy(), connectivity=8)
    if count <= 1:
        raise TabletopSegmentationError("no occupied cells remained above the support plane")
    label_image = torch.from_numpy(labels)
    per_point = label_image[cells[:, 1], cells[:, 0]]
    populations = torch.bincount(per_point, minlength=count)
    populations[0] = 0
    best = int(torch.argmax(populations))
    size = int(populations[best])
    if size < minimum_cluster_points:
        raise TabletopSegmentationError(
            f"largest cluster held {size} points; at least {minimum_cluster_points} required"
        )
    return per_point == best


def largest_planar_cluster(
    cloud: PointCloud[torch.Tensor],
    plane: SupportPlane,
    *,
    cell_size_metres: float = 0.02,
    minimum_cluster_points: int = 100,
) -> PointCloud[torch.Tensor]:
    """Keep the biggest connected group of points seen from above the plane."""

    mask = largest_planar_cluster_mask(
        cloud,
        plane,
        cell_size_metres=cell_size_metres,
        minimum_cluster_points=minimum_cluster_points,
    )
    return _select(cloud, mask)


def estimate_tabletop_box_with_points(
    cloud: PointCloud[torch.Tensor],
    bounds: WorkspaceBounds,
    *,
    label: str = "box",
    confidence: float = 1.0,
    plane_distance_threshold_metres: float = 0.01,
    clearance_metres: float = 0.01,
    cell_size_metres: float = 0.02,
    minimum_cluster_points: int = 100,
    minimum_plane_inliers: int = 100,
    seed: int = 0,
) -> tuple[TabletopSegmentation, PointCloud[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Like :func:`estimate_tabletop_box`, but also returns the point-level evidence.

    A viewer that must show the table plane distinguished from the object (rather than
    just the final box) needs the cropped workspace cloud plus, in its point order, the
    boolean mask of points above the plane and the boolean mask of the selected cluster.
    ``estimate_tabletop_box`` stays a thin wrapper over this so there is exactly one
    place that runs the plane fit and the clustering.
    """

    if not isinstance(cloud, PointCloud):
        raise TypeError("cloud must be a PointCloud")
    cropped = crop_to_workspace(cloud, bounds)
    workspace_points = int(cropped.xyz.shape[0])
    if workspace_points < minimum_cluster_points:
        raise TabletopSegmentationError(
            f"{workspace_points} points fell inside the workspace; at least "
            f"{minimum_cluster_points} required"
        )
    plane = fit_support_plane(
        cropped.xyz,
        distance_threshold_metres=plane_distance_threshold_metres,
        minimum_inliers=minimum_plane_inliers,
        seed=seed,
    )
    above, oriented_plane = points_above_plane(
        cropped,
        plane,
        clearance_metres=clearance_metres,
    )
    above_points = int(above.xyz.shape[0])
    above_mask = oriented_plane.signed_distance(cropped.xyz) > clearance_metres
    cluster_mask_in_above = largest_planar_cluster_mask(
        above,
        oriented_plane,
        cell_size_metres=cell_size_metres,
        minimum_cluster_points=minimum_cluster_points,
    )
    cluster = _select(above, cluster_mask_in_above)
    bounds_3d: OrientedBox3D = estimate_observed_oriented_box(
        cluster,
        minimum_points=minimum_cluster_points,
    )
    cluster_points = int(cluster.xyz.shape[0])
    cluster_mask = torch.zeros(workspace_points, dtype=torch.bool)
    cluster_mask[above_mask.nonzero(as_tuple=True)[0][cluster_mask_in_above]] = True
    segmentation = TabletopSegmentation(
        observation=ObjectObservation3D(
            source=cluster.source,
            label=label,
            confidence=confidence,
            bounds=bounds_3d,
            selection_kind=SelectionKind.FULL_CLOUD,
            geometry_kind=cluster.geometry_kind,
            point_count=cluster_points,
            method=_METHOD,
            image_box=None,
        ),
        plane=oriented_plane,
        workspace_point_count=workspace_points,
        above_plane_point_count=above_points,
        cluster_point_count=cluster_points,
    )
    return segmentation, cropped, above_mask, cluster_mask


def estimate_tabletop_box(
    cloud: PointCloud[torch.Tensor],
    bounds: WorkspaceBounds,
    *,
    label: str = "box",
    confidence: float = 1.0,
    plane_distance_threshold_metres: float = 0.01,
    clearance_metres: float = 0.01,
    cell_size_metres: float = 0.02,
    minimum_cluster_points: int = 100,
    minimum_plane_inliers: int = 100,
    seed: int = 0,
) -> TabletopSegmentation:
    """Estimate one box's oriented bounds without any image detector.

    The returned observation is expressed in the cloud's own frame and keeps the source
    header, so the capture timestamp survives into whatever the caller publishes.
    """

    segmentation, _, _, _ = estimate_tabletop_box_with_points(
        cloud,
        bounds,
        label=label,
        confidence=confidence,
        plane_distance_threshold_metres=plane_distance_threshold_metres,
        clearance_metres=clearance_metres,
        cell_size_metres=cell_size_metres,
        minimum_cluster_points=minimum_cluster_points,
        minimum_plane_inliers=minimum_plane_inliers,
        seed=seed,
    )
    return segmentation


__all__ = [
    "SupportPlane",
    "TabletopSegmentation",
    "TabletopSegmentationError",
    "WorkspaceBounds",
    "crop_to_workspace",
    "estimate_tabletop_box",
    "estimate_tabletop_box_with_points",
    "fit_support_plane",
    "largest_planar_cluster",
    "largest_planar_cluster_mask",
    "points_above_plane",
    "transform_point_cloud",
]
