"""Object-cloud extraction and initial observed-cuboid estimation."""

from __future__ import annotations

import math

import numpy as np
import torch
from numpy.typing import NDArray

from vision_pipeline.contracts import ComputePlacement, MemoryPlacement, SampleHeader
from vision_pipeline.geometry import GeometryKind, PinholeIntrinsics, PointCloud
from vision_pipeline.geometry.pointcloud import depth_to_point_cloud
from vision_pipeline.geometry.spatial import OrientedBox3D, Pose3D, QuaternionXyzw
from vision_pipeline.image import ImageFrame
from vision_pipeline.perception.contracts import BoundingBox2D

from .contracts import ObjectObservation3D, SelectionKind


def bounding_box_mask(
    box: BoundingBox2D,
    *,
    width: int,
    height: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Rasterize a clipped half-open detector box into a boolean image mask."""

    if width <= 0 or height <= 0:
        raise ValueError("mask width and height must be positive")
    x_min = max(0, min(width, math.floor(box.x_min)))
    y_min = max(0, min(height, math.floor(box.y_min)))
    x_max = max(0, min(width, math.ceil(box.x_max)))
    y_max = max(0, min(height, math.ceil(box.y_max)))
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bounding box does not overlap the image")
    mask = torch.zeros((height, width), dtype=torch.bool, device=device)
    mask[y_min:y_max, x_min:x_max] = True
    return mask


def extract_object_point_cloud(
    depth_metres: torch.Tensor,
    color: ImageFrame,
    source: SampleHeader,
    intrinsics: PinholeIntrinsics,
    *,
    segmentation_mask: torch.Tensor | None = None,
    image_box: BoundingBox2D | None = None,
    min_depth_metres: float,
    max_depth_metres: float,
    sampling_stride: int,
    geometry_kind: GeometryKind,
    produced_on: ComputePlacement,
    memory: MemoryPlacement,
) -> tuple[PointCloud[torch.Tensor], SelectionKind]:
    """Deproject only one selected object's pixels.

    A segmentation mask is preferred. A detector box is an explicit coarse fallback;
    callers must not mistake its background-contaminated cloud for a segmented object.
    """

    if (segmentation_mask is None) == (image_box is None):
        raise ValueError("provide exactly one of segmentation_mask or image_box")
    if segmentation_mask is not None:
        selection = segmentation_mask
        selection_kind = SelectionKind.SEGMENTATION_MASK
    else:
        assert image_box is not None
        selection = bounding_box_mask(
            image_box,
            width=color.width,
            height=color.height,
            device=depth_metres.device,
        )
        selection_kind = SelectionKind.BOUNDING_BOX
    cloud = depth_to_point_cloud(
        depth_metres,
        color,
        source,
        intrinsics,
        min_depth_metres=min_depth_metres,
        max_depth_metres=max_depth_metres,
        sampling_stride=sampling_stride,
        geometry_kind=geometry_kind,
        produced_on=produced_on,
        memory=memory,
        selection_mask=selection,
    )
    if cloud.xyz.shape[0] == 0:
        raise ValueError("object selection contains no valid depth points")
    return cloud, selection_kind


def _canonicalize_axes(axes: torch.Tensor) -> torch.Tensor:
    """Choose deterministic PCA signs and enforce a right-handed rotation."""

    result = axes.clone()
    for column in range(3):
        axis = result[:, column]
        dominant = int(torch.argmax(torch.abs(axis)).item())
        if axis[dominant].item() < 0:
            result[:, column] = -axis
    if torch.linalg.det(result).item() < 0:
        result[:, 2] = -result[:, 2]
    return result


def _matrix_to_quaternion_xyzw(matrix: NDArray[np.float64]) -> QuaternionXyzw:
    """Convert one proper 3x3 rotation matrix to a normalized quaternion."""

    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        diagonal = np.diag(matrix)
        largest = int(np.argmax(diagonal))
        if largest == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif largest == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
    quaternion = np.asarray((x, y, z, w), dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    if quaternion[3] < 0:
        quaternion = -quaternion
    return tuple(float(value) for value in quaternion)  # type: ignore[return-value]


def estimate_observed_oriented_box(
    cloud: PointCloud[torch.Tensor],
    *,
    minimum_points: int = 20,
    minimum_extent_metres: float = 1e-4,
) -> OrientedBox3D:
    """Fit a PCA oriented box around the currently visible object points.

    This is an observed-surface bound, not a claim about unobserved geometry. PCA axes
    are ambiguous for symmetric objects and must later be stabilized by a 3D tracker.
    """

    if minimum_points < 3:
        raise ValueError("minimum_points must be at least three")
    if not math.isfinite(minimum_extent_metres) or minimum_extent_metres <= 0:
        raise ValueError("minimum_extent_metres must be positive and finite")
    xyz = cloud.xyz
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("point cloud XYZ values must have [N, 3] shape")
    xyz = xyz[torch.isfinite(xyz).all(dim=1)].to(dtype=torch.float64)
    if xyz.shape[0] < minimum_points:
        raise ValueError(
            f"object cloud has {xyz.shape[0]} finite points; at least {minimum_points} required"
        )

    mean = xyz.mean(dim=0)
    centered = xyz - mean
    covariance = centered.T @ centered / (xyz.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    axes = _canonicalize_axes(eigenvectors[:, order])
    local = centered @ axes
    minimum = local.amin(dim=0)
    maximum = local.amax(dim=0)
    size = maximum - minimum
    if torch.any(size <= minimum_extent_metres).item():
        raise ValueError("object cloud is degenerate and cannot define a 3D cuboid")
    local_center = (minimum + maximum) / 2
    world_center = mean + axes @ local_center

    rotation = axes.detach().to(device="cpu", dtype=torch.float64).numpy()
    quaternion = _matrix_to_quaternion_xyzw(rotation)
    position = tuple(float(value) for value in world_center.tolist())
    extents = tuple(float(value) for value in size.tolist())
    return OrientedBox3D(
        pose=Pose3D(
            reference_frame=cloud.source.frame_id,
            position_metres=position,  # type: ignore[arg-type]
            orientation_xyzw=quaternion,
        ),
        size_metres=extents,  # type: ignore[arg-type]
    )


def estimate_object_observation(
    cloud: PointCloud[torch.Tensor],
    *,
    label: str,
    confidence: float,
    selection_kind: SelectionKind,
    image_box: BoundingBox2D | None = None,
    minimum_points: int = 20,
) -> ObjectObservation3D:
    """Build the compact 3D result that downstream tracking will consume."""

    bounds = estimate_observed_oriented_box(cloud, minimum_points=minimum_points)
    return ObjectObservation3D(
        source=cloud.source,
        label=label,
        confidence=confidence,
        bounds=bounds,
        selection_kind=selection_kind,
        geometry_kind=cloud.geometry_kind,
        point_count=int(cloud.xyz.shape[0]),
        method="pca-observed-oriented-box",
        image_box=image_box,
    )


__all__ = [
    "bounding_box_mask",
    "estimate_object_observation",
    "estimate_observed_oriented_box",
    "extract_object_point_cloud",
]
