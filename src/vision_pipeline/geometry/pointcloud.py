"""Typed point clouds and vectorized depth-image deprojection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Generic, TypeVar

import numpy as np
import torch

from vision_pipeline.contracts import ComputePlacement, MemoryPlacement, SampleHeader
from vision_pipeline.image import ImageFrame, PixelFormat

from .camera import PinholeIntrinsics

TensorT = TypeVar("TensorT")


class GeometryKind(StrEnum):
    """Whether scale and geometry came from measurement or model inference."""

    MEASURED_METRIC = "measured_metric"
    ESTIMATED_METRIC = "estimated_metric"
    UNSCALED = "unscaled"


@dataclass(frozen=True, slots=True)
class PointCloud(Generic[TensorT]):
    """Unorganized XYZ point cloud tied to its source image and camera frame."""

    source: SampleHeader
    xyz: TensorT
    colors_rgb: TensorT | None
    intrinsics: PinholeIntrinsics
    geometry_kind: GeometryKind
    sampling_stride: int
    produced_on: ComputePlacement
    memory: MemoryPlacement

    def __post_init__(self) -> None:
        if not isinstance(self.source, SampleHeader):
            raise TypeError("source must be a SampleHeader")
        if not isinstance(self.intrinsics, PinholeIntrinsics):
            raise TypeError("intrinsics must be PinholeIntrinsics")
        if not isinstance(self.geometry_kind, GeometryKind):
            raise TypeError("geometry_kind must be GeometryKind")
        if self.sampling_stride <= 0:
            raise ValueError("sampling_stride must be positive")


def _depth_plane(depth_metres: torch.Tensor) -> torch.Tensor:
    if depth_metres.ndim == 4 and depth_metres.shape[:2] == (1, 1):
        return depth_metres[0, 0]
    if depth_metres.ndim == 2:
        return depth_metres
    raise ValueError("depth must have [H, W] or [1, 1, H, W] shape")


def depth_to_point_cloud(
    depth_metres: torch.Tensor,
    color: ImageFrame,
    source: SampleHeader,
    intrinsics: PinholeIntrinsics,
    *,
    min_depth_metres: float,
    max_depth_metres: float,
    sampling_stride: int,
    geometry_kind: GeometryKind,
    produced_on: ComputePlacement,
    memory: MemoryPlacement,
    selection_mask: torch.Tensor | None = None,
) -> PointCloud[torch.Tensor]:
    """Deproject valid sampled depth pixels without leaving the tensor device.

    ``selection_mask`` optionally limits deprojection to selected source-image pixels,
    allowing segmentation to remove background geometry before creating a cloud.
    """

    if sampling_stride <= 0:
        raise ValueError("sampling_stride must be positive")
    if (
        not math.isfinite(min_depth_metres)
        or not math.isfinite(max_depth_metres)
        or min_depth_metres < 0
        or max_depth_metres <= min_depth_metres
    ):
        raise ValueError("depth limits must be finite, non-negative, and increasing")
    depth = _depth_plane(depth_metres)
    height, width = int(depth.shape[0]), int(depth.shape[1])
    if (width, height) != (color.width, color.height):
        raise ValueError("depth and color image dimensions differ")
    if (width, height) != (intrinsics.width, intrinsics.height):
        raise ValueError("intrinsics resolution does not match the depth image")

    selected: torch.Tensor | None = None
    if selection_mask is not None:
        selected = _depth_plane(selection_mask)
        if tuple(selected.shape) != (height, width):
            raise ValueError("selection mask and depth image dimensions differ")
        selected = selected.to(device=depth.device, dtype=torch.bool)

    rows = torch.arange(0, height, sampling_stride, device=depth.device)
    columns = torch.arange(0, width, sampling_stride, device=depth.device)
    pixel_y, pixel_x = torch.meshgrid(rows, columns, indexing="ij")
    sampled_depth = depth[::sampling_stride, ::sampling_stride].to(dtype=torch.float32)
    valid = (
        torch.isfinite(sampled_depth)
        & (sampled_depth >= min_depth_metres)
        & (sampled_depth <= max_depth_metres)
    )
    if selected is not None:
        valid &= selected[::sampling_stride, ::sampling_stride]
    z = sampled_depth[valid]
    x = (pixel_x[valid].to(dtype=torch.float32) - intrinsics.cx) * z / intrinsics.fx
    y = (pixel_y[valid].to(dtype=torch.float32) - intrinsics.cy) * z / intrinsics.fy
    xyz = torch.stack((x, y, z), dim=1)

    rgb = color.data if color.pixel_format is PixelFormat.RGB8 else color.data[..., ::-1]
    sampled_rgb = np.ascontiguousarray(rgb[::sampling_stride, ::sampling_stride])
    colors = torch.from_numpy(sampled_rgb).to(device=depth.device)[valid]
    return PointCloud(
        source=source,
        xyz=xyz,
        colors_rgb=colors,
        intrinsics=intrinsics,
        geometry_kind=geometry_kind,
        sampling_stride=sampling_stride,
        produced_on=produced_on,
        memory=memory,
    )


__all__ = ["GeometryKind", "PointCloud", "depth_to_point_cloud"]
