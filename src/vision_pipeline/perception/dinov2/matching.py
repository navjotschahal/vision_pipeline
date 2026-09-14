"""DINOv2 patch correspondence without an object detector."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from vision_pipeline.perception.contracts import FeatureBatch, FeatureGeometry


@dataclass(frozen=True, slots=True)
class ImagePoint:
    x: float
    y: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.x) or not math.isfinite(self.y):
            raise ValueError("image point coordinates must be finite")
        if self.x < 0 or self.y < 0:
            raise ValueError("image point coordinates must be non-negative")


@dataclass(frozen=True, slots=True)
class SparsePatchMatch:
    reference_point: ImagePoint
    matched_point: ImagePoint
    reference_grid_xy: tuple[int, int]
    matched_grid_xy: tuple[int, int]
    cosine_similarity: float


def _source_to_grid(point: ImagePoint, geometry: FeatureGeometry) -> tuple[int, int]:
    source_x = min(point.x, math.nextafter(float(geometry.source_width), 0.0))
    source_y = min(point.y, math.nextafter(float(geometry.source_height), 0.0))
    model_x = source_x * geometry.scale_x + geometry.content_left
    model_y = source_y * geometry.scale_y + geometry.content_top
    grid_x = min(geometry.grid_width - 1, max(0, int(model_x // geometry.patch_width)))
    grid_y = min(geometry.grid_height - 1, max(0, int(model_y // geometry.patch_height)))
    return grid_x, grid_y


def _grid_to_source(grid_xy: tuple[int, int], geometry: FeatureGeometry) -> ImagePoint:
    grid_x, grid_y = grid_xy
    model_x = (grid_x + 0.5) * geometry.patch_width
    model_y = (grid_y + 0.5) * geometry.patch_height
    source_x = (model_x - geometry.content_left) / geometry.scale_x
    source_y = (model_y - geometry.content_top) / geometry.scale_y
    return ImagePoint(
        x=min(max(source_x, 0.0), math.nextafter(float(geometry.source_width), 0.0)),
        y=min(max(source_y, 0.0), math.nextafter(float(geometry.source_height), 0.0)),
    )


def match_reference_patch(
    reference: FeatureBatch[torch.Tensor],
    target: FeatureBatch[torch.Tensor],
    reference_point: ImagePoint,
) -> SparsePatchMatch:
    """Find the target patch with maximum cosine similarity to a reference patch."""

    if reference.model_id != target.model_id:
        raise ValueError("patch matching requires the same feature model")
    reference_features = reference.spatial_features
    target_features = target.spatial_features
    if reference_features.ndim != 4 or target_features.ndim != 4:
        raise ValueError("spatial features must have [B, C, H, W] shape")
    if reference_features.shape[0] != 1 or target_features.shape[0] != 1:
        raise ValueError("patch matching currently requires single-image feature batches")
    if reference_features.shape[1] != target_features.shape[1]:
        raise ValueError("reference and target embedding dimensions differ")
    if reference_features.device != target_features.device:
        raise ValueError("reference and target features must share a device")

    reference_grid = _source_to_grid(reference_point, reference.geometry)
    reference_descriptor = reference_features[
        0, :, reference_grid[1], reference_grid[0]
    ].to(dtype=torch.float32)
    reference_descriptor = functional.normalize(reference_descriptor, dim=0)
    target_descriptors = target_features[0].to(dtype=torch.float32).flatten(1).transpose(0, 1)
    target_descriptors = functional.normalize(target_descriptors, dim=1)
    similarities = target_descriptors @ reference_descriptor
    flat_index = int(torch.argmax(similarities).item())
    target_grid = (
        flat_index % target.geometry.grid_width,
        flat_index // target.geometry.grid_width,
    )
    snapped_reference = _grid_to_source(reference_grid, reference.geometry)
    matched_point = _grid_to_source(target_grid, target.geometry)
    return SparsePatchMatch(
        reference_point=snapped_reference,
        matched_point=matched_point,
        reference_grid_xy=reference_grid,
        matched_grid_xy=target_grid,
        cosine_similarity=float(similarities[flat_index].item()),
    )


__all__ = ["ImagePoint", "SparsePatchMatch", "match_reference_patch"]
