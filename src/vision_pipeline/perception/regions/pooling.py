"""GPU-preserving pooling of dense patch features inside detection boxes."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as functional

from vision_pipeline.perception.contracts import DetectionBatch, FeatureBatch
from vision_pipeline.perception.regions.contracts import (
    FeatureGridRegion,
    RegionDescriptorBatch,
)


def _grid_region(
    detection_index: int,
    detections: DetectionBatch,
    features: FeatureBatch[torch.Tensor],
) -> FeatureGridRegion:
    box = detections.detections[detection_index].box
    geometry = features.geometry

    model_x_min = box.x_min * geometry.scale_x + geometry.content_left
    model_y_min = box.y_min * geometry.scale_y + geometry.content_top
    model_x_max = box.x_max * geometry.scale_x + geometry.content_left
    model_y_max = box.y_max * geometry.scale_y + geometry.content_top

    x_min = max(0, min(math.floor(model_x_min / geometry.patch_width), geometry.grid_width - 1))
    y_min = max(
        0,
        min(math.floor(model_y_min / geometry.patch_height), geometry.grid_height - 1),
    )
    x_max = max(
        x_min + 1,
        min(math.ceil(model_x_max / geometry.patch_width), geometry.grid_width),
    )
    y_max = max(
        y_min + 1,
        min(math.ceil(model_y_max / geometry.patch_height), geometry.grid_height),
    )
    return FeatureGridRegion(detection_index, x_min, y_min, x_max, y_max)


def pool_detection_regions(
    detections: DetectionBatch,
    features: FeatureBatch[torch.Tensor],
) -> RegionDescriptorBatch[torch.Tensor]:
    """Mean-pool and L2-normalize one DINO descriptor per YOLO detection.

    The dense feature tensor is never copied to host memory. Detection coordinates are
    mapped through the recorded resize/padding transform into half-open patch bounds.
    """

    if detections.source != features.source:
        raise ValueError("detections and features must derive from the same source sample")
    geometry = features.geometry
    if (detections.image_width, detections.image_height) != (
        geometry.source_width,
        geometry.source_height,
    ):
        raise ValueError("detection and feature source dimensions do not match")

    spatial = features.spatial_features
    if spatial.ndim != 4 or spatial.shape[0] != 1:
        raise ValueError("spatial features must have single-image [1,C,H,W] shape")
    if (spatial.shape[3], spatial.shape[2]) != (
        geometry.grid_width,
        geometry.grid_height,
    ):
        raise ValueError("spatial feature tensor does not match declared grid geometry")

    regions = tuple(
        _grid_region(index, detections, features)
        for index in range(len(detections.detections))
    )
    pooled = [
        spatial[
            0,
            :,
            region.y_min : region.y_max,
            region.x_min : region.x_max,
        ].mean(dim=(1, 2))
        for region in regions
    ]
    descriptors = (
        functional.normalize(torch.stack(pooled), dim=1)
        if pooled
        else spatial.new_empty((0, spatial.shape[1]))
    )
    return RegionDescriptorBatch(
        detections=detections,
        features=features,
        descriptors=descriptors,
        regions=regions,
        pooling="mean+l2",
    )


__all__ = ["pool_detection_regions"]
