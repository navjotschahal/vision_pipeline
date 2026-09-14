"""OpenCV visualizations for DINOv2 dense predictions and correspondences."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from vision_pipeline.image import ImageFrame

from .matching import ImagePoint
from .tasks import DepthPrediction, SemanticSegmentation


def render_depth(prediction: DepthPrediction[torch.Tensor]) -> NDArray[np.uint8]:
    depth = prediction.depth_metres[0, 0].detach().to(device="cpu", dtype=torch.float32).numpy()
    finite = depth[np.isfinite(depth)]
    if finite.size == 0:
        raise ValueError("depth prediction contains no finite values")
    near = float(np.percentile(finite, 2))
    far = float(np.percentile(finite, 98))
    scale = max(far - near, 1e-6)
    # Near geometry is warm and far geometry is cool.
    normalized = np.clip((far - depth) / scale, 0.0, 1.0)
    gray = cast(NDArray[np.uint8], (normalized * 255).astype(np.uint8))
    view = cast(NDArray[np.uint8], cv2.applyColorMap(gray, cv2.COLORMAP_TURBO))
    cv2.putText(
        view,
        f"metric depth {near:.2f}-{far:.2f} m (display percentiles)",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return view


def _semantic_palette(count: int) -> NDArray[np.uint8]:
    """Generate a deterministic, high-contrast Pascal-style palette."""

    palette = np.zeros((count, 3), dtype=np.uint8)
    for class_id in range(count):
        value = class_id
        bit = 0
        while value:
            palette[class_id, 2] |= ((value >> 0) & 1) << (7 - bit)
            palette[class_id, 1] |= ((value >> 1) & 1) << (7 - bit)
            palette[class_id, 0] |= ((value >> 2) & 1) << (7 - bit)
            value >>= 3
            bit += 1
    return palette


def render_semantic_segmentation(
    image: ImageFrame,
    prediction: SemanticSegmentation[torch.Tensor],
    *,
    alpha: float = 0.58,
) -> NDArray[np.uint8]:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    class_ids = prediction.class_ids[0].detach().to(device="cpu").numpy()
    if class_ids.shape != (image.height, image.width):
        raise ValueError("segmentation dimensions do not match the source image")
    palette = _semantic_palette(len(prediction.class_names))
    color_mask = palette[class_ids]
    source_bgr = image.data if image.pixel_format.value == "bgr8" else image.data[..., ::-1]
    blended = cv2.addWeighted(
        np.ascontiguousarray(source_bgr),
        1.0 - alpha,
        np.ascontiguousarray(color_mask),
        alpha,
        0.0,
    )

    ids, counts = np.unique(class_ids, return_counts=True)
    ordering = np.argsort(counts)[::-1][:5]
    labels = [prediction.class_names[int(ids[index])] for index in ordering]
    cv2.putText(
        blended,
        "ADE20K: " + ", ".join(labels),
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return cast(NDArray[np.uint8], blended)


def draw_point(
    image: NDArray[np.uint8],
    point: ImagePoint,
    *,
    color: tuple[int, int, int],
    label: str,
) -> NDArray[np.uint8]:
    view = image.copy()
    center = (round(point.x), round(point.y))
    cv2.circle(view, center, 12, color, 3, cv2.LINE_AA)
    cv2.drawMarker(view, center, color, cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
    cv2.putText(
        view,
        label,
        (max(8, center[0] + 14), max(24, center[1] - 14)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    return view


__all__ = ["draw_point", "render_depth", "render_semantic_segmentation"]
