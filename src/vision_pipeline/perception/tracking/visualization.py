"""OpenCV visualization for persistent tracking identities."""

from __future__ import annotations

from typing import TypeVar

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.image import ImageFrame
from vision_pipeline.perception.tracking.contracts import TrackingBatch

TrackingTensorT = TypeVar("TrackingTensorT")


def draw_tracks(
    image: ImageFrame, batch: TrackingBatch[TrackingTensorT]
) -> NDArray[np.uint8]:
    """Return a BGR image annotated with class labels and stable track IDs."""

    detections = batch.regions.detections
    if (detections.image_width, detections.image_height) != (image.width, image.height):
        raise ValueError("tracking result dimensions do not match the image")
    canvas = image.data.copy()
    for tracked in batch.tracked_detections:
        detection = detections.detections[tracked.detection_index]
        box = detection.box
        left, top = int(round(box.x_min)), int(round(box.y_min))
        right, bottom = int(round(box.x_max)), int(round(box.y_max))
        color = _track_color(tracked.track_id)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        text = f"{detection.label} #{tracked.track_id} {detection.confidence:.2f}"
        cv2.putText(
            canvas,
            text,
            (left, max(18, top - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def _track_color(track_id: int) -> tuple[int, int, int]:
    return (
        64 + (track_id * 47) % 192,
        64 + (track_id * 89) % 192,
        64 + (track_id * 137) % 192,
    )


__all__ = ["draw_tracks"]
