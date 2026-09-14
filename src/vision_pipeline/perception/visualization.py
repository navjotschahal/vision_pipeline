"""OpenCV visualization kept separate from perception results."""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.image import ImageFrame
from vision_pipeline.perception.contracts import DetectionBatch


def draw_detections(image: ImageFrame, batch: DetectionBatch) -> NDArray[np.uint8]:
    """Return a BGR copy annotated with typed detection results."""

    if (batch.image_width, batch.image_height) != (image.width, image.height):
        raise ValueError("detection result dimensions do not match the image")
    canvas = image.data.copy()
    for detection in batch.detections:
        box = detection.box
        left, top = int(round(box.x_min)), int(round(box.y_min))
        right, bottom = int(round(box.x_max)), int(round(box.y_max))
        color = _class_color(detection.class_id)
        cv2.rectangle(canvas, (left, top), (right, bottom), color, 2)
        text = f"{detection.label} {detection.confidence:.2f}"
        text_y = max(18, top - 6)
        cv2.putText(
            canvas,
            text,
            (left, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )
    return canvas


def _class_color(class_id: int) -> tuple[int, int, int]:
    return (
        64 + (class_id * 47) % 192,
        64 + (class_id * 89) % 192,
        64 + (class_id * 137) % 192,
    )
