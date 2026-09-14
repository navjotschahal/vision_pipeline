"""Shared configuration and errors for interchangeable YOLO implementations."""

from __future__ import annotations

from dataclasses import dataclass


class YoloError(RuntimeError):
    """A YOLO implementation could not load or run a detection model."""


@dataclass(frozen=True, slots=True)
class YoloConfig:
    model: str = "yolo26n.pt"
    device: str = "cpu"
    confidence: float = 0.25
    iou: float = 0.7
    image_size: int = 640
    max_detections: int = 300

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between zero and one")
        if not 0 <= self.iou <= 1:
            raise ValueError("iou must be between zero and one")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")
