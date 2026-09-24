"""Optional, live COCO-17 pose backend for the shared human-pose contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from vision_pipeline.contracts import ComputeKind, ComputePlacement, SampleHeader
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception.pose.contracts import HumanJoint, HumanKeypoint2D, HumanPose2D

COCO_JOINTS = (
    HumanJoint.NOSE,
    HumanJoint.LEFT_EYE,
    HumanJoint.RIGHT_EYE,
    HumanJoint.LEFT_EAR,
    HumanJoint.RIGHT_EAR,
    HumanJoint.LEFT_SHOULDER,
    HumanJoint.RIGHT_SHOULDER,
    HumanJoint.LEFT_ELBOW,
    HumanJoint.RIGHT_ELBOW,
    HumanJoint.LEFT_WRIST,
    HumanJoint.RIGHT_WRIST,
    HumanJoint.LEFT_HIP,
    HumanJoint.RIGHT_HIP,
    HumanJoint.LEFT_KNEE,
    HumanJoint.RIGHT_KNEE,
    HumanJoint.LEFT_ANKLE,
    HumanJoint.RIGHT_ANKLE,
)


@dataclass(frozen=True, slots=True)
class YoloPoseConfig:
    model: str = "yolo11n-pose.pt"
    device: str | None = None
    minimum_confidence: float = 0.5
    person_index: int = 0

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if not math.isfinite(self.minimum_confidence) or not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be between zero and one")
        if self.person_index < 0:
            raise ValueError("person_index must be non-negative")


class YoloPoseEstimator:
    """Choose one explicitly indexed person; identity is not tracked across frames."""

    def __init__(self, config: YoloPoseConfig | None = None, *, model: Any = None) -> None:
        self.config = config or YoloPoseConfig()
        if model is None:
            from ultralytics import YOLO  # type: ignore[attr-defined]

            model = YOLO(self.config.model)
        self._model = model
        self.last_person_count = 0

    def estimate(self, image: ImageFrame, source_header: SampleHeader) -> HumanPose2D | None:
        pixels = image.data
        if image.pixel_format is PixelFormat.RGB8:
            pixels = np.ascontiguousarray(pixels[:, :, ::-1])
        options: dict[str, object] = {"verbose": False, "conf": self.config.minimum_confidence}
        if self.config.device is not None:
            options["device"] = self.config.device
        result = self._model.predict(pixels, **options)[0]
        if result.keypoints is None or result.keypoints.xy is None:
            self.last_person_count = 0
            return None
        xy = result.keypoints.xy.cpu().numpy()
        self.last_person_count = len(xy)
        if len(xy) <= self.config.person_index:
            return None
        scores = result.keypoints.conf
        confidences = (
            scores.cpu().numpy()[self.config.person_index]
            if scores is not None
            else np.ones(len(COCO_JOINTS))
        )
        points = xy[self.config.person_index]
        if len(points) != len(COCO_JOINTS):
            raise ValueError("pose checkpoint must emit COCO-17 keypoints")
        landmarks = tuple(
            HumanKeypoint2D(joint, (float(point[0]), float(point[1])), float(score))
            for joint, point, score in zip(COCO_JOINTS, points, confidences, strict=True)
            if np.isfinite(point).all() and np.isfinite(score) and score > 0
        )
        return HumanPose2D(
            source_header=source_header,
            person_id=f"detection-index-{self.config.person_index}",
            landmarks=landmarks,
            producer=f"ultralytics/{self.config.model}",
            produced_on=ComputePlacement(ComputeKind.HOST_CPU, "ultralytics-output"),
        )


__all__ = ["COCO_JOINTS", "YoloPoseConfig", "YoloPoseEstimator"]
