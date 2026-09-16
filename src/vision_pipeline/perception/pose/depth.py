"""Robustly lift image landmarks with depth aligned to the same color frame."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import ComputeKind, ComputePlacement, FrameId
from vision_pipeline.geometry.camera import PinholeIntrinsics
from vision_pipeline.perception.pose.contracts import (
    HumanKeypoint3D,
    HumanPose2D,
    HumanPose3D,
)


@dataclass(frozen=True, slots=True)
class DepthLiftingConfig:
    """Validity policy for a local depth patch around each RGB landmark."""

    window_radius_pixels: int = 2
    minimum_valid_samples: int = 3
    minimum_depth_metres: float = 0.2
    maximum_depth_metres: float = 4.0
    maximum_depth_mad_metres: float = 0.08
    minimum_keypoint_confidence: float = 0.5

    def __post_init__(self) -> None:
        for name in ("window_radius_pixels", "minimum_valid_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.window_radius_pixels < 0:
            raise ValueError("window_radius_pixels must be non-negative")
        if self.minimum_valid_samples <= 0:
            raise ValueError("minimum_valid_samples must be positive")
        for name in (
            "minimum_depth_metres",
            "maximum_depth_metres",
            "maximum_depth_mad_metres",
            "minimum_keypoint_confidence",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not 0 <= self.minimum_depth_metres < self.maximum_depth_metres:
            raise ValueError("depth limits must be non-negative and increasing")
        if self.maximum_depth_mad_metres < 0:
            raise ValueError("maximum_depth_mad_metres must be non-negative")
        if not 0 <= self.minimum_keypoint_confidence <= 1:
            raise ValueError("minimum_keypoint_confidence must be between zero and one")


class DepthPoseLifter:
    """Lift 2D joints into an optical frame without inventing missing depth."""

    def __init__(self, config: DepthLiftingConfig | None = None) -> None:
        self._config = config or DepthLiftingConfig()

    def lift(
        self,
        pose: HumanPose2D,
        depth_metres: NDArray[np.float32],
        intrinsics: PinholeIntrinsics,
        *,
        reference_frame: FrameId,
    ) -> HumanPose3D:
        if not isinstance(pose, HumanPose2D):
            raise TypeError("pose must be a HumanPose2D")
        if not isinstance(depth_metres, np.ndarray):
            raise TypeError("depth_metres must be a numpy.ndarray")
        if depth_metres.ndim != 2:
            raise ValueError("depth_metres must have HxW shape")
        if not np.issubdtype(depth_metres.dtype, np.floating):
            raise ValueError("depth_metres must use a floating-point dtype")
        if depth_metres.shape != (intrinsics.height, intrinsics.width):
            raise ValueError("depth dimensions must match the aligned-depth intrinsics")
        if not isinstance(reference_frame, FrameId):
            raise TypeError("reference_frame must be a FrameId")

        lifted: list[HumanKeypoint3D] = []
        for landmark in pose.landmarks:
            if landmark.confidence < self._config.minimum_keypoint_confidence:
                continue
            x, y = landmark.image_xy
            column = int(round(x))
            row = int(round(y))
            if not 0 <= column < intrinsics.width or not 0 <= row < intrinsics.height:
                continue
            radius = self._config.window_radius_pixels
            patch = depth_metres[
                max(0, row - radius) : min(intrinsics.height, row + radius + 1),
                max(0, column - radius) : min(intrinsics.width, column + radius + 1),
            ]
            valid = patch[
                np.isfinite(patch)
                & (patch >= self._config.minimum_depth_metres)
                & (patch <= self._config.maximum_depth_metres)
            ]
            if valid.size < self._config.minimum_valid_samples:
                continue
            depth = float(np.median(valid))
            mad = float(np.median(np.abs(valid - depth)))
            if mad > self._config.maximum_depth_mad_metres:
                continue
            lifted.append(
                HumanKeypoint3D(
                    joint=landmark.joint,
                    position_metres=(
                        (x - intrinsics.cx) * depth / intrinsics.fx,
                        (y - intrinsics.cy) * depth / intrinsics.fy,
                        depth,
                    ),
                    confidence=landmark.confidence,
                    depth_mad_metres=mad,
                    depth_sample_count=int(valid.size),
                )
            )

        return HumanPose3D(
            source_header=pose.source_header,
            person_id=pose.person_id,
            reference_frame=reference_frame,
            landmarks=tuple(lifted),
            producer=f"{pose.producer}+aligned-depth-median",
            produced_on=ComputePlacement(ComputeKind.HOST_CPU, "numpy"),
        )


__all__ = ["DepthLiftingConfig", "DepthPoseLifter"]
