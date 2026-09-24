"""RGB or aligned RGB-D pose-to-link pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import FrameId, SampleHeader
from vision_pipeline.geometry.camera import PinholeIntrinsics
from vision_pipeline.image import ImageFrame
from vision_pipeline.perception.pose import (
    DepthPoseLifter,
    HumanPose2D,
    HumanPose3D,
    HumanPoseEstimator,
)
from vision_pipeline.perception.pose.chains import LimbChain, visible_links


@dataclass(frozen=True, slots=True)
class PoseLinkObservation:
    pose_2d: HumanPose2D | None
    pose_3d: HumanPose3D | None
    image_links: dict[str, tuple[tuple[str, str], ...]]
    metric_links: dict[str, tuple[tuple[str, str], ...]]


class HumanPoseLinkPipeline:
    def __init__(
        self,
        estimator: HumanPoseEstimator,
        chains: tuple[LimbChain, ...],
        *,
        lifter: DepthPoseLifter | None = None,
        minimum_confidence: float = 0.6,
    ) -> None:
        if not chains or len({chain.name for chain in chains}) != len(chains):
            raise ValueError("chains must have unique names and at least one entry")
        self.estimator = estimator
        self.chains = chains
        self.lifter = lifter or DepthPoseLifter()
        self.minimum_confidence = minimum_confidence

    def process(
        self,
        image: ImageFrame,
        header: SampleHeader,
        *,
        depth_metres: NDArray[np.float32] | None = None,
        intrinsics: PinholeIntrinsics | None = None,
        optical_frame: FrameId | None = None,
    ) -> PoseLinkObservation:
        supplied = (depth_metres is not None, intrinsics is not None, optical_frame is not None)
        if any(supplied) and not all(supplied):
            raise ValueError(
                "aligned depth, intrinsics and optical frame must be supplied together"
            )
        if optical_frame is not None and optical_frame != header.frame_id:
            raise ValueError("optical frame must match the source image frame")
        pose_2d = self.estimator.estimate(image, header)
        pose_3d = (
            self.lifter.lift(pose_2d, depth_metres, intrinsics, reference_frame=optical_frame)
            if pose_2d is not None
            and all(supplied)
            and depth_metres is not None
            and intrinsics is not None
            and optical_frame is not None
            else None
        )

        def selected(
            pose: HumanPose2D | HumanPose3D | None,
        ) -> dict[str, tuple[tuple[str, str], ...]]:
            return {
                chain.name: tuple(
                    (a.value, b.value)
                    for a, b in visible_links(
                        pose, chain, minimum_confidence=self.minimum_confidence
                    )
                )
                if pose is not None
                else ()
                for chain in self.chains
            }

        return PoseLinkObservation(pose_2d, pose_3d, selected(pose_2d), selected(pose_3d))


__all__ = ["HumanPoseLinkPipeline", "PoseLinkObservation"]
