from __future__ import annotations

import numpy as np
import pytest

from vision_pipeline.contracts import (
    ClockDomain,
    ClockKind,
    ComputeKind,
    ComputePlacement,
    FrameId,
    MeasurementKind,
    SampleHeader,
    TimePoint,
)
from vision_pipeline.geometry import PinholeIntrinsics
from vision_pipeline.perception.pose import (
    DepthLiftingConfig,
    DepthPoseLifter,
    HumanJoint,
    HumanKeypoint2D,
    HumanPose2D,
)

CLOCK = ClockDomain("host/test-pose", ClockKind.HOST_MONOTONIC)
OPTICAL = FrameId("realsense_color_optical")
CPU = ComputePlacement(ComputeKind.HOST_CPU, "test-cpu")


def header(sequence: int = 0) -> SampleHeader:
    return SampleHeader(
        sample_id=f"rgb/{sequence}",
        source_id="realsense/rgb",
        sequence_number=sequence,
        measurement_kind=MeasurementKind.RGB_IMAGE,
        captured_at=None,
        received_at=TimePoint(1_000_000_000 + sequence, CLOCK),
        frame_id=OPTICAL,
        calibration=None,
        producer="test-camera",
        produced_on=CPU,
    )


def pose(*landmarks: HumanKeypoint2D) -> HumanPose2D:
    return HumanPose2D(
        source_header=header(),
        person_id="operator-1",
        landmarks=landmarks,
        producer="fake-pose",
        produced_on=CPU,
    )


def test_depth_lifter_uses_robust_patch_and_aligned_intrinsics() -> None:
    depth = np.full((5, 5), 2.0, dtype=np.float32)
    depth[2, 3] = 0.0
    observation = pose(
        HumanKeypoint2D(HumanJoint.LEFT_WRIST, (3.0, 2.0), 0.9),
        HumanKeypoint2D(HumanJoint.RIGHT_WRIST, (1.0, 2.0), 0.2),
    )
    lifter = DepthPoseLifter(
        DepthLiftingConfig(window_radius_pixels=1, minimum_valid_samples=3)
    )

    result = lifter.lift(
        observation,
        depth,
        PinholeIntrinsics(5, 5, fx=2, fy=2, cx=2, cy=2),
        reference_frame=OPTICAL,
    )

    assert len(result.landmarks) == 1
    wrist = result.get(HumanJoint.LEFT_WRIST)
    assert wrist is not None
    assert wrist.position_metres == pytest.approx((1.0, 0.0, 2.0))
    assert wrist.depth_sample_count == 8
    assert wrist.depth_mad_metres == 0
    assert result.source_header is observation.source_header
    assert result.reference_frame == OPTICAL
    assert result.producer == "fake-pose+aligned-depth-median"
    assert result.get(HumanJoint.RIGHT_WRIST) is None


def test_depth_lifter_rejects_discontinuous_and_missing_depth() -> None:
    depth = np.array(
        [
            [1.0, 1.0, 1.0],
            [3.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    observation = pose(HumanKeypoint2D(HumanJoint.LEFT_WRIST, (1.0, 1.0), 1.0))
    lifter = DepthPoseLifter(
        DepthLiftingConfig(
            window_radius_pixels=1,
            minimum_valid_samples=3,
            maximum_depth_mad_metres=0.1,
        )
    )

    result = lifter.lift(
        observation,
        depth,
        PinholeIntrinsics(3, 3, fx=2, fy=2, cx=1, cy=1),
        reference_frame=OPTICAL,
    )

    # Median absolute deviation catches a landmark straddling foreground/background.
    assert result.landmarks == ()


def test_depth_lifter_requires_exact_aligned_grid() -> None:
    observation = pose(HumanKeypoint2D(HumanJoint.LEFT_WRIST, (1.0, 1.0), 1.0))
    lifter = DepthPoseLifter()

    with pytest.raises(ValueError, match="dimensions must match"):
        lifter.lift(
            observation,
            np.ones((2, 2), dtype=np.float32),
            PinholeIntrinsics(3, 3, fx=2, fy=2, cx=1, cy=1),
            reference_frame=OPTICAL,
        )
    with pytest.raises(ValueError, match="floating-point"):
        lifter.lift(
            observation,
            np.ones((3, 3), dtype=np.uint16),  # type: ignore[arg-type]
            PinholeIntrinsics(3, 3, fx=2, fy=2, cx=1, cy=1),
            reference_frame=OPTICAL,
        )


def test_pose_contract_rejects_duplicate_anatomical_joints() -> None:
    landmark = HumanKeypoint2D(HumanJoint.LEFT_WRIST, (1.0, 1.0), 1.0)

    with pytest.raises(ValueError, match="duplicate"):
        pose(landmark, landmark)

