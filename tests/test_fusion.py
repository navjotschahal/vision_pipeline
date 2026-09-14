from __future__ import annotations

import pytest

from vision_pipeline.apps.fusion_simulation import run_fusion_simulation
from vision_pipeline.config import FusionExperimentConfig
from vision_pipeline.contracts import ClockDomain, ClockKind, FrameId, TimePoint
from vision_pipeline.fusion import ImuPositionKalmanFilter, ImuSample, PositionObservation3D

CLOCK = ClockDomain("simulation/test-fusion", ClockKind.SIMULATION)
WORLD = FrameId("world")
DIAGONAL_COVARIANCE = (0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01)


def imu(time_seconds: float, acceleration_x: float = 0.0) -> ImuSample:
    return ImuSample(
        source_id="imu/test",
        measured_at=TimePoint(round(time_seconds * 1_000_000_000), CLOCK),
        reference_frame=WORLD,
        linear_acceleration_metres_per_second2=(acceleration_x, 0.0, 0.0),
        angular_velocity_radians_per_second=(0.0, 0.0, 0.0),
        acceleration_covariance=DIAGONAL_COVARIANCE,
        gravity_compensated=True,
    )


def test_imu_prediction_integrates_position_and_velocity() -> None:
    filter_ = ImuPositionKalmanFilter(initial_time=TimePoint(0, CLOCK), reference_frame=WORLD)

    state = filter_.predict(imu(1.0, acceleration_x=2.0))

    assert state.position_metres == pytest.approx((1.0, 0.0, 0.0))
    assert state.velocity_metres_per_second == pytest.approx((2.0, 0.0, 0.0))
    assert state.contributing_sources == ("imu/test",)


def test_position_correction_reduces_position_error_and_records_source() -> None:
    filter_ = ImuPositionKalmanFilter(initial_time=TimePoint(0, CLOCK), reference_frame=WORLD)
    filter_.predict(imu(1.0))
    before = abs(filter_.state.position_metres[0] - 2.0)

    state = filter_.correct_position(
        PositionObservation3D(
            source_id="rgbd/camera-a",
            measured_at=TimePoint(1_000_000_000, CLOCK),
            reference_frame=WORLD,
            position_metres=(2.0, 0.0, 0.0),
            covariance=DIAGONAL_COVARIANCE,
        )
    )

    assert abs(state.position_metres[0] - 2.0) < before
    assert state.correction_count == 1
    assert state.contributing_sources == ("imu/test", "rgbd/camera-a")


def test_raw_gravity_including_imu_is_rejected() -> None:
    filter_ = ImuPositionKalmanFilter(initial_time=TimePoint(0, CLOCK), reference_frame=WORLD)
    sample = imu(0.01)
    sample = ImuSample(
        source_id=sample.source_id,
        measured_at=sample.measured_at,
        reference_frame=sample.reference_frame,
        linear_acceleration_metres_per_second2=sample.linear_acceleration_metres_per_second2,
        angular_velocity_radians_per_second=sample.angular_velocity_radians_per_second,
        acceleration_covariance=sample.acceleration_covariance,
        gravity_compensated=False,
    )

    with pytest.raises(ValueError, match="gravity-compensated"):
        filter_.predict(sample)


def test_simulated_camera_corrections_bound_imu_drift() -> None:
    result = run_fusion_simulation(FusionExperimentConfig(), print_progress=False)

    assert result.camera_a_updates == 60
    assert result.camera_b_updates == 120
    assert result.fused_position_rmse_metres < result.raw_camera_rmse_metres
    assert result.fused_position_rmse_metres < result.imu_dead_reckoning_rmse_metres / 5
