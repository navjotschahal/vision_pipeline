"""Deterministic asynchronous multi-sensor fusion learning exercise."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.config import FusionExperimentConfig
from vision_pipeline.contracts import ClockDomain, ClockKind, FrameId, TimePoint
from vision_pipeline.fusion import ImuPositionKalmanFilter, ImuSample, PositionObservation3D


@dataclass(frozen=True, slots=True)
class FusionSimulationResult:
    fused_position_rmse_metres: float
    imu_dead_reckoning_rmse_metres: float
    raw_camera_rmse_metres: float
    camera_a_updates: int
    camera_b_updates: int
    final_position_metres: tuple[float, float, float]
    final_velocity_metres_per_second: tuple[float, float, float]


def _rmse(errors: list[NDArray[np.float64]]) -> float:
    return math.sqrt(float(np.mean([float(error @ error) for error in errors])))


def _covariance(std: float) -> tuple[float, ...]:
    return tuple(float(value) for value in np.diag([std * std] * 3).reshape(-1))


def _true_acceleration(seconds: float) -> NDArray[np.float64]:
    return np.asarray(
        (
            0.18 * math.sin(0.9 * seconds),
            -0.12 * math.cos(0.6 * seconds),
            0.06 * math.sin(0.4 * seconds),
        ),
        dtype=np.float64,
    )


def run_fusion_simulation(
    config: FusionExperimentConfig,
    *,
    print_progress: bool = True,
) -> FusionSimulationResult:
    """Fuse one prepared IMU stream with two calibrated RGB-D position streams."""

    rng = np.random.default_rng(config.seed)
    clock = ClockDomain("simulation/fusion-experiment", ClockKind.SIMULATION)
    world = FrameId("world")
    start = TimePoint(0, clock)
    initial_truth_position = np.asarray((0.5, -0.3, 1.2), dtype=np.float64)
    initial_truth_velocity = np.asarray((0.1, 0.04, -0.02), dtype=np.float64)
    initial_estimate = initial_truth_position + np.asarray((0.2, -0.15, 0.1))
    filter_ = ImuPositionKalmanFilter(
        initial_time=start,
        reference_frame=world,
        initial_position_metres=tuple(initial_estimate),
        initial_velocity_metres_per_second=(0.0, 0.0, 0.0),
        acceleration_process_std_metres_per_second2=(
            config.acceleration_process_std_metres_per_second2
        ),
    )
    dead_reckoning = ImuPositionKalmanFilter(
        initial_time=start,
        reference_frame=world,
        initial_position_metres=tuple(initial_estimate),
        initial_velocity_metres_per_second=(0.0, 0.0, 0.0),
        acceleration_process_std_metres_per_second2=(
            config.acceleration_process_std_metres_per_second2
        ),
    )
    dt = 1 / config.imu_hz
    steps = round(config.duration_seconds * config.imu_hz)
    camera_a_period = config.imu_hz // config.camera_a_hz
    camera_b_period = config.imu_hz // config.camera_b_hz
    report_period = config.imu_hz // config.report_hz
    truth_position = initial_truth_position.copy()
    truth_velocity = initial_truth_velocity.copy()
    imu_bias = np.asarray((0.035, -0.025, 0.02), dtype=np.float64)
    fused_errors: list[NDArray[np.float64]] = []
    dead_errors: list[NDArray[np.float64]] = []
    camera_errors: list[NDArray[np.float64]] = []
    camera_a_updates = 0
    camera_b_updates = 0
    imu_covariance = _covariance(config.imu_noise_std_metres_per_second2)

    if print_progress:
        print("simulated target/body carries the IMU; RGB-D cameras observe its 3D position")
        print("time    truth xyz (m)              fused xyz (m)              error")
    for step in range(1, steps + 1):
        seconds = step * dt
        acceleration = _true_acceleration(seconds - dt / 2)
        truth_position += truth_velocity * dt + 0.5 * acceleration * dt * dt
        truth_velocity += acceleration * dt
        measured_acceleration = (
            acceleration
            + imu_bias
            + rng.normal(
                0,
                config.imu_noise_std_metres_per_second2,
                size=3,
            )
        )
        timestamp = TimePoint(round(seconds * 1_000_000_000), clock)
        imu_sample = ImuSample(
            source_id="imu/on-body",
            measured_at=timestamp,
            reference_frame=world,
            linear_acceleration_metres_per_second2=tuple(measured_acceleration),
            angular_velocity_radians_per_second=(0.0, 0.0, 0.0),
            acceleration_covariance=imu_covariance,
            gravity_compensated=True,
        )
        filter_.predict(imu_sample)
        dead_reckoning.predict(imu_sample)

        if step % camera_a_period == 0:
            error = rng.normal(0, config.camera_a_position_std_metres, size=3)
            filter_.correct_position(
                PositionObservation3D(
                    source_id="rgbd/camera-a",
                    measured_at=timestamp,
                    reference_frame=world,
                    position_metres=tuple(truth_position + error),
                    covariance=_covariance(config.camera_a_position_std_metres),
                )
            )
            camera_errors.append(error)
            camera_a_updates += 1
        if step % camera_b_period == 0:
            error = rng.normal(0, config.camera_b_position_std_metres, size=3)
            filter_.correct_position(
                PositionObservation3D(
                    source_id="rgbd/camera-b",
                    measured_at=timestamp,
                    reference_frame=world,
                    position_metres=tuple(truth_position + error),
                    covariance=_covariance(config.camera_b_position_std_metres),
                )
            )
            camera_errors.append(error)
            camera_b_updates += 1

        fused_position = np.asarray(filter_.state.position_metres)
        dead_position = np.asarray(dead_reckoning.state.position_metres)
        fused_errors.append(fused_position - truth_position)
        dead_errors.append(dead_position - truth_position)
        if print_progress and (step % report_period == 0 or step == steps):
            print(
                f"{seconds:4.1f}  "
                f"[{truth_position[0]:6.3f} {truth_position[1]:6.3f} {truth_position[2]:6.3f}]  "
                f"[{fused_position[0]:6.3f} {fused_position[1]:6.3f} {fused_position[2]:6.3f}]  "
                f"{np.linalg.norm(fused_position - truth_position):.3f} m"
            )

    state = filter_.state
    result = FusionSimulationResult(
        fused_position_rmse_metres=_rmse(fused_errors),
        imu_dead_reckoning_rmse_metres=_rmse(dead_errors),
        raw_camera_rmse_metres=_rmse(camera_errors),
        camera_a_updates=camera_a_updates,
        camera_b_updates=camera_b_updates,
        final_position_metres=state.position_metres,
        final_velocity_metres_per_second=state.velocity_metres_per_second,
    )
    if print_progress:
        print(
            "RMSE: "
            f"fused={result.fused_position_rmse_metres:.3f} m  "
            f"IMU-only={result.imu_dead_reckoning_rmse_metres:.3f} m  "
            f"raw RGB-D observations={result.raw_camera_rmse_metres:.3f} m"
        )
        print(
            f"corrections: camera-a={camera_a_updates} camera-b={camera_b_updates}; "
            "small Kalman matrices intentionally run on CPU"
        )
    return result


__all__ = ["FusionSimulationResult", "run_fusion_simulation"]
