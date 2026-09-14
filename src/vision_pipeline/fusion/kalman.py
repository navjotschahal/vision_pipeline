"""A compact IMU-prediction and position-correction Kalman filter."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import ClockDomainMismatchError, FrameId, TimePoint
from vision_pipeline.geometry.spatial import Vector3

from .contracts import FusedKinematicState, ImuSample, PositionObservation3D


def _array3(value: Vector3) -> NDArray[np.float64]:
    return np.asarray(value, dtype=np.float64)


class ImuPositionKalmanFilter:
    """Fuse high-rate prepared IMU acceleration with asynchronous 3D positions.

    The state is ``[px, py, pz, vx, vy, vz]``. Orientation, gravity removal, sensor
    extrinsics, and clock mapping are deliberately outside this small research filter.
    """

    def __init__(
        self,
        *,
        initial_time: TimePoint,
        reference_frame: FrameId,
        initial_position_metres: Vector3 = (0.0, 0.0, 0.0),
        initial_velocity_metres_per_second: Vector3 = (0.0, 0.0, 0.0),
        initial_position_std_metres: float = 0.5,
        initial_velocity_std_metres_per_second: float = 1.0,
        acceleration_process_std_metres_per_second2: float = 0.2,
    ) -> None:
        for value, name in (
            (initial_position_std_metres, "initial_position_std_metres"),
            (initial_velocity_std_metres_per_second, "initial_velocity_std_metres_per_second"),
            (
                acceleration_process_std_metres_per_second2,
                "acceleration_process_std_metres_per_second2",
            ),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        self._time = initial_time
        self._reference_frame = reference_frame
        self._state = np.concatenate(
            (
                _array3(initial_position_metres),
                _array3(initial_velocity_metres_per_second),
            )
        )
        variances = np.asarray(
            [initial_position_std_metres**2] * 3 + [initial_velocity_std_metres_per_second**2] * 3,
            dtype=np.float64,
        )
        self._covariance = np.diag(variances)
        self._acceleration_process_variance = acceleration_process_std_metres_per_second2**2
        self._sources: set[str] = set()
        self._correction_count = 0

    def predict(self, sample: ImuSample) -> FusedKinematicState:
        """Advance to an IMU timestamp using constant acceleration over the interval."""

        self._require_time_domain(sample.measured_at)
        self._require_frame(sample.reference_frame)
        if not sample.gravity_compensated:
            raise ValueError("IMU acceleration must be gravity-compensated before fusion")
        elapsed_nanoseconds = sample.measured_at - self._time
        if elapsed_nanoseconds < 0:
            raise ValueError("out-of-order IMU sample")
        dt = elapsed_nanoseconds / 1_000_000_000
        if dt > 0:
            identity = np.eye(3, dtype=np.float64)
            transition = np.eye(6, dtype=np.float64)
            transition[:3, 3:] = identity * dt
            control = np.vstack((identity * (0.5 * dt * dt), identity * dt))
            acceleration = _array3(sample.linear_acceleration_metres_per_second2)
            acceleration_covariance = np.asarray(
                sample.acceleration_covariance,
                dtype=np.float64,
            ).reshape(3, 3)
            process_acceleration_covariance = acceleration_covariance + (
                identity * self._acceleration_process_variance
            )
            self._state = transition @ self._state + control @ acceleration
            self._covariance = (
                transition @ self._covariance @ transition.T
                + control @ process_acceleration_covariance @ control.T
            )
        self._time = sample.measured_at
        self._sources.add(sample.source_id)
        return self.state

    def correct_position(self, observation: PositionObservation3D) -> FusedKinematicState:
        """Correct the current state with one calibrated position observation."""

        self._require_time_domain(observation.measured_at)
        self._require_frame(observation.reference_frame)
        if observation.measured_at != self._time:
            raise ValueError("position correction timestamp must equal the current filter time")
        measurement = _array3(observation.position_metres)
        measurement_covariance = np.asarray(observation.covariance, dtype=np.float64).reshape(3, 3)
        if not np.allclose(measurement_covariance, measurement_covariance.T, atol=1e-12):
            raise ValueError("position covariance must be symmetric")
        if np.any(np.linalg.eigvalsh(measurement_covariance) <= 0):
            raise ValueError("position covariance must be positive definite")

        observation_matrix = np.zeros((3, 6), dtype=np.float64)
        observation_matrix[:, :3] = np.eye(3, dtype=np.float64)
        innovation = measurement - observation_matrix @ self._state
        innovation_covariance = (
            observation_matrix @ self._covariance @ observation_matrix.T + measurement_covariance
        )
        kalman_gain = np.linalg.solve(
            innovation_covariance,
            observation_matrix @ self._covariance,
        ).T
        self._state += kalman_gain @ innovation

        identity = np.eye(6, dtype=np.float64)
        residual = identity - kalman_gain @ observation_matrix
        # Joseph form preserves symmetry and positive semidefiniteness numerically.
        self._covariance = (
            residual @ self._covariance @ residual.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        self._sources.add(observation.source_id)
        self._correction_count += 1
        return self.state

    @property
    def state(self) -> FusedKinematicState:
        position = tuple(float(value) for value in self._state[:3])
        velocity = tuple(float(value) for value in self._state[3:])
        covariance = tuple(float(value) for value in self._covariance.reshape(-1))
        return FusedKinematicState(
            state_time=self._time,
            reference_frame=self._reference_frame,
            position_metres=position,  # type: ignore[arg-type]
            velocity_metres_per_second=velocity,  # type: ignore[arg-type]
            covariance=covariance,
            contributing_sources=tuple(sorted(self._sources)),
            correction_count=self._correction_count,
        )

    def _require_time_domain(self, timestamp: TimePoint) -> None:
        if timestamp.clock != self._time.clock:
            raise ClockDomainMismatchError(
                f"fusion clock {self._time.clock.domain_id!r} cannot consume "
                f"{timestamp.clock.domain_id!r} without a clock mapping"
            )

    def _require_frame(self, frame: FrameId) -> None:
        if frame != self._reference_frame:
            raise ValueError(
                f"fusion frame {self._reference_frame.value!r} cannot consume "
                f"{frame.value!r} without an extrinsic transform"
            )


__all__ = ["ImuPositionKalmanFilter"]
