"""Camera-to-world extrinsic from tape measurements and the D435I accelerometer.

A deliberately simple, motion-free estimate for getting a grasp experiment running:

- **Roll and pitch** come from gravity. At rest the accelerometer reads the reaction to
  gravity, i.e. the world "up" direction, expressed in the IMU frame; the SDK's factory
  IMU-to-color rotation carries it into the color optical frame.
- **Position** of the color lens centre is taped relative to any reference point whose
  world coordinates are known.
- **Heading** (rotation about vertical) cannot come from the IMU (no magnetometer, and the
  gyro drifts). It is either taped directly as an angle, or derived from an *aim point*:
  put a marker on the table under the image-centre crosshair and tape its position; the
  horizontal direction from camera to marker is the optical axis heading.

The world frame is assumed gravity-aligned with ``z`` up (true for CPF's MuJoCo world,
``openarm_body_link0``: x forward, y left, z up). Accuracy is limited by the tape (about
1-2 cm) and by heading (1 deg is ~17 mm at 1 m); use it as a starting estimate, not as a
calibration.
"""

from __future__ import annotations

import importlib
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from vision_pipeline.contracts import CalibrationRef, FrameId
from vision_pipeline.geometry.spatial import RigidTransform3D

from .hand_eye import Matrix, make_transform, rigid_transform_from_matrix


class ManualExtrinsicError(ValueError):
    """The tape/IMU description is incomplete, unmeasured, or geometrically invalid."""


def _vector3(value: object, name: str) -> NDArray[np.float64]:
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 3:
        raise ManualExtrinsicError(f"{name} must be a list of three numbers")
    array = np.asarray([float(item) for item in value], dtype=np.float64)
    if not np.isfinite(array).all():
        raise ManualExtrinsicError(f"{name} must be finite")
    return array


@dataclass(frozen=True, slots=True)
class TapeMeasurement:
    """Tape measurements, all in world axes (metres) except ``heading_deg``.

    ``camera_offset_from_reference_metres`` runs from the reference point to the colour
    lens centre. Exactly one of ``heading_deg`` (optical axis direction projected on the
    floor, measured from world +x toward +y) or ``aim_point_world_metres`` is given.
    """

    world_frame: str
    camera_frame: str
    reference_point_world_metres: tuple[float, float, float]
    camera_offset_from_reference_metres: tuple[float, float, float]
    heading_deg: float | None
    aim_point_world_metres: tuple[float, float, float] | None
    measured_on: str
    notes: str = ""

    def __post_init__(self) -> None:
        for name in ("world_frame", "camera_frame", "measured_on"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ManualExtrinsicError(f"{name} must be a non-empty string")
        if (self.heading_deg is None) == (self.aim_point_world_metres is None):
            raise ManualExtrinsicError("give exactly one of heading_deg or aim_point_world_metres")
        if self.heading_deg is not None and not math.isfinite(self.heading_deg):
            raise ManualExtrinsicError("heading_deg must be finite")

    @property
    def camera_position_world(self) -> NDArray[np.float64]:
        return np.asarray(self.reference_point_world_metres, dtype=np.float64) + np.asarray(
            self.camera_offset_from_reference_metres, dtype=np.float64
        )

    def heading_rad(self) -> float:
        if self.heading_deg is not None:
            return math.radians(self.heading_deg)
        assert self.aim_point_world_metres is not None
        direction = np.asarray(self.aim_point_world_metres) - self.camera_position_world
        if math.hypot(direction[0], direction[1]) < 0.10:
            raise ManualExtrinsicError(
                "aim point is within 10 cm (horizontally) of the camera; heading is ill-defined"
            )
        return math.atan2(float(direction[1]), float(direction[0]))


def load_tape_measurement(path: str | Path) -> TapeMeasurement:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ManualExtrinsicError(f"{path}: top level must be a mapping")
    if raw.get("measured") is not True:
        raise ManualExtrinsicError(
            f"{path}: set 'measured: true' after replacing every placeholder with a tape reading"
        )
    try:
        aim = raw.get("aim_point_world_metres")
        heading = raw.get("heading_deg")
        return TapeMeasurement(
            world_frame=str(raw["world_frame"]),
            camera_frame=str(raw["camera_frame"]),
            reference_point_world_metres=tuple(
                _vector3(raw["reference_point_world_metres"], "reference_point_world_metres")
            ),
            camera_offset_from_reference_metres=tuple(
                _vector3(
                    raw["camera_offset_from_reference_metres"],
                    "camera_offset_from_reference_metres",
                )
            ),
            heading_deg=None if heading is None else float(heading),
            aim_point_world_metres=(
                None if aim is None else tuple(_vector3(aim, "aim_point_world_metres"))
            ),
            measured_on=str(raw["measured_on"]),
            notes=str(raw.get("notes", "")),
        )
    except KeyError as error:
        raise ManualExtrinsicError(f"{path}: missing {error}") from error


@dataclass(frozen=True, slots=True)
class GravitySample:
    """Averaged accelerometer reading and the SDK's IMU-to-colour rotation."""

    mean_acceleration_imu: tuple[float, float, float]
    standard_deviation_imu: tuple[float, float, float]
    samples: int
    duration_s: float
    color_from_imu_rotation: tuple[float, ...]

    def up_in_color(self) -> NDArray[np.float64]:
        rotation = np.asarray(self.color_from_imu_rotation, dtype=np.float64).reshape(3, 3)
        up = rotation @ np.asarray(self.mean_acceleration_imu, dtype=np.float64)
        norm = float(np.linalg.norm(up))
        if not 8.0 < norm < 11.5:
            raise ManualExtrinsicError(
                f"accelerometer magnitude {norm:.2f} m/s^2 is not gravity; was the camera moving?"
            )
        return up / norm


def sample_gravity(
    serial: str | None, *, duration_s: float = 1.5, rate_hz: int = 200
) -> GravitySample:
    """Read the D435I accelerometer alone while the camera is still.

    Uses its own pipeline because combining the motion stream with colour on a USB 2
    link timed out on this camera. Call before opening the RGB-D source.
    """

    rs: Any = importlib.import_module("pyrealsense2")  # optional; imported only here

    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, rate_hz)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # Resolve the colour profile only to read the factory extrinsics, then stream accel alone.
    profile = config.resolve(rs.pipeline_wrapper(pipeline))
    accel_profile = profile.get_stream(rs.stream.accel)
    color_profile = profile.get_stream(rs.stream.color)
    extrinsics = accel_profile.get_extrinsics_to(color_profile)
    # librealsense stores rotation column-major.
    color_from_imu = np.asarray(extrinsics.rotation, dtype=np.float64).reshape(3, 3).T

    motion_only = rs.config()
    if serial:
        motion_only.enable_device(serial)
    motion_only.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, rate_hz)
    readings: list[tuple[float, float, float]] = []
    pipeline.start(motion_only)
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration_s:
            frames = pipeline.wait_for_frames(2000)
            accel = frames.first_or_default(rs.stream.accel)
            if accel:
                data = accel.as_motion_frame().get_motion_data()
                readings.append((float(data.x), float(data.y), float(data.z)))
    finally:
        pipeline.stop()
    if len(readings) < 20:
        raise ManualExtrinsicError(f"only {len(readings)} accelerometer samples arrived")
    values = np.asarray(readings)
    mean = values.mean(axis=0)
    spread = values.std(axis=0)
    return GravitySample(
        mean_acceleration_imu=(float(mean[0]), float(mean[1]), float(mean[2])),
        standard_deviation_imu=(float(spread[0]), float(spread[1]), float(spread[2])),
        samples=len(readings),
        duration_s=time.monotonic() - started,
        color_from_imu_rotation=tuple(float(v) for v in color_from_imu.reshape(-1)),
    )


def world_from_camera_rotation(
    up_in_camera: NDArray[np.float64], heading_rad: float
) -> NDArray[np.float64]:
    """Rotation mapping camera-frame vectors into a z-up world with the given heading.

    ``up_in_camera`` fixes roll and pitch. The optical axis (camera +z), projected onto
    the horizontal plane, is placed at ``heading_rad`` from world +x toward world +y.
    """

    up = np.asarray(up_in_camera, dtype=np.float64)
    up = up / np.linalg.norm(up)
    optical = np.array([0.0, 0.0, 1.0])
    forward = optical - (optical @ up) * up
    if np.linalg.norm(forward) < math.sin(math.radians(5.0)):
        raise ManualExtrinsicError(
            "camera looks within 5 deg of straight up or down; heading is undefined"
        )
    forward /= np.linalg.norm(forward)
    side = np.cross(up, forward)
    world_x = math.cos(heading_rad) * forward - math.sin(heading_rad) * side
    world_y = np.cross(up, world_x)
    camera_from_world = np.column_stack((world_x, world_y, up))
    return camera_from_world.T


@dataclass(frozen=True, slots=True)
class ManualExtrinsic:
    """``world_from_camera`` with the evidence it was built from."""

    world_from_camera: Matrix
    tape: TapeMeasurement
    gravity: GravitySample
    camera_pitch_down_deg: float
    # Tilt of the image x axis from horizontal (0 when the image horizon is level).
    camera_roll_deg: float
    heading_deg: float

    @property
    def calibration(self) -> CalibrationRef:
        return CalibrationRef(
            calibration_id=f"{self.tape.camera_frame}-to-{self.tape.world_frame}/tape-imu",
            revision=self.tape.measured_on,
        )

    def as_rigid_transform(self) -> RigidTransform3D:
        return rigid_transform_from_matrix(
            self.world_from_camera,
            source_frame=FrameId(self.tape.camera_frame),
            target_frame=FrameId(self.tape.world_frame),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "method": "tape position + accelerometer roll/pitch + taped heading",
            "world_frame": self.tape.world_frame,
            "camera_frame": self.tape.camera_frame,
            "camera_position_world_metres": self.world_from_camera[:3, 3].tolist(),
            "heading_deg": self.heading_deg,
            "camera_pitch_down_deg": self.camera_pitch_down_deg,
            "camera_roll_deg": self.camera_roll_deg,
            "gravity_samples": self.gravity.samples,
            "measured_on": self.tape.measured_on,
            "world_from_camera": self.world_from_camera.tolist(),
        }


def build_manual_extrinsic(tape: TapeMeasurement, gravity: GravitySample) -> ManualExtrinsic:
    up = gravity.up_in_color()
    heading = tape.heading_rad()
    rotation = world_from_camera_rotation(up, heading)
    world_from_camera = make_transform(rotation, tape.camera_position_world)
    optical_world = rotation @ np.array([0.0, 0.0, 1.0])
    right_world = rotation @ np.array([1.0, 0.0, 0.0])
    pitch_down = math.degrees(math.asin(max(-1.0, min(1.0, -float(optical_world[2])))))
    roll = math.degrees(math.asin(max(-1.0, min(1.0, -float(right_world[2])))))
    return ManualExtrinsic(
        world_from_camera=world_from_camera,
        tape=tape,
        gravity=gravity,
        camera_pitch_down_deg=pitch_down,
        camera_roll_deg=roll,
        heading_deg=math.degrees(heading),
    )


__all__ = [
    "GravitySample",
    "ManualExtrinsic",
    "ManualExtrinsicError",
    "TapeMeasurement",
    "build_manual_extrinsic",
    "load_tape_measurement",
    "sample_gravity",
    "world_from_camera_rotation",
]
