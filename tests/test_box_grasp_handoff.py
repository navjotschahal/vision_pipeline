from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vision_pipeline.calibration.manual_extrinsic import (
    GravitySample,
    ManualExtrinsicError,
    TapeMeasurement,
    build_manual_extrinsic,
    load_tape_measurement,
    world_from_camera_rotation,
)
from vision_pipeline.perception.objects.box_grasp import (
    BoxEstimateWindow,
    BoxFitError,
    CpfGraspLimits,
    box_corners_world,
    cpf_handoff,
    fit_box_to_world_points,
)


def camera_rotation(heading_deg: float, pitch_down_deg: float, roll_deg: float) -> np.ndarray:
    """world_from_camera for an optical frame (x right, y down, z forward)."""

    heading, pitch, roll = (math.radians(v) for v in (heading_deg, pitch_down_deg, roll_deg))
    forward = np.array(
        [math.cos(heading) * math.cos(pitch), math.sin(heading) * math.cos(pitch), -math.sin(pitch)]
    )
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    level = np.column_stack((right, down, forward))
    # Roll about the optical axis.
    c, s = math.cos(roll), math.sin(roll)
    return level @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def gravity_for(
    world_from_camera: np.ndarray, color_from_imu: np.ndarray | None = None
) -> GravitySample:
    color_from_imu = np.eye(3) if color_from_imu is None else color_from_imu
    up_color = world_from_camera.T @ np.array([0.0, 0.0, 1.0])
    reading = color_from_imu.T @ (9.81 * up_color)
    return GravitySample(
        mean_acceleration_imu=tuple(reading),  # type: ignore[arg-type]
        standard_deviation_imu=(0.01, 0.01, 0.01),
        samples=300,
        duration_s=1.5,
        color_from_imu_rotation=tuple(color_from_imu.reshape(-1)),
    )


@pytest.mark.parametrize(
    ("heading", "pitch", "roll"), [(180.0, 45.0, 0.0), (150.0, 30.0, 4.0), (-60.0, 55.0, -3.0)]
)
def test_rotation_from_gravity_and_heading_recovers_the_camera(
    heading: float, pitch: float, roll: float
) -> None:
    truth = camera_rotation(heading, pitch, roll)
    up_in_camera = truth.T @ np.array([0.0, 0.0, 1.0])
    optical = truth @ np.array([0.0, 0.0, 1.0])

    estimate = world_from_camera_rotation(up_in_camera, math.atan2(optical[1], optical[0]))

    assert estimate == pytest.approx(truth, abs=1e-9)


def test_camera_looking_straight_down_has_no_heading() -> None:
    with pytest.raises(ManualExtrinsicError, match="heading"):
        world_from_camera_rotation(np.array([0.0, 0.0, -1.0]), 0.0)


def test_extrinsic_uses_imu_to_color_rotation_and_aim_point() -> None:
    truth = camera_rotation(170.0, 40.0, 2.0)
    tilt = np.array([[1, -0.0207, 0], [0.0207, 1, 0], [0, 0, 1.0]])
    u, _, vt = np.linalg.svd(tilt)
    color_from_imu = u @ vt
    camera = np.array([0.95, 0.10, 0.55])
    optical = truth @ np.array([0.0, 0.0, 1.0])
    aim = camera + optical * (-(camera[2] - 0.20) / optical[2])  # where the axis meets z = 0.20
    tape = TapeMeasurement(
        world_frame="openarm_body_link0",
        camera_frame="realsense_color_optical",
        reference_point_world_metres=(0.03, 0.0, 0.40),
        camera_offset_from_reference_metres=tuple(camera - (0.03, 0.0, 0.40)),  # type: ignore[arg-type]
        heading_deg=None,
        aim_point_world_metres=tuple(aim),  # type: ignore[arg-type]
        measured_on="2026-09-16",
    )

    extrinsic = build_manual_extrinsic(tape, gravity_for(truth, color_from_imu))

    assert extrinsic.world_from_camera[:3, :3] == pytest.approx(truth, abs=1e-9)
    assert extrinsic.world_from_camera[:3, 3] == pytest.approx(camera)
    assert extrinsic.camera_pitch_down_deg == pytest.approx(40.0)
    assert extrinsic.heading_deg == pytest.approx(170.0)
    assert extrinsic.as_rigid_transform().apply_point((0.0, 0.0, 1.0)) == pytest.approx(
        tuple(camera + optical)
    )


def test_tape_file_must_be_marked_measured(tmp_path: Path) -> None:
    path = tmp_path / "tape.yaml"
    body = """
world_frame: openarm_body_link0
camera_frame: realsense_color_optical
reference_point_world_metres: [0.03, 0.0, 0.40]
camera_offset_from_reference_metres: [0.9, 0.1, 0.15]
heading_deg: 180.0
measured_on: "2026-09-16"
"""
    path.write_text("measured: false\n" + body)
    with pytest.raises(ManualExtrinsicError, match="measured: true"):
        load_tape_measurement(path)
    path.write_text("measured: true\naim_point_world_metres: [0.3, 0.0, 0.2]\n" + body)
    with pytest.raises(ManualExtrinsicError, match="exactly one"):
        load_tape_measurement(path)
    path.write_text("measured: true\n" + body)
    assert load_tape_measurement(path).camera_position_world == pytest.approx((0.93, 0.1, 0.55))


def test_accelerometer_reading_must_look_like_gravity() -> None:
    sample = gravity_for(np.eye(3))
    moving = GravitySample(
        mean_acceleration_imu=(0.0, -3.0, 0.0),
        standard_deviation_imu=(1.0, 1.0, 1.0),
        samples=300,
        duration_s=1.5,
        color_from_imu_rotation=sample.color_from_imu_rotation,
    )
    with pytest.raises(ManualExtrinsicError, match="gravity"):
        moving.up_in_color()


def box_surface_points(
    centre: tuple[float, float, float],
    half: tuple[float, float, float],
    yaw_deg: float,
    rng: np.random.Generator,
    *,
    camera_xy: tuple[float, float] = (1.0, 0.1),
    noise: float = 0.002,
) -> np.ndarray:
    """Top face plus the two side faces a camera at ``camera_xy`` (above) can see."""

    yaw = math.radians(yaw_deg)
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1.0]]
    )
    ha, hb, hz = half
    faces = [rng.uniform((-ha, -hb, hz), (ha, hb, hz), (3000, 3))]
    to_camera = rotation.T @ (np.array([*camera_xy, 0.0]) - np.array([*centre[:2], 0.0]))
    faces.append(
        rng.uniform(
            (np.sign(to_camera[0]) * ha, -hb, -hz), (np.sign(to_camera[0]) * ha, hb, hz), (1500, 3)
        )
    )
    faces.append(
        rng.uniform(
            (-ha, np.sign(to_camera[1]) * hb, -hz), (ha, np.sign(to_camera[1]) * hb, hz), (1500, 3)
        )
    )
    local = np.vstack(faces)
    return local @ rotation.T + np.asarray(centre) + rng.normal(0, noise, local.shape)


@pytest.mark.parametrize("yaw", [0.0, 12.0, -30.0])
def test_box_fit_recovers_centre_extents_and_yaw(yaw: float) -> None:
    rng = np.random.default_rng(4)
    centre, half = (0.30, -0.02, 0.32), (0.08, 0.05, 0.10)
    points = box_surface_points(centre, half, yaw, rng)
    table = (np.array([0.0, 0.0, 1.0]), -0.22)  # z = 0.22 plane under the box

    estimate = fit_box_to_world_points(
        points, support_plane_world=table, frameset_id="f", selection_id=1, depth_captured_at=None
    )

    assert estimate.bottom_from_support_plane
    assert estimate.centre_world == pytest.approx(centre, abs=0.006)
    assert estimate.half_extents == pytest.approx(half, abs=0.006)
    assert estimate.yaw_deg == pytest.approx(yaw, abs=1.0)
    corners = box_corners_world(estimate)
    assert corners[:, 2].min() == pytest.approx(0.22, abs=0.002)


def test_box_fit_without_support_plane_uses_lowest_points() -> None:
    rng = np.random.default_rng(9)
    points = box_surface_points((0.3, 0.0, 0.3), (0.1, 0.1, 0.08), 0.0, rng)
    estimate = fit_box_to_world_points(
        points, support_plane_world=None, frameset_id="f", selection_id=1, depth_captured_at=None
    )
    assert not estimate.bottom_from_support_plane
    assert estimate.half_extents[2] == pytest.approx(0.08, abs=0.01)
    with pytest.raises(BoxFitError):
        fit_box_to_world_points(
            points[:10],
            support_plane_world=None,
            frameset_id="f",
            selection_id=1,
            depth_captured_at=None,
        )


def test_handoff_is_ready_only_for_a_square_stable_box_in_the_workspace() -> None:
    rng = np.random.default_rng(2)
    window = BoxEstimateWindow(size=30)
    for _ in range(20):
        points = box_surface_points((0.30, 0.0, 0.32), (0.08, 0.05, 0.10), 0.0, rng)
        window.add(
            fit_box_to_world_points(
                points,
                support_plane_world=(np.array([0, 0, 1.0]), -0.22),
                frameset_id="f",
                selection_id=3,
                depth_captured_at=None,
            )
        )
    stable = window.summary()
    assert stable is not None and stable.samples == 20

    record = cpf_handoff(stable, CpfGraspLimits(), extrinsic={"world_frame": "openarm_body_link0"})

    assert record["ready"], record["problems"]
    assert record["grasp_planner_args"][0] == "--object-pos"
    assert float(record["grasp_planner_args"][5]) == pytest.approx(0.05, abs=0.006)
    left = record["attractors_before_squeeze"]["left"]
    assert left[1] - record["object_pos"][1] == pytest.approx(record["half_width"] + 0.093)

    rotated = fit_box_to_world_points(
        box_surface_points((0.30, 0.0, 0.32), (0.08, 0.05, 0.10), 20.0, rng),
        support_plane_world=None,
        frameset_id="g",
        selection_id=4,
        depth_captured_at=None,
    )
    window.add(rotated)  # a new selection resets the window
    summary = window.summary()
    assert summary is not None and summary.samples == 1
    problems = cpf_handoff(summary, CpfGraspLimits(), extrinsic={})["problems"]
    assert any("yaw" in problem for problem in problems)
    assert any("samples" in problem for problem in problems)
