from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from vision_pipeline.calibration import CharucoBoardSpec, CharucoTarget
from vision_pipeline.calibration.hand_eye import (
    HandEyeDegenerateError,
    HandEyeMount,
    HandEyeSample,
    board_observation_from_points,
    board_view_angle_deg,
    calibrate_hand_eye,
    estimate_board_pose,
    invert_transform,
    make_transform,
    matrix_from_rigid_transform,
    motion_diversity,
    relative_base_error,
    rigid_transform_from_matrix,
    rotation_angle_deg,
    select_diverse_pose,
    validate_hand_eye,
)
from vision_pipeline.contracts import FrameId

K = np.array([[606.0, 0.0, 323.9], [0.0, 606.2, 239.6], [0.0, 0.0, 1.0]])
NO_DISTORTION = np.zeros(5)
SPEC = CharucoBoardSpec(7, 5, 0.03, 0.022, "DICT_4X4_50")
BOARD_POINTS = np.asarray(SPEC.create_board().getChessboardCorners(), dtype=np.float64)


def rotation(axis: tuple[float, float, float], degrees: float) -> np.ndarray:
    vector = np.asarray(axis, dtype=np.float64)
    vector = vector / np.linalg.norm(vector) * math.radians(degrees)
    matrix, _ = cv2.Rodrigues(vector)
    return matrix


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """``base_from_camera`` for an optical frame (z forward, y down) at ``eye``."""

    z = (target - eye) / np.linalg.norm(target - eye)
    x = np.cross(z, up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return make_transform(np.column_stack((x, y, z)), eye)


def observe(
    camera_from_target: np.ndarray, rng: np.random.Generator, noise_pixels: float = 0.2
) -> object:
    rvec, _ = cv2.Rodrigues(camera_from_target[:3, :3])
    image, _ = cv2.projectPoints(BOARD_POINTS, rvec, camera_from_target[:3, 3], K, NO_DISTORTION)
    image = image.reshape(-1, 2) + rng.normal(0.0, noise_pixels, (len(BOARD_POINTS), 2))
    return board_observation_from_points(BOARD_POINTS, image, K, NO_DISTORTION)


def board_poses(
    rng: np.random.Generator, count: int, axes: list[tuple[float, float, float]] | None = None
) -> list[np.ndarray]:
    """Board poses in the camera frame, facing it and rotated up to 35 degrees."""

    poses = []
    for index in range(count):
        axis = tuple(rng.normal(size=3)) if axes is None else axes[index % len(axes)]
        tilt = rotation(axis, float(rng.uniform(-35, 35)))  # type: ignore[arg-type]
        position = (rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.08), rng.uniform(0.55, 0.85))
        centre = BOARD_POINTS.mean(axis=0)
        poses.append(make_transform(tilt, np.asarray(position) - tilt @ centre))
    return poses


def eye_to_hand_samples(
    rng: np.random.Generator,
    count: int,
    axes: list[tuple[float, float, float]] | None = None,
) -> tuple[list[HandEyeSample], np.ndarray, np.ndarray]:
    base_from_camera = look_at(
        np.array([0.95, 0.05, 0.55]), np.array([0.35, 0.0, 0.25]), np.array([0, 0, 1.0])
    )
    flange_from_target = make_transform(rotation((0.2, 1.0, 0.1), 25.0), (0.01, -0.03, 0.06))
    samples = []
    for index, camera_from_target in enumerate(board_poses(rng, count, axes)):
        base_from_flange = (
            base_from_camera @ camera_from_target @ invert_transform(flange_from_target)
        )
        observation = observe(camera_from_target, rng)
        samples.append(HandEyeSample(f"pose-{index:02d}", base_from_flange, observation))  # type: ignore[arg-type]
    return samples, base_from_camera, flange_from_target


def test_eye_to_hand_recovers_camera_and_board_offset() -> None:
    rng = np.random.default_rng(7)
    samples, truth, board_truth = eye_to_hand_samples(rng, 28)

    calibration = calibrate_hand_eye(samples[:20], HandEyeMount.EYE_TO_HAND, K, NO_DISTORTION)
    validation = validate_hand_eye(calibration, samples[20:], K, NO_DISTORTION)

    assert np.linalg.norm(calibration.camera_to_robot[:3, 3] - truth[:3, 3]) < 0.002
    assert rotation_angle_deg(calibration.camera_to_robot[:3, :3].T @ truth[:3, :3]) < 0.2
    assert np.linalg.norm(calibration.target_offset[:3, 3] - board_truth[:3, 3]) < 0.002
    assert calibration.reprojection_rms_pixels < 0.4
    assert calibration.rejected_sample_ids == ()
    assert set(calibration.closed_form) == {"tsai", "park", "horaud", "andreff", "daniilidis"}
    assert max(mm for mm, _ in calibration.closed_form_deviation.values()) < 10.0
    assert validation.samples == 8
    assert validation.reprojection_rms_pixels < 0.5
    assert validation.corner_error_mean_mm < 3.0


def test_eye_in_hand_recovers_wrist_camera() -> None:
    rng = np.random.default_rng(11)
    flange_from_camera = make_transform(rotation((1.0, 0.3, 0.0), 12.0), (0.05, 0.0, 0.08))
    base_from_target = make_transform(rotation((0, 0, 1.0), 30.0), (0.5, -0.1, 0.02))
    samples = []
    for index, camera_from_target in enumerate(board_poses(rng, 20)):
        base_from_camera = base_from_target @ invert_transform(camera_from_target)
        base_from_flange = base_from_camera @ invert_transform(flange_from_camera)
        samples.append(
            HandEyeSample(f"wrist-{index}", base_from_flange, observe(camera_from_target, rng))  # type: ignore[arg-type]
        )

    calibration = calibrate_hand_eye(samples, HandEyeMount.EYE_IN_HAND, K, NO_DISTORTION)

    assert np.linalg.norm(calibration.camera_to_robot[:3, 3] - flange_from_camera[:3, 3]) < 0.002
    assert (
        rotation_angle_deg(calibration.camera_to_robot[:3, :3].T @ flange_from_camera[:3, :3]) < 0.2
    )
    assert np.linalg.norm(calibration.target_offset[:3, 3] - base_from_target[:3, 3]) < 0.002


def test_rotations_about_one_axis_are_rejected_as_degenerate() -> None:
    rng = np.random.default_rng(3)
    samples, _, _ = eye_to_hand_samples(rng, 12, axes=[(0.0, 0.0, 1.0)])

    assert motion_diversity([s.base_from_flange[:3, :3] for s in samples]).axis_spread < 0.01
    with pytest.raises(HandEyeDegenerateError, match="parallel"):
        calibrate_hand_eye(samples, HandEyeMount.EYE_TO_HAND, K, NO_DISTORTION)


def test_a_bad_robot_pose_is_rejected_without_biasing_the_result() -> None:
    rng = np.random.default_rng(5)
    samples, truth, _ = eye_to_hand_samples(rng, 20)
    corrupted = samples[4].base_from_flange.copy()
    corrupted[:3, 3] += (0.03, -0.02, 0.0)  # e.g. joint states from the wrong instant
    samples[4] = HandEyeSample(samples[4].sample_id, corrupted, samples[4].board)

    calibration = calibrate_hand_eye(samples, HandEyeMount.EYE_TO_HAND, K, NO_DISTORTION)

    assert calibration.rejected_sample_ids == ("pose-04",)
    assert np.linalg.norm(calibration.camera_to_robot[:3, 3] - truth[:3, 3]) < 0.003


def test_too_few_samples_are_refused() -> None:
    samples, _, _ = eye_to_hand_samples(np.random.default_rng(1), 5)
    with pytest.raises(HandEyeDegenerateError, match="at least 8"):
        calibrate_hand_eye(samples, HandEyeMount.EYE_TO_HAND, K, NO_DISTORTION)


def test_board_pose_from_a_rendered_charuco_image() -> None:
    target = CharucoTarget(SPEC)
    board_image = target.generate_image(1400)
    # Board-plane metres -> board-image pixels, fitted from a detection on the flat image
    # so the test does not depend on OpenCV's margin and border layout.
    flat = target.detect(cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR))
    assert flat is not None
    source = BOARD_POINTS[flat.ids.reshape(-1), :2]
    affine = np.linalg.lstsq(
        np.c_[source, np.ones(len(source))], flat.corners.reshape(-1, 2), rcond=None
    )[0].T
    image_from_board = np.vstack((affine, (0.0, 0.0, 1.0)))
    camera_from_target = make_transform(rotation((0.3, 1.0, 0.0), 25.0), (-0.1, -0.06, 0.55))
    plane = K @ np.column_stack(
        (camera_from_target[:3, 0], camera_from_target[:3, 1], camera_from_target[:3, 3])
    )
    camera_image = cv2.warpPerspective(
        board_image,
        plane @ np.linalg.inv(image_from_board),
        (640, 480),
        flags=cv2.INTER_AREA,
        borderValue=255,
    )

    observation = target.detect(cv2.cvtColor(camera_image, cv2.COLOR_GRAY2BGR))
    assert observation is not None
    estimate = estimate_board_pose(target, observation, K, NO_DISTORTION)

    assert estimate is not None
    assert estimate.corner_count >= 20
    assert np.linalg.norm(estimate.camera_from_target[:3, 3] - camera_from_target[:3, 3]) < 0.003
    assert (
        rotation_angle_deg(estimate.camera_from_target[:3, :3].T @ camera_from_target[:3, :3]) < 0.5
    )
    assert board_view_angle_deg(estimate.camera_from_target) < 40.0


def test_collinear_corners_do_not_produce_a_pose() -> None:
    row = BOARD_POINTS[np.isclose(BOARD_POINTS[:, 1], BOARD_POINTS[0, 1])]
    image = np.column_stack((row[:, 0] * 1000 + 100, np.full(len(row), 200.0)))
    assert board_observation_from_points(row, image, K, NO_DISTORTION, minimum_corners=4) is None


def test_next_pose_is_the_most_differently_rotated_acceptable_candidate() -> None:
    captured = [
        make_transform(np.eye(3), (0, 0, 0)),
        make_transform(rotation((1, 0, 0), 20), (0, 0, 0)),
    ]
    candidates = [
        make_transform(rotation((1, 0, 0), 10), (0, 0, 0)),
        make_transform(rotation((0, 1, 0), 40), (0, 0, 0)),
        make_transform(rotation((0, 0, 1), 60), (0, 0, 0)),
    ]

    assert select_diverse_pose(candidates, captured) == 2
    # Excluding the z-rotated candidate leaves the 40-degree y rotation as most novel.
    assert select_diverse_pose(candidates, captured, acceptable=lambda pose: pose[2, 2] < 0.99) == 1
    assert select_diverse_pose([], captured) is None


def test_matrix_and_repository_transform_conventions_agree() -> None:
    matrix = make_transform(rotation((0.4, -1.0, 0.3), 130.0), (0.2, -0.5, 1.1))
    transform = rigid_transform_from_matrix(
        matrix, source_frame=FrameId("camera"), target_frame=FrameId("base")
    )
    point = np.array([0.3, -0.2, 0.7])

    assert transform.apply_point(tuple(point)) == pytest.approx(
        tuple((matrix @ np.r_[point, 1])[:3])
    )
    assert matrix_from_rigid_transform(transform) == pytest.approx(matrix)


def test_relative_base_error_is_zero_for_consistent_arms() -> None:
    left_base_from_camera = look_at(np.array([0.9, 0.2, 0.5]), np.zeros(3), np.array([0, 0, 1.0]))
    left_from_right = make_transform(rotation((0, 0, 1), 5), (0.0, 0.062, 0.0))
    right_base_from_camera = invert_transform(left_from_right) @ left_base_from_camera

    millimetres, degrees = relative_base_error(
        left_base_from_camera, right_base_from_camera, left_from_right
    )
    assert millimetres == pytest.approx(0.0, abs=1e-9)
    assert degrees == pytest.approx(0.0, abs=1e-6)
