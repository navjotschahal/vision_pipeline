from __future__ import annotations

import cv2
import numpy as np
import pytest

from vision_pipeline.calibration import (
    CharucoBoardSpec,
    CharucoObservation,
    CharucoTarget,
    PairedCharucoObservation,
    PairedObservationCollector,
    calibrate_stereo,
)


def target() -> CharucoTarget:
    return CharucoTarget(CharucoBoardSpec(7, 5, 0.04, 0.03, "DICT_4X4_50"))


def test_generated_charuco_board_is_detectable() -> None:
    board = target()

    observation = board.detect(board.generate_image(1400))

    assert observation is not None
    assert observation.corner_count == 24


def test_collector_rejects_an_identical_view() -> None:
    board = target()
    observation = board.detect(board.generate_image(1400))
    assert observation is not None
    pair = PairedCharucoObservation(observation, observation, host_receipt_skew_ms=1.0)
    collector = PairedObservationCollector(
        required_pairs=2,
        minimum_shared_corners=12,
        minimum_view_novelty=0.05,
        maximum_pair_skew_ms=20,
    )

    assert collector.consider(pair) == "accepted"
    assert collector.consider(pair) == "duplicate-view"


def test_stereo_calibration_recovers_known_camera_transform() -> None:
    board = target()
    object_points = np.asarray(board.board.getChessboardCorners(), dtype=np.float32)
    ids = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)
    matrix_a = np.asarray(((800.0, 0, 640), (0, 805.0, 360), (0, 0, 1)))
    matrix_b = np.asarray(((790.0, 0, 640), (0, 795.0, 360), (0, 0, 1)))
    expected_rotation, _ = cv2.Rodrigues(np.asarray((0.01, -0.02, 0.005)))
    expected_translation = np.asarray(((-0.18,), (0.01,), (0.005,)))
    pairs: list[PairedCharucoObservation] = []
    for index in range(12):
        board_rotation_vector = np.asarray(
            (-0.3 + 0.05 * index, 0.2 * np.sin(index * 0.7), -0.12 + 0.02 * index)
        )
        board_rotation, _ = cv2.Rodrigues(board_rotation_vector)
        board_translation = np.asarray(
            ((-0.12 + 0.015 * (index % 4),), (-0.08 + 0.012 * (index % 3),), (0.7 + 0.04 * index,))
        )
        image_a, _ = cv2.projectPoints(
            object_points,
            board_rotation_vector,
            board_translation,
            matrix_a,
            None,
        )
        rotation_b = expected_rotation @ board_rotation
        translation_b = expected_rotation @ board_translation + expected_translation
        rotation_vector_b, _ = cv2.Rodrigues(rotation_b)
        image_b, _ = cv2.projectPoints(
            object_points,
            rotation_vector_b,
            translation_b,
            matrix_b,
            None,
        )
        pairs.append(
            PairedCharucoObservation(
                CharucoObservation(1280, 720, image_a.astype(np.float32), ids),
                CharucoObservation(1280, 720, image_b.astype(np.float32), ids),
                host_receipt_skew_ms=1.0,
            )
        )

    result = calibrate_stereo(board, pairs)

    assert result.rms_reprojection_error_pixels < 0.001
    assert result.translation_camera_a_to_b_metres == pytest.approx(
        tuple(expected_translation.reshape(-1)), abs=1e-4
    )
    assert np.asarray(result.rotation_camera_a_to_b).reshape(3, 3) == pytest.approx(
        expected_rotation, abs=1e-4
    )
