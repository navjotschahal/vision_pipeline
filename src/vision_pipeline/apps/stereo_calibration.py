"""Live two-camera ChArUco calibration host."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import cast

import cv2
import numpy as np
import yaml
from numpy.typing import NDArray

from vision_pipeline.calibration import (
    CameraCalibration,
    CharucoBoardSpec,
    CharucoObservation,
    CharucoTarget,
    PairedCharucoObservation,
    PairedObservationCollector,
    StereoCalibration,
    calibrate_stereo,
    common_corner_count,
)
from vision_pipeline.config import StereoCalibrationAppConfig
from vision_pipeline.contracts import ClockDomain, ClockKind, FrameId
from vision_pipeline.sources.opencv_webcam import OpenCvWebcam, WebcamBackend, WebcamConfig


def _target(config: StereoCalibrationAppConfig) -> CharucoTarget:
    return CharucoTarget(
        CharucoBoardSpec(
            squares_x=config.squares_x,
            squares_y=config.squares_y,
            square_length_metres=config.square_length_metres,
            marker_length_metres=config.marker_length_metres,
            dictionary=config.dictionary,
        )
    )


def generate_calibration_board(config: StereoCalibrationAppConfig) -> Path:
    """Generate the configured board image for printing without rescaling."""

    path = Path(config.board_image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = _target(config).generate_image(config.board_image_width_pixels)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"OpenCV could not write calibration board {path}")
    return path


def _camera_document(calibration: CameraCalibration, frame_id: str) -> dict[str, object]:
    return {
        "frame_id": frame_id,
        "image_width": calibration.image_width,
        "image_height": calibration.image_height,
        "camera_matrix_row_major": list(calibration.camera_matrix),
        "distortion_coefficients": list(calibration.distortion_coefficients),
        "rms_reprojection_error_pixels": calibration.rms_reprojection_error_pixels,
        "mean_reprojection_error_pixels": calibration.mean_reprojection_error_pixels,
        "observation_count": calibration.observation_count,
    }


def save_stereo_calibration(
    calibration: StereoCalibration,
    config: StereoCalibrationAppConfig,
    skews_ms: list[float],
) -> Path:
    """Write a portable, human-readable calibration bundle."""

    document = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "board": {
            "type": "charuco",
            "squares_x": config.squares_x,
            "squares_y": config.squares_y,
            "square_length_metres": config.square_length_metres,
            "marker_length_metres": config.marker_length_metres,
            "dictionary": config.dictionary,
        },
        "camera_a": _camera_document(calibration.camera_a, config.camera_a_frame_id),
        "camera_b": _camera_document(calibration.camera_b, config.camera_b_frame_id),
        "extrinsics": {
            "convention": "X_camera_b = R_camera_a_to_b * X_camera_a + t_camera_a_to_b",
            "rotation_camera_a_to_b_row_major": list(calibration.rotation_camera_a_to_b),
            "translation_camera_a_to_b_metres": list(calibration.translation_camera_a_to_b_metres),
            "essential_matrix_row_major": list(calibration.essential_matrix),
            "fundamental_matrix_row_major": list(calibration.fundamental_matrix),
            "rms_reprojection_error_pixels": calibration.rms_reprojection_error_pixels,
            "paired_observation_count": calibration.paired_observation_count,
        },
        "timing": {
            "status": "host-receipt-skew-observed; hardware synchronization not established",
            "mean_sequential_read_skew_ms": mean(skews_ms),
            "maximum_sequential_read_skew_ms": max(skews_ms),
        },
    }
    path = Path(config.output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _draw_observation(
    image: NDArray[np.uint8],
    observation: CharucoObservation | None,
) -> NDArray[np.uint8]:
    preview = image.copy()
    if observation is not None:
        cv2.aruco.drawDetectedCornersCharuco(
            preview,
            observation.corners,
            observation.ids,
            (0, 255, 0),
        )
    return preview


def _annotate(
    image: NDArray[np.uint8],
    text: str,
    *,
    y: int,
    color: tuple[int, int, int] = (255, 255, 255),
) -> None:
    cv2.putText(
        image,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )


def _preview_pair(
    camera_a: NDArray[np.uint8],
    camera_b: NDArray[np.uint8],
    *,
    observation_a: CharucoObservation | None,
    observation_b: CharucoObservation | None,
    accepted: int,
    required: int,
    decision: str,
    skew_ms: float,
) -> NDArray[np.uint8]:
    view_a = _draw_observation(camera_a, observation_a)
    view_b = _draw_observation(camera_b, observation_b)
    target_height = 360
    target_width = 640
    view_a = cast(NDArray[np.uint8], cv2.resize(view_a, (target_width, target_height)))
    view_b = cast(NDArray[np.uint8], cv2.resize(view_b, (target_width, target_height)))
    _annotate(view_a, "A: laptop webcam", y=28)
    _annotate(view_b, "B: iPhone camera", y=28)
    combined = np.hstack((view_a, view_b))
    _annotate(
        combined,
        f"automatic pairs {accepted}/{required}  decision={decision}  read-skew={skew_ms:.1f} ms",
        y=target_height - 42,
        color=(0, 255, 255),
    )
    _annotate(
        combined,
        "move board through both views: center, edges, near/far, and tilted; Q aborts",
        y=target_height - 14,
    )
    return combined


def run_stereo_calibration(config: StereoCalibrationAppConfig) -> Path | None:
    """Collect diverse synchronized board views and calibrate both cameras."""

    target = _target(config)
    collector = PairedObservationCollector(
        required_pairs=config.required_pairs,
        minimum_shared_corners=config.minimum_shared_corners,
        minimum_view_novelty=config.minimum_view_novelty,
        maximum_pair_skew_ms=config.maximum_pair_skew_ms,
    )
    shared_clock = ClockDomain("host/monotonic/stereo-calibration", ClockKind.HOST_MONOTONIC)
    backend = WebcamBackend(config.backend)
    camera_a = OpenCvWebcam(
        WebcamConfig(
            device_index=config.camera_a_device_index,
            backend=backend,
            width=config.width,
            height=config.height,
            fps=config.fps,
            source_id="calibration/camera-a/rgb",
            frame_id=FrameId(config.camera_a_frame_id),
        ),
        host_clock=shared_clock,
    )
    camera_b = OpenCvWebcam(
        WebcamConfig(
            device_index=config.camera_b_device_index,
            backend=backend,
            width=config.width,
            height=config.height,
            fps=config.fps,
            source_id="calibration/camera-b/rgb",
            frame_id=FrameId(config.camera_b_frame_id),
        ),
        host_clock=shared_clock,
    )
    window = "automatic stereo ChArUco calibration"
    skews_ms: list[float] = []
    aborted = False
    cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(window, 1280, 420)
    try:
        with camera_a, camera_b:
            print(
                f"camera A={camera_a.info.actual.width}x{camera_a.info.actual.height} "
                f"camera B={camera_b.info.actual.width}x{camera_b.info.actual.height}"
            )
            while not collector.complete:
                sample_a = camera_a.read()
                sample_b = camera_b.read()
                skew_ms = (
                    abs(
                        sample_b.header.received_at.nanoseconds
                        - sample_a.header.received_at.nanoseconds
                    )
                    / 1_000_000
                )
                observation_a = target.detect(sample_a.payload.data)
                observation_b = target.detect(sample_b.payload.data)
                decision = "board-not-visible-in-both"
                if observation_a is not None and observation_b is not None:
                    pair = PairedCharucoObservation(
                        camera_a=observation_a,
                        camera_b=observation_b,
                        host_receipt_skew_ms=skew_ms,
                    )
                    decision = collector.consider(pair)
                    if decision == "accepted":
                        skews_ms.append(skew_ms)
                        print(
                            f"accepted pair {len(collector.observations)}/{config.required_pairs} "
                            f"shared-corners={common_corner_count(pair)} skew={skew_ms:.1f} ms"
                        )
                preview = _preview_pair(
                    sample_a.payload.data,
                    sample_b.payload.data,
                    observation_a=observation_a,
                    observation_b=observation_b,
                    accepted=len(collector.observations),
                    required=config.required_pairs,
                    decision=decision,
                    skew_ms=skew_ms,
                )
                cv2.imshow(window, preview)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    aborted = True
                    break
    finally:
        cv2.destroyWindow(window)
    if aborted:
        return None

    calibration = calibrate_stereo(target, collector.observations)
    path = save_stereo_calibration(calibration, config, skews_ms)
    print(
        f"calibration RMS={calibration.rms_reprojection_error_pixels:.3f} px "
        f"A mean={calibration.camera_a.mean_reprojection_error_pixels:.3f} px "
        f"B mean={calibration.camera_b.mean_reprojection_error_pixels:.3f} px"
    )
    print(f"wrote calibration: {path}")
    return path


__all__ = [
    "generate_calibration_board",
    "run_stereo_calibration",
    "save_stereo_calibration",
]
