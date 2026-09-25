"""Hand-eye calibration of the fixed RealSense against the OpenArm, through CPF.

    python -m vision_pipeline.apps.openarm_hand_eye board                      # printable board
    python -m vision_pipeline.apps.openarm_hand_eye capture --arm right --square-mm 30.0
    python -m vision_pipeline.apps.openarm_hand_eye solve --session <session dir>
    python -m vision_pipeline.apps.openarm_hand_eye check --arm right          # overlay the arm

Semi-autonomous eye-to-hand calibration. The ChArUco board is clamped in the hand, the
CPF driver runs in a hand-guidable mode with ``--shm`` so it publishes joint state, and
the operator moves the arm by hand. The tool captures on its own whenever

- the driver is publishing and not faulted,
- the arm has been still for ``--settle-s`` and the board is detected with a good PnP,
- the hand orientation differs from every captured pose by ``--min-rotation-deg``,

averages the board corners and joint angles over several frames, and stores the image,
the joint state, and ``world_from_hand`` (MuJoCo forward kinematics in CPF's world frame,
``openarm_body_link0``). ``s`` or quitting solves: closed form plus reprojection
refinement, held-out validation, and ``calibrations/openarm_v1/current.json`` for
``cpf_box_handoff``. Nothing here commands motion; the command block is never written.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.calibration.charuco import CharucoBoardSpec, CharucoTarget
from vision_pipeline.calibration.hand_eye import (
    BoardObservation,
    HandEyeCalibration,
    HandEyeDegenerateError,
    HandEyeMount,
    HandEyeSample,
    HandEyeValidation,
    Matrix,
    board_observation_from_points,
    board_view_angle_deg,
    calibrate_hand_eye,
    invert_transform,
    motion_diversity,
    rotation_angle_deg,
    validate_hand_eye,
)
from vision_pipeline.calibration.openarm_cpf import (
    CALIBRATION_DIR,
    CURRENT_CALIBRATION,
    DEFAULT_RIG_PATH,
    REPOSITORY_ROOT,
    RESULT_SCHEMA,
    ArmState,
    CpfArm,
    CpfRig,
    MujocoForwardKinematics,
    SharedMemoryArms,
    arm_skeleton_bodies,
    camera_orientation_summary,
    load_cpf_rig,
    load_extrinsic_file,
    save_hand_eye_result,
    transform_difference,
)

WINDOW = "openarm hand-eye"
SESSIONS_DIR = CALIBRATION_DIR / "sessions"
TAPE_EXTRINSIC = REPOSITORY_ROOT / "recordings" / "cpf_handoffs" / "last_extrinsic.json"
_ZERO_DISTORTION = np.zeros(5)
_GREEN = (60, 220, 60)
_YELLOW = (0, 220, 255)
_RED = (60, 60, 255)
_CYAN = (255, 220, 0)
_WHITE = (235, 235, 235)
_MAGENTA = (255, 0, 255)


# ---------------------------------------------------------------------------------------
# Camera


class ColorCamera:
    """Colour-only RealSense stream; depth is not needed for a ChArUco calibration.

    The calibration is of the colour optical frame, which is the same physical frame at
    every colour resolution, so it can be captured at 1280x720 (larger markers) and used
    by the 640x480 pipeline. Corners are undistorted with librealsense's own model before
    PnP, so the solver always sees a pinhole camera; the D435I reports zero coefficients,
    the D455 does not.
    """

    def __init__(self, serial: str | None, width: int, height: int, fps: int | None) -> None:
        rs: Any = importlib.import_module("pyrealsense2")
        self._rs = rs
        device = None
        for candidate in rs.context().query_devices():
            if serial is None or candidate.get_info(rs.camera_info.serial_number) == serial:
                device = candidate
                break
        if device is None:
            raise RuntimeError(f"no RealSense device with serial {serial!r} is connected")
        self.serial = str(device.get_info(rs.camera_info.serial_number))
        self.name = str(device.get_info(rs.camera_info.name))
        self.usb = (
            str(device.get_info(rs.camera_info.usb_type_descriptor))
            if device.supports(rs.camera_info.usb_type_descriptor)
            else "?"
        )
        self.firmware = (
            str(device.get_info(rs.camera_info.firmware_version))
            if device.supports(rs.camera_info.firmware_version)
            else None
        )
        if fps is None:
            # USB 2 sustains 1280x720 colour at 6 fps only (measured 2026-09-24).
            fps = 30 if (width, height) == (640, 480) else (15 if self.usb.startswith("3") else 6)
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self._pipeline = rs.pipeline()
        profile = self._pipeline.start(config)
        video = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self.intrinsics = video.get_intrinsics()
        i = self.intrinsics
        self.width, self.height, self.fps = int(i.width), int(i.height), int(video.fps())
        self.camera_matrix = np.array(
            [[float(i.fx), 0.0, float(i.ppx)], [0.0, float(i.fy), float(i.ppy)], [0.0, 0.0, 1.0]]
        )
        self.distortion_model = str(i.model).rsplit(".", maxsplit=1)[-1]
        self.distortion_coefficients = [float(c) for c in i.coeffs]
        self._undistort = any(abs(c) > 0.0 for c in self.distortion_coefficients)
        for _ in range(5):  # let auto-exposure settle
            self._pipeline.wait_for_frames(5000)

    def read(self, timeout_ms: int = 5000) -> NDArray[np.uint8]:
        frames = self._pipeline.wait_for_frames(timeout_ms)
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("frameset without a colour frame")
        return cast(NDArray[np.uint8], np.asanyarray(color.get_data()).copy())

    def undistort_points(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not self._undistort:
            return points
        k = self.camera_matrix
        out = np.empty_like(points)
        for row, (u, v) in enumerate(points):
            x, y, z = self._rs.rs2_deproject_pixel_to_point(
                self.intrinsics, [float(u), float(v)], 1.0
            )
            out[row] = (k[0, 0] * x / z + k[0, 2], k[1, 1] * y / z + k[1, 2])
        return out

    def info(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "serial": self.serial,
            "firmware": self.firmware,
            "usb_type_descriptor": self.usb,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "camera_matrix": self.camera_matrix.tolist(),
            "distortion_model": self.distortion_model,
            "distortion_coefficients": self.distortion_coefficients,
            "corners_undistorted_by": "librealsense rs2_deproject_pixel_to_point"
            if self._undistort
            else "none needed (zero coefficients)",
        }

    def close(self) -> None:
        with contextlib.suppress(Exception):  # a dead pipeline must not mask the cause
            self._pipeline.stop()


# ---------------------------------------------------------------------------------------
# Board detection


@dataclass(frozen=True, slots=True)
class Detection:
    ids: NDArray[np.int32]
    corners_px: NDArray[np.float64]
    corners_undistorted: NDArray[np.float64]
    observation: BoardObservation
    view_angle_deg: float

    @property
    def distance_m(self) -> float:
        return float(np.linalg.norm(self.observation.camera_from_target[:3, 3]))


@dataclass(frozen=True, slots=True)
class ArucoGridSpec:
    """A grid of plain ArUco markers (OpenCV ``GridBoard``), no chessboard between them.

    ``ids`` lists the marker ids row by row, left to right, with the board upright; the
    default counts up from ``first_id``. Lengths are the printed black square's edge and
    the white gap between neighbouring squares, both measured on the print.
    """

    markers_x: int
    markers_y: int
    marker_length_metres: float
    separation_metres: float
    dictionary: str
    first_id: int = 0
    ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.markers_x < 1 or self.markers_y < 1 or self.markers_x * self.markers_y < 2:
            raise ValueError("an ArUco grid needs at least two markers")
        if self.marker_length_metres <= 0 or self.separation_metres <= 0:
            raise ValueError("marker length and separation must be positive")
        if not hasattr(cv2.aruco, self.dictionary):
            raise ValueError(f"unknown OpenCV ArUco dictionary {self.dictionary!r}")
        if self.ids is not None and len(self.ids) != self.markers_x * self.markers_y:
            raise ValueError("ids must list every marker of the grid, row by row")

    @property
    def marker_ids(self) -> tuple[int, ...]:
        count = self.markers_x * self.markers_y
        if self.ids is not None:
            return self.ids
        return tuple(range(self.first_id, self.first_id + count))

    def create_board(self) -> Any:
        dictionary = cv2.aruco.getPredefinedDictionary(int(getattr(cv2.aruco, self.dictionary)))
        return cv2.aruco.GridBoard(
            (self.markers_x, self.markers_y),
            self.marker_length_metres,
            self.separation_metres,
            dictionary,
            np.asarray(self.marker_ids, dtype=np.int32),
        )


def _aruco_detector(dictionary_name: str) -> Any:
    dictionary = cv2.aruco.getPredefinedDictionary(int(getattr(cv2.aruco, dictionary_name)))
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(dictionary, parameters)


class CharucoPattern:
    """ChArUco board: identified chessboard corners, the most precise option."""

    kind = "charuco"

    def __init__(self, spec: CharucoBoardSpec) -> None:
        self.spec = spec
        self.target = CharucoTarget(spec)
        self.object_points = np.asarray(self.target.board.getChessboardCorners(), dtype=np.float64)

    def detect(
        self, gray: NDArray[np.uint8]
    ) -> tuple[NDArray[np.int32], NDArray[np.float64]] | None:
        observation = self.target.detect(gray)
        if observation is None:
            return None
        return (
            observation.ids.reshape(-1).astype(np.int32),
            observation.corners.reshape(-1, 2).astype(np.float64),
        )

    def record(self) -> dict[str, Any]:
        spec = self.spec
        return {
            "pattern": self.kind,
            "squares_x": spec.squares_x,
            "squares_y": spec.squares_y,
            "square_length_metres": spec.square_length_metres,
            "marker_length_metres": spec.marker_length_metres,
            "dictionary": spec.dictionary,
            "legacy_pattern": spec.legacy_pattern,
        }

    def describe(self) -> str:
        spec = self.spec
        return (
            f"ChArUco {spec.squares_x}x{spec.squares_y}, square "
            f"{spec.square_length_metres * 1e3:.1f} mm, marker "
            f"{spec.marker_length_metres * 1e3:.1f} mm, {spec.dictionary}"
        )


class ArucoGridPattern:
    """Plain ArUco marker grid: four corners per marker, corner id = marker index * 4 + k.

    Marker corners are refined to sub-pixel but are still less precise than chessboard
    corners; expect a somewhat larger reprojection RMS than with a ChArUco board.
    """

    kind = "aruco_grid"

    def __init__(self, spec: ArucoGridSpec) -> None:
        self.spec = spec
        self.board = spec.create_board()
        self.detector = _aruco_detector(spec.dictionary)
        self._marker_index = {int(marker_id): k for k, marker_id in enumerate(spec.marker_ids)}
        self.object_points = np.concatenate(
            [
                np.asarray(points, dtype=np.float64).reshape(4, 3)
                for points in self.board.getObjPoints()
            ]
        )

    def detect(
        self, gray: NDArray[np.uint8]
    ) -> tuple[NDArray[np.int32], NDArray[np.float64]] | None:
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            return None
        corner_ids: list[int] = []
        pixels: list[NDArray[np.float64]] = []
        for marker_corners, marker_id in zip(corners, ids.reshape(-1), strict=True):
            index = self._marker_index.get(int(marker_id))
            if index is None:
                continue
            quad = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
            corner_ids.extend(index * 4 + k for k in range(4))
            pixels.extend(quad)
        if not corner_ids:
            return None
        return np.asarray(corner_ids, dtype=np.int32), np.asarray(pixels, dtype=np.float64)

    def record(self) -> dict[str, Any]:
        spec = self.spec
        return {
            "pattern": self.kind,
            "markers_x": spec.markers_x,
            "markers_y": spec.markers_y,
            "marker_length_metres": spec.marker_length_metres,
            "separation_metres": spec.separation_metres,
            "dictionary": spec.dictionary,
            "ids": list(spec.marker_ids),
        }

    def describe(self) -> str:
        spec = self.spec
        return (
            f"ArUco grid {spec.markers_x}x{spec.markers_y}, marker "
            f"{spec.marker_length_metres * 1e3:.1f} mm, gap {spec.separation_metres * 1e3:.1f} mm, "
            f"{spec.dictionary}, ids {list(spec.marker_ids)}"
        )


Pattern = CharucoPattern | ArucoGridPattern


class PatternDetector:
    """Detect the pattern, undistort its corners, and solve its pose in the camera frame."""

    def __init__(
        self, pattern: Pattern, camera_matrix: NDArray[np.float64], min_corners: int
    ) -> None:
        self.pattern = pattern
        self.object_points = pattern.object_points
        self.camera_matrix = camera_matrix
        self.min_corners = min_corners

    def detect(self, image: NDArray[np.uint8], undistort: Any) -> tuple[Detection | None, str]:
        gray = cast(NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        found = self.pattern.detect(gray)
        if found is None:
            return None, f"{self.pattern.kind} pattern not detected"
        ids, raw = found
        if ids.size < self.min_corners:
            return None, f"only {ids.size} corners (need {self.min_corners})"
        undistorted = undistort(raw)
        board = board_observation_from_points(
            self.object_points[ids],
            undistorted,
            self.camera_matrix,
            _ZERO_DISTORTION,
            minimum_corners=self.min_corners,
        )
        if board is None:
            return None, "PnP failed (collinear corners?)"
        return (
            Detection(ids, raw, undistorted, board, board_view_angle_deg(board.camera_from_target)),
            "",
        )

    def observation_from_points(
        self, ids: NDArray[np.int32], undistorted: NDArray[np.float64]
    ) -> BoardObservation | None:
        return board_observation_from_points(
            self.object_points[ids],
            undistorted,
            self.camera_matrix,
            _ZERO_DISTORTION,
            minimum_corners=self.min_corners,
        )


def charuco_spec_from_args(rig: CpfRig, args: argparse.Namespace) -> CharucoBoardSpec:
    base = rig.rig.board
    squares = (
        tuple(args.squares) if getattr(args, "squares", None) else (base.squares_x, base.squares_y)
    )
    square = (
        args.square_mm / 1000.0 if getattr(args, "square_mm", None) else base.square_length_metres
    )
    marker = (
        args.marker_mm / 1000.0 if getattr(args, "marker_mm", None) else base.marker_length_metres
    )
    if getattr(args, "square_mm", None) and not getattr(args, "marker_mm", None):
        # Keep the printed marker/square ratio when only the square was measured.
        marker = square * base.marker_length_metres / base.square_length_metres
    return CharucoBoardSpec(
        squares_x=int(squares[0]),
        squares_y=int(squares[1]),
        square_length_metres=float(square),
        marker_length_metres=float(marker),
        dictionary=str(getattr(args, "dictionary", None) or base.dictionary),
        legacy_pattern=bool(getattr(args, "legacy_pattern", False)),
    )


def pattern_from_args(rig: CpfRig, args: argparse.Namespace) -> Pattern:
    """The pattern the operator has: the rig's ChArUco board by default, or a marker grid."""

    if getattr(args, "pattern", "charuco") != "aruco-grid":
        return CharucoPattern(charuco_spec_from_args(rig, args))
    required = (
        ("--marker-mm", getattr(args, "marker_mm", None)),
        ("--gap-mm", getattr(args, "gap_mm", None)),
        ("--dictionary", getattr(args, "dictionary", None)),
    )
    missing = [name for name, value in required if not value]
    if missing:
        raise ValueError(
            "an ArUco grid needs " + ", ".join(missing) + "; run `identify` for the dictionary "
            "and ids, and measure the marker edge and the gap on the print"
        )
    markers = tuple(args.markers) if getattr(args, "markers", None) else (3, 2)
    ids = tuple(int(value) for value in args.ids) if getattr(args, "ids", None) else None
    return ArucoGridPattern(
        ArucoGridSpec(
            markers_x=int(markers[0]),
            markers_y=int(markers[1]),
            marker_length_metres=float(args.marker_mm) / 1000.0,
            separation_metres=float(args.gap_mm) / 1000.0,
            dictionary=str(args.dictionary),
            first_id=int(getattr(args, "first_id", 0) or 0),
            ids=ids,
        )
    )


def pattern_from_record(record: dict[str, Any]) -> Pattern:
    if record.get("pattern", "charuco") == "aruco_grid":
        ids = record.get("ids")
        return ArucoGridPattern(
            ArucoGridSpec(
                markers_x=int(record["markers_x"]),
                markers_y=int(record["markers_y"]),
                marker_length_metres=float(record["marker_length_metres"]),
                separation_metres=float(record["separation_metres"]),
                dictionary=str(record["dictionary"]),
                ids=tuple(int(value) for value in ids) if ids else None,
            )
        )
    return CharucoPattern(
        CharucoBoardSpec(
            squares_x=int(record["squares_x"]),
            squares_y=int(record["squares_y"]),
            square_length_metres=float(record["square_length_metres"]),
            marker_length_metres=float(record["marker_length_metres"]),
            dictionary=str(record["dictionary"]),
            legacy_pattern=bool(record.get("legacy_pattern", False)),
        )
    )


# Smallest dictionaries first, so a tie on marker count picks the most likely family.
_IDENTIFY_DICTIONARIES = (
    "DICT_4X4_50",
    "DICT_4X4_100",
    "DICT_4X4_250",
    "DICT_4X4_1000",
    "DICT_5X5_50",
    "DICT_5X5_100",
    "DICT_5X5_250",
    "DICT_5X5_1000",
    "DICT_6X6_50",
    "DICT_6X6_100",
    "DICT_6X6_250",
    "DICT_6X6_1000",
    "DICT_7X7_50",
    "DICT_7X7_100",
    "DICT_7X7_250",
    "DICT_7X7_1000",
    "DICT_ARUCO_ORIGINAL",
    "DICT_ARUCO_MIP_36h12",
    "DICT_APRILTAG_16h5",
    "DICT_APRILTAG_25h9",
    "DICT_APRILTAG_36h10",
    "DICT_APRILTAG_36h11",
)


def _unit(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    return vector / max(float(np.linalg.norm(vector)), 1e-9)


def identify_markers(grays: list[NDArray[np.uint8]]) -> dict[str, Any]:
    """Which dictionary decodes the most markers, their ids laid out in rows, and sizes.

    Markers must appear in at least half of the frames. Rows and columns are taken in the
    markers' own frame (corner 0 to corner 1 is "right"), so the result is the print's
    row-major layout however the board is held, which is what ``GridBoard`` assumes. Rows
    are split where the "down" coordinate jumps by more than half the smallest centre
    spacing, which tolerates perspective; ``regular`` is false when the rows come out
    ragged (oblique view or a missed marker). The gap-to-marker ratio comes from the
    centre spacing of neighbours; it is a check for the ruler, not a substitute.
    """

    results: list[tuple[str, dict[int, NDArray[np.float64]]]] = []
    for name in _IDENTIFY_DICTIONARIES:
        detector = _aruco_detector(name)
        seen: dict[int, list[NDArray[np.float64]]] = {}
        for gray in grays:
            corners, ids, _ = detector.detectMarkers(gray)
            if ids is None:
                continue
            for marker_corners, marker_id in zip(corners, ids.reshape(-1), strict=True):
                quad = np.asarray(marker_corners, dtype=np.float64).reshape(4, 2)
                seen.setdefault(int(marker_id), []).append(quad)
        needed = max(1, len(grays) // 2)
        stable = {
            marker_id: np.mean(np.stack(quads), axis=0)
            for marker_id, quads in seen.items()
            if len(quads) >= needed
        }
        results.append((name, stable))
    name, stable = max(results, key=lambda item: len(item[1]))
    if not stable:
        return {"found": False}
    quads = list(stable.values())
    right = _unit(np.median(np.stack([_unit(q[1] - q[0]) for q in quads]), axis=0))
    down = _unit(np.median(np.stack([_unit(q[3] - q[0]) for q in quads]), axis=0))
    side = float(
        np.median(
            [np.mean([np.linalg.norm(q[k] - q[(k + 1) % 4]) for k in range(4)]) for q in quads]
        )
    )
    centres = {marker_id: quad.mean(axis=0) for marker_id, quad in stable.items()}
    along = {marker_id: float(centre @ right) for marker_id, centre in centres.items()}
    across = {marker_id: float(centre @ down) for marker_id, centre in centres.items()}
    marker_ids = list(centres)
    pitch = min(
        (
            float(np.linalg.norm(centres[a] - centres[b]))
            for i, a in enumerate(marker_ids)
            for b in marker_ids[i + 1 :]
        ),
        default=side,
    )
    rows: list[list[int]] = []
    previous: int | None = None
    for marker_id in sorted(marker_ids, key=lambda i: across[i]):
        if previous is not None and across[marker_id] - across[previous] < 0.5 * pitch:
            rows[-1].append(marker_id)
        else:
            rows.append([marker_id])
        previous = marker_id
    rows = [sorted(row, key=lambda i: along[i]) for row in rows]
    columns = max(len(row) for row in rows)
    spacings = [
        float(np.linalg.norm(centres[row[k + 1]] - centres[row[k]]))
        for row in rows
        for k in range(len(row) - 1)
    ]
    ratio = float(np.median(spacings)) / side - 1.0 if spacings else None
    return {
        "found": True,
        "dictionary": name,
        "also_decodes": [n for n, s in results if len(s) == len(stable) and n != name],
        "markers": len(stable),
        "ids_by_row": rows,
        "columns": columns,
        "rows": len(rows),
        "regular": all(len(row) == columns for row in rows),
        "marker_side_px": side,
        "gap_over_marker": ratio,
        "print_right_in_image_deg": math.degrees(math.atan2(float(right[1]), float(right[0]))),
        "corners": stable,
    }


# ---------------------------------------------------------------------------------------
# Capture bookkeeping


@dataclass(slots=True)
class Stillness:
    """True once every joint speed has stayed under ``threshold`` for ``settle_s``."""

    threshold_rad_s: float
    settle_s: float
    _still_since: float | None = None

    def update(self, dq: NDArray[np.float64], now: float) -> tuple[bool, float]:
        if float(np.max(np.abs(dq))) > self.threshold_rad_s:
            self._still_since = None
            return False, 0.0
        if self._still_since is None:
            self._still_since = now
        held = now - self._still_since
        return held >= self.settle_s, held


@dataclass(slots=True)
class PoseAccumulator:
    """Frames of one static pose, averaged into a single sample."""

    frames: list[tuple[Detection, ArmState, NDArray[np.uint8]]] = field(default_factory=list)
    started: float | None = None

    def reset(self) -> None:
        self.frames.clear()
        self.started = None

    def add(
        self, detection: Detection, state: ArmState, image: NDArray[np.uint8], now: float
    ) -> None:
        if self.started is None:
            self.started = now
        self.frames.append((detection, state, image))

    def ready(self, frames_needed: int, now: float, min_seconds: float) -> bool:
        return (
            self.started is not None
            and len(self.frames) >= frames_needed
            and now - self.started >= min_seconds
        )


@dataclass(frozen=True, slots=True)
class CapturedPose:
    sample_id: str
    captured_at: str
    q: NDArray[np.float64]
    q_std_max: float
    dq_max: float
    hand_shm: NDArray[np.float64]
    world_from_hand: Matrix
    ids: NDArray[np.int32]
    corners_px: NDArray[np.float64]
    corners_undistorted: NDArray[np.float64]
    observation: BoardObservation
    view_angle_deg: float
    frames_averaged: int
    image: NDArray[np.uint8]

    def as_sample(self) -> HandEyeSample:
        return HandEyeSample(self.sample_id, self.world_from_hand, self.observation)

    def record(self, image_name: str) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "captured_at": self.captured_at,
            "q": self.q.tolist(),
            "q_std_max": self.q_std_max,
            "dq_max": self.dq_max,
            "hand_shm": self.hand_shm.tolist(),
            "hand_fk": self.world_from_hand[:3, 3].tolist(),
            "world_from_hand": self.world_from_hand.tolist(),
            "board": {
                "ids": self.ids.tolist(),
                "corners_px": self.corners_px.tolist(),
                "corners_undistorted_px": self.corners_undistorted.tolist(),
                "camera_from_target": self.observation.camera_from_target.tolist(),
                "pnp_rms_px": self.observation.reprojection_rms_pixels,
                "view_angle_deg": self.view_angle_deg,
                "frames_averaged": self.frames_averaged,
            },
            "image": image_name,
        }


def average_pose(
    accumulator: PoseAccumulator,
    detector: PatternDetector,
    fk: MujocoForwardKinematics,
    sample_id: str,
) -> CapturedPose | None:
    frames = accumulator.frames
    common = set(frames[0][0].ids.tolist())
    for detection, _, _ in frames[1:]:
        common &= set(detection.ids.tolist())
    ids = np.asarray(sorted(common), dtype=np.int32)
    if ids.size < detector.min_corners:
        return None
    raw = np.zeros((ids.size, 2), dtype=np.float64)
    undistorted = np.zeros((ids.size, 2), dtype=np.float64)
    for detection, _, _ in frames:
        order = {int(value): index for index, value in enumerate(detection.ids.tolist())}
        rows = [order[int(value)] for value in ids]
        raw = raw + np.asarray(detection.corners_px[rows], dtype=np.float64)
        undistorted = undistorted + np.asarray(
            detection.corners_undistorted[rows], dtype=np.float64
        )
    raw = raw / len(frames)
    undistorted = undistorted / len(frames)
    observation = detector.observation_from_points(ids, undistorted)
    if observation is None:
        return None
    q = np.stack([state.q for _, state, _ in frames])
    dq_max = max(float(np.max(np.abs(state.dq))) for _, state, _ in frames)
    q_mean = q.mean(axis=0)
    middle = frames[len(frames) // 2]
    return CapturedPose(
        sample_id=sample_id,
        captured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        q=q_mean,
        q_std_max=float(q.std(axis=0).max()),
        dq_max=dq_max,
        hand_shm=middle[1].hand,
        world_from_hand=fk.world_from_body(q_mean),
        ids=ids,
        corners_px=raw,
        corners_undistorted=undistorted,
        observation=observation,
        view_angle_deg=board_view_angle_deg(observation.camera_from_target),
        frames_averaged=len(frames),
        image=middle[2],
    )


def nearest_rotation_deg(captured: list[CapturedPose], rotation: NDArray[np.float64]) -> float:
    return min(
        (rotation_angle_deg(pose.world_from_hand[:3, :3].T @ rotation) for pose in captured),
        default=180.0,
    )


def rotation_hint(captured: list[CapturedPose]) -> str:
    """Which world axis the captured hand rotations have exercised least."""

    if len(captured) < 2:
        return "rotate the hand about two different axes between poses (20-40 deg)"
    scatter = np.zeros((3, 3))
    rotations = [pose.world_from_hand[:3, :3] for pose in captured]
    for i, first in enumerate(rotations):
        for second in rotations[i + 1 :]:
            rvec, _ = cv2.Rodrigues(second @ first.T)  # relative rotation, world axes
            angle = float(np.linalg.norm(rvec))
            if math.degrees(angle) >= 2.0:
                axis = np.asarray(rvec, dtype=np.float64).reshape(3) / angle
                scatter = scatter + np.outer(axis, axis).astype(np.float64)
    values, vectors = np.linalg.eigh(scatter)
    if values[-1] <= 0:
        return "rotate the hand between poses"
    weakest = vectors[:, 0]
    names = (
        "x (tilt the board sideways)",
        "y (nod the board up/down)",
        "z (turn the board left/right)",
    )
    axis_name = names[int(np.argmax(np.abs(weakest)))]
    if values[1] / values[-1] < 0.35:
        return f"least covered: rotation about world {axis_name}"
    return "rotation coverage is good; add poses across the grasp volume"


# ---------------------------------------------------------------------------------------
# Drawing


def project_world_points(
    points: NDArray[np.float64], camera_from_world: Matrix, camera_matrix: NDArray[np.float64]
) -> NDArray[np.float64] | None:
    camera = points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    if np.any(camera[:, 2] <= 0.05):
        return None
    u = camera_matrix[0, 0] * camera[:, 0] / camera[:, 2] + camera_matrix[0, 2]
    v = camera_matrix[1, 1] * camera[:, 1] / camera[:, 2] + camera_matrix[1, 2]
    return np.stack((u, v), axis=1)


def draw_skeleton(
    image: NDArray[np.uint8],
    fk: MujocoForwardKinematics,
    q: NDArray[np.float64],
    arm: str,
    camera_from_world: Matrix,
    camera_matrix: NDArray[np.float64],
    colour: tuple[int, int, int] = _CYAN,
) -> None:
    points = fk.body_positions(q, arm_skeleton_bodies(arm))
    pixels = project_world_points(points, camera_from_world, camera_matrix)
    if pixels is None:
        return
    pts = np.rint(pixels).astype(np.int32)
    for a, b in zip(pts[:-1], pts[1:], strict=True):
        cv2.line(image, tuple(a), tuple(b), colour, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(image, tuple(p), 4, colour, -1, cv2.LINE_AA)
    cv2.putText(
        image,
        f"{arm} arm (FK)",
        tuple(pts[-1] + (6, -6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        colour,
        1,
        cv2.LINE_AA,
    )


def draw_detection(
    image: NDArray[np.uint8],
    detection: Detection,
    camera_matrix: NDArray[np.float64],
    colour: tuple[int, int, int],
) -> None:
    for u, v in np.rint(detection.corners_px).astype(np.int32):
        cv2.circle(image, (int(u), int(v)), 3, colour, -1, cv2.LINE_AA)
    # Board axes (x red, y green, z blue), drawn by hand: cv2.drawFrameAxes warns on every
    # frame whose axis endpoint leaves the image.
    axes = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.0, 0.05]])
    pixels = project_world_points(axes, detection.observation.camera_from_target, camera_matrix)
    if pixels is None:
        return
    pts = np.rint(pixels).astype(np.int32)
    for k, axis_colour in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        cv2.line(image, tuple(pts[0]), tuple(pts[k + 1]), axis_colour, 2, cv2.LINE_AA)


def draw_panel(image: NDArray[np.uint8], lines: list[tuple[str, tuple[int, int, int]]]) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 22 * len(lines) + 8), (0, 0, 0), -1)
    for row, (text, colour) in enumerate(lines):
        cv2.putText(
            image, text, (8, 22 * (row + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA
        )


# ---------------------------------------------------------------------------------------
# Session storage and solving


class CaptureSession:
    def __init__(self, directory: Path, header: dict[str, Any]) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "session.json").write_text(json.dumps(header, indent=2))
        self.poses: list[CapturedPose] = []

    def add(self, pose: CapturedPose) -> None:
        image_name = f"{pose.sample_id}.png"
        cv2.imwrite(str(self.directory / image_name), pose.image)
        (self.directory / f"{pose.sample_id}.json").write_text(
            json.dumps(pose.record(image_name), indent=2)
        )
        self.poses.append(pose)

    def undo(self) -> CapturedPose | None:
        if not self.poses:
            return None
        pose = self.poses.pop()
        for suffix in (".png", ".json"):
            (self.directory / f"{pose.sample_id}{suffix}").unlink(missing_ok=True)
        return pose

    def next_id(self) -> str:
        return f"pose-{len(self.poses):02d}"


def load_session(
    directory: Path,
) -> tuple[dict[str, Any], list[HandEyeSample], list[dict[str, Any]]]:
    header = json.loads((directory / "session.json").read_text())
    camera_matrix = np.asarray(header["camera"]["camera_matrix"], dtype=np.float64)
    pattern = pattern_from_record(header["board"])
    detector = PatternDetector(pattern, camera_matrix, int(header["capture"]["min_corners"]))
    samples: list[HandEyeSample] = []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("pose-*.json")):
        record = json.loads(path.read_text())
        ids = np.asarray(record["board"]["ids"], dtype=np.int32)
        undistorted = np.asarray(record["board"]["corners_undistorted_px"], dtype=np.float64)
        observation = detector.observation_from_points(ids, undistorted)
        if observation is None:
            print(f"skipping {path.name}: PnP failed on stored corners")
            continue
        samples.append(
            HandEyeSample(
                str(record["sample_id"]),
                np.asarray(record["world_from_hand"], dtype=np.float64),
                observation,
            )
        )
        records.append(record)
    return header, samples, records


def _validation_record(validation: HandEyeValidation | None) -> dict[str, Any] | None:
    if validation is None:
        return None
    return {key: getattr(validation, key) for key in validation.__dataclass_fields__}


def solve_samples(
    samples: list[HandEyeSample],
    camera_matrix: NDArray[np.float64],
    *,
    holdout_every: int,
    minimum_samples: int,
) -> tuple[HandEyeCalibration, HandEyeValidation | None, HandEyeCalibration]:
    """Solve on a training split, validate on the rest, then solve on everything."""

    if len(samples) >= max(12, minimum_samples + 3) and holdout_every > 1:
        held = [s for i, s in enumerate(samples) if i % holdout_every == holdout_every - 1]
        train = [s for i, s in enumerate(samples) if i % holdout_every != holdout_every - 1]
    else:
        held, train = [], list(samples)
    partial = calibrate_hand_eye(
        train,
        HandEyeMount.EYE_TO_HAND,
        camera_matrix,
        _ZERO_DISTORTION,
        minimum_samples=minimum_samples,
    )
    validation = validate_hand_eye(partial, held, camera_matrix, _ZERO_DISTORTION) if held else None
    final = (
        calibrate_hand_eye(
            samples,
            HandEyeMount.EYE_TO_HAND,
            camera_matrix,
            _ZERO_DISTORTION,
            minimum_samples=minimum_samples,
        )
        if held
        else partial
    )
    return partial, validation, final


def build_result_record(
    header: dict[str, Any],
    final: HandEyeCalibration,
    validation: HandEyeValidation | None,
    session_dir: Path,
) -> dict[str, Any]:
    x = final.camera_to_robot
    record: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "method": (
            "ChArUco eye-to-hand hand-eye: OpenCV closed form (best of Tsai/Park/Horaud/"
            "Andreff/Daniilidis) + joint reprojection refinement of camera and board offset; "
            "FK from CPF's MuJoCo model on shared-memory joint state"
        ),
        "world_frame": header["world_frame"],
        "camera_frame": header["camera_frame"],
        "camera_serial": header["camera"]["serial"],
        "arm": header["arm"],
        "flange_body": header["flange_body"],
        "measured_on": datetime.now(UTC).date().isoformat(),
        "session_dir": str(session_dir),
        "color_resolution": [header["camera"]["width"], header["camera"]["height"]],
        "camera_matrix": header["camera"]["camera_matrix"],
        "board": header["board"],
        "world_from_camera": x.tolist(),
        "camera_from_world": invert_transform(x).tolist(),
        "camera_position_world_metres": x[:3, 3].tolist(),
        **camera_orientation_summary(x),
        "flange_from_target": final.target_offset.tolist(),
        "samples": len(final.sample_ids),
        "sample_ids": list(final.sample_ids),
        "rejected_sample_ids": list(final.rejected_sample_ids),
        "reprojection_rms_pixels": final.reprojection_rms_pixels,
        "per_sample_rms_pixels": dict(final.per_sample_rms_pixels),
        "closed_form_deviation_mm_deg": {
            k: list(v) for k, v in final.closed_form_deviation.items()
        },
        "diversity": {k: getattr(final.diversity, k) for k in final.diversity.__dataclass_fields__},
        "validation": _validation_record(validation),
        "comparisons": {},
    }
    comparisons: dict[str, Any] = {}
    if TAPE_EXTRINSIC.is_file():
        tape = np.asarray(
            json.loads(TAPE_EXTRINSIC.read_text())["world_from_camera"], dtype=np.float64
        )
        mm, deg = transform_difference(x, tape)
        comparisons["tape_imu_extrinsic"] = {
            "file": str(TAPE_EXTRINSIC),
            "translation_mm": mm,
            "rotation_deg": deg,
        }
    other = "left" if header["arm"] == "right" else "right"
    other_file = CALIBRATION_DIR / f"current_{other}.json"
    if other_file.is_file():
        other_x = np.asarray(
            json.loads(other_file.read_text())["world_from_camera"], dtype=np.float64
        )
        mm, deg = transform_difference(x, other_x)
        comparisons[f"{other}_arm_calibration"] = {
            "file": str(other_file),
            "translation_mm": mm,
            "rotation_deg": deg,
        }
    record["comparisons"] = comparisons
    return record


def print_result(record: dict[str, Any], validation: HandEyeValidation | None) -> None:
    pos = np.round(record["camera_position_world_metres"], 4).tolist()
    print("\n=== hand-eye result ===")
    print(f"camera position in {record['world_frame']}: {pos} m")
    print(
        f"heading {record['heading_deg']:.2f} deg, pitch down "
        f"{record['camera_pitch_down_deg']:.2f} deg, "
        f"image roll {record['camera_roll_deg']:.2f} deg"
    )
    print(
        f"samples used {record['samples']} (rejected {record['rejected_sample_ids']}), "
        f"reprojection RMS {record['reprojection_rms_pixels']:.2f} px"
    )
    d = record["diversity"]
    print(
        f"motion diversity: max relative rotation {d['max_relative_rotation_deg']:.1f} deg, "
        f"median {d['median_relative_rotation_deg']:.1f} deg, axis spread {d['axis_spread']:.2f}"
    )
    worst = max(record["closed_form_deviation_mm_deg"].items(), key=lambda kv: kv[1][0])
    print(
        "closed-form methods vs refined: worst "
        f"{worst[0]} {worst[1][0]:.1f} mm / {worst[1][1]:.2f} deg"
    )
    if validation is not None:
        print(
            f"held-out ({validation.samples} poses): reprojection "
            f"{validation.reprojection_rms_pixels:.2f} px, "
            f"corner error mean {validation.corner_error_mean_mm:.1f} mm, "
            f"p95 {validation.corner_error_p95_mm:.1f} mm, "
            f"rotation mean {validation.rotation_error_mean_deg:.2f} deg"
        )
    else:
        print("held-out validation: skipped (fewer than 12 samples)")
    for name, comparison in record["comparisons"].items():
        print(
            f"vs {name}: {comparison['translation_mm']:.1f} mm, "
            f"{comparison['rotation_deg']:.2f} deg"
        )
    per_sample = record["per_sample_rms_pixels"]
    print("per-sample RMS px: " + ", ".join(f"{k[-2:]}:{v:.2f}" for k, v in per_sample.items()))


def solve_and_save(
    session_dir: Path, *, holdout_every: int, minimum_samples: int
) -> dict[str, Any] | None:
    header, samples, _ = load_session(session_dir)
    camera_matrix = np.asarray(header["camera"]["camera_matrix"], dtype=np.float64)
    try:
        _, validation, final = solve_samples(
            samples, camera_matrix, holdout_every=holdout_every, minimum_samples=minimum_samples
        )
    except HandEyeDegenerateError as error:
        print(f"cannot solve yet: {error}")
        return None
    record = build_result_record(header, final, validation, session_dir)
    print_result(record, validation)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_hand_eye_result(
        CALIBRATION_DIR / f"hand_eye_{header['arm']}_{stamp}.json", record, make_current=True
    )
    (CALIBRATION_DIR / f"current_{header['arm']}.json").write_text(json.dumps(record, indent=2))
    (session_dir / "result.json").write_text(json.dumps(record, indent=2))
    print(f"saved {path}\nsaved {CURRENT_CALIBRATION} (used by cpf_box_handoff by default)")
    return record


# ---------------------------------------------------------------------------------------
# Subcommands


def run_board(args: argparse.Namespace) -> int:
    rig = load_cpf_rig(args.rig)
    pattern = pattern_from_args(rig, args)
    px_per_mm = 10  # 10000 px/m -> exactly 254 dpi, so every length is a whole pixel count
    margin_px = round(args.margin_mm * px_per_mm)
    if isinstance(pattern, CharucoPattern):
        spec = pattern.spec
        board_mm = (
            spec.squares_x * spec.square_length_metres * 1000,
            spec.squares_y * spec.square_length_metres * 1000,
        )
        measure = "MEASURE one square across several squares (e.g. 6 squares / 6) -> --square-mm"
    else:
        grid = pattern.spec
        pitch = (grid.marker_length_metres + grid.separation_metres) * 1000
        board_mm = (
            (grid.markers_x - 1) * pitch + grid.marker_length_metres * 1000,
            (grid.markers_y - 1) * pitch + grid.marker_length_metres * 1000,
        )
        measure = "MEASURE a marker's black edge -> --marker-mm and the white gap -> --gap-mm"
    width = round(board_mm[0] * px_per_mm) + 2 * margin_px
    height = round(board_mm[1] * px_per_mm) + 2 * margin_px
    board = pattern.spec.create_board()
    image = board.generateImage((width, height), marginSize=margin_px, borderBits=1)
    pil: Any = importlib.import_module("PIL.Image")
    dpi = px_per_mm * 25.4
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pil.fromarray(image).save(f"{out}.png", dpi=(dpi, dpi))
    pil.fromarray(image).save(f"{out}.pdf", resolution=dpi)
    page_mm = (width / px_per_mm, height / px_per_mm)
    print(f"wrote {out}.png and {out}.pdf at {dpi:.0f} dpi")
    print(
        f"{pattern.describe()}; pattern {board_mm[0]:.0f}x{board_mm[1]:.0f} mm "
        f"on a {page_mm[0]:.0f}x{page_mm[1]:.0f} mm page"
    )
    print(f"print at 100 % / actual size, glue flat to a stiff board, then {measure}")
    return 0


def run_identify(args: argparse.Namespace) -> int:
    """Work out which ArUco/AprilTag dictionary a printed pattern uses, and its layout."""

    grays: list[NDArray[np.uint8]] = []
    if args.image:
        loaded = cv2.imread(args.image, cv2.IMREAD_COLOR)
        if loaded is None:
            raise ValueError(f"could not read {args.image}")
        annotated = cast(NDArray[np.uint8], loaded)
        grays.append(cast(NDArray[np.uint8], cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)))
    else:
        rig = load_cpf_rig(args.rig)
        width, height = int(args.color[0]), int(args.color[1])
        fps = int(args.color[2]) if len(args.color) > 2 else None
        camera = ColorCamera(rig.rig.camera_serial, width, height, fps)
        print(f"hold the pattern still in view for {args.seconds:.0f} s ...", flush=True)
        try:
            deadline = time.monotonic() + args.seconds
            annotated = camera.read()
            while time.monotonic() < deadline:
                annotated = camera.read()
                grays.append(cast(NDArray[np.uint8], cv2.cvtColor(annotated, cv2.COLOR_BGR2GRAY)))
        finally:
            camera.close()
        grays = grays[-20:]
    report = identify_markers(grays)
    if not report["found"]:
        print("no ArUco/AprilTag markers found with any predefined dictionary")
        return 1
    for row in report["ids_by_row"]:
        for marker_id in row:
            quad = np.rint(report["corners"][marker_id]).astype(np.int32)
            cv2.polylines(annotated, [quad.reshape(-1, 1, 2)], True, _GREEN, 2, cv2.LINE_AA)
            cv2.circle(annotated, (int(quad[0][0]), int(quad[0][1])), 6, _RED, -1, cv2.LINE_AA)
            cv2.putText(
                annotated, str(marker_id), (int(quad[0][0]) + 8, int(quad[0][1]) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, _YELLOW, 2, cv2.LINE_AA,
            )  # fmt: skip
    out = CALIBRATION_DIR / f"identify_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), annotated)
    rows = report["ids_by_row"]
    ids_row_major = [marker_id for row in rows for marker_id in row]
    print(f"dictionary: {report['dictionary']}  (also decodes: {report['also_decodes'] or 'none'})")
    print(
        f"markers: {report['markers']} in {report['rows']} row(s) x {report['columns']} "
        "column(s), in the print's own frame:"
    )
    for row in rows:
        print("  row:", row)
    if not report["regular"]:
        print(
            "WARNING: rows are ragged (oblique view or a missed marker); hold the board flat, "
            "facing the camera, and rerun before trusting the layout below"
        )
    print(
        f"the print's right-hand direction points at {report['print_right_in_image_deg']:.0f} deg "
        "in the image (0 = image right, 90 = image down)"
    )
    ratio = report["gap_over_marker"]
    print(
        f"marker edge ~{report['marker_side_px']:.0f} px in the image"
        + (
            f"; gap / marker edge ~{ratio:.2f} from centre spacing (check with the ruler)"
            if ratio is not None
            else ""
        )
    )
    print(f"annotated image (red dot = each marker's corner 0): {out}")
    first = ids_row_major[0]
    consecutive = ids_row_major == list(range(first, first + len(ids_row_major)))
    ids_arg = f"--first-id {first}" if consecutive else "--ids " + " ".join(map(str, ids_row_major))
    print(
        "\nmeasure on the print: --marker-mm = outer edge of one black square, "
        "--gap-mm = white space between two neighbouring black squares"
    )
    print(
        "if this is a plain marker grid, capture with:\n"
        "  python -m vision_pipeline.apps.openarm_hand_eye capture --arm right "
        f"--pattern aruco-grid --markers {report['columns']} {report['rows']} "
        f"--dictionary {report['dictionary']} {ids_arg} \\\n"
        "      --marker-mm <measured> --gap-mm <measured>"
    )
    print(
        "if the markers sit inside a chessboard it is a ChArUco board: use --pattern charuco "
        "with --squares, --square-mm and this dictionary instead"
    )
    return 0


def _capture_header(
    rig: CpfRig, arm: CpfArm, camera: ColorCamera, pattern: Pattern, args: argparse.Namespace
) -> dict[str, Any]:
    return {
        "schema": "openarm-cpf-hand-eye-session-v1",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "arm": arm.name,
        "flange_body": arm.mujoco_body,
        "joint_names": list(arm.profile.joint_names),
        "world_frame": rig.rig.reference_frame,
        "camera_frame": rig.rig.camera_frame,
        "model_xml": str(rig.model_xml),
        "rig": str(args.rig),
        "camera": camera.info(),
        "board": pattern.record(),
        "capture": {
            "min_corners": args.min_corners,
            "min_rotation_deg": args.min_rotation_deg,
            "settle_s": args.settle_s,
            "dq_still_rad_s": args.dq_still,
            "frames_per_pose": args.frames_per_pose,
            "max_view_angle_deg": args.max_view_angle,
            "max_pnp_rms_px": args.max_pnp_rms,
        },
    }


def run_capture(args: argparse.Namespace) -> int:
    rig = load_cpf_rig(args.rig)
    arm = rig.arm(args.arm)
    pattern = pattern_from_args(rig, args)
    print(f"pattern: {pattern.describe()}")
    if isinstance(pattern, CharucoPattern) and not args.square_mm:
        print(
            "WARNING: using the rig's nominal square size "
            f"{pattern.spec.square_length_metres * 1000:.1f} mm; "
            "pass the measured --square-mm for a real run"
        )
    fk = MujocoForwardKinematics(rig.model_xml, arm.profile.joint_names, arm.mujoco_body)
    shm = SharedMemoryArms(rig.hw_dir, rig.shm_name)
    if shm.created:
        print(
            f"created shared memory {rig.shm_name}. Now start the driver in another terminal:\n"
            f"  cd {rig.hw_dir} && ./build/hold_pose_demo --config config/hold_pose.yml "
            f"--arm {arm.name} --mode damp --duration 0 --shm {rig.shm_name}"
        )
    width, height = int(args.color[0]), int(args.color[1])
    fps = int(args.color[2]) if len(args.color) > 2 else None
    camera = ColorCamera(rig.rig.camera_serial, width, height, fps)
    print(
        f"camera {camera.name} serial {camera.serial} usb {camera.usb} "
        f"colour {camera.width}x{camera.height}@{camera.fps} "
        f"fx {camera.camera_matrix[0, 0]:.1f} distortion {camera.distortion_model} "
        f"{camera.distortion_coefficients}"
    )
    detector = PatternDetector(pattern, camera.camera_matrix, args.min_corners)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session = CaptureSession(
        SESSIONS_DIR / f"{stamp}_{arm.name}", _capture_header(rig, arm, camera, pattern, args)
    )
    print(f"session {session.directory}")
    stillness = Stillness(args.dq_still, args.settle_s)
    accumulator = PoseAccumulator()
    fk_checked = args.skip_fk_check
    overlay: Matrix | None = None
    flash_until = 0.0
    force_capture = False
    last_status = 0.0
    result: dict[str, Any] | None = None
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            image = camera.read()
            now = time.monotonic()
            state = shm.read(arm.shm_index)
            lines: list[tuple[str, tuple[int, int, int]]] = []
            capturable = False
            status = ""
            status_colour = _RED
            detection: Detection | None = None
            reason = ""
            if state is None or not state.publishing:
                status = "NO DRIVER STATE"
                reason = shm.why_not_publishing(state, arm.name)
                accumulator.reset()
            elif state.fault != 0:
                status = f"DRIVER FAULT {state.fault} - restart the driver"
                accumulator.reset()
            else:
                if not fk_checked:
                    fk_hand = fk.world_from_body(state.q)[:3, 3]
                    gap = float(np.linalg.norm(fk_hand - state.hand)) * 1e3
                    print(
                        f"FK check: model hand {np.round(fk_hand, 4).tolist()} vs driver hand "
                        f"{np.round(state.hand, 4).tolist()} -> {gap:.1f} mm"
                    )
                    if gap > 5.0:
                        print(
                            "forward kinematics disagree with the driver by more than 5 mm; "
                            "wrong model, body, or joint order"
                        )
                        return 1
                    fk_checked = True
                detection, reason = detector.detect(image, camera.undistort_points)
                still, held = stillness.update(state.dq, now)
                if detection is None:
                    status, status_colour = reason.upper(), _RED
                    accumulator.reset()
                elif detection.view_angle_deg > args.max_view_angle:
                    status = f"BOARD TOO OBLIQUE ({detection.view_angle_deg:.0f} deg)"
                    accumulator.reset()
                elif detection.observation.reprojection_rms_pixels > args.max_pnp_rms:
                    rms = detection.observation.reprojection_rms_pixels
                    status = f"PNP RMS {rms:.1f} px - board flat? lighting?"
                    accumulator.reset()
                elif not still:
                    status, status_colour = (
                        ("MOVING" if held == 0.0 else f"SETTLING {held:.1f}/{args.settle_s:.1f} s"),
                        _YELLOW,
                    )
                    accumulator.reset()
                else:
                    rotation = fk.world_from_body(state.q)[:3, :3]
                    nearest = nearest_rotation_deg(session.poses, rotation)
                    if nearest < args.min_rotation_deg and not force_capture:
                        status = (
                            f"TOO SIMILAR: nearest pose {nearest:.1f} deg away "
                            f"(need {args.min_rotation_deg:.0f})"
                        )
                        status_colour = _YELLOW
                        accumulator.reset()
                    else:
                        capturable = True
                        accumulator.add(detection, state, image, now)
                        status = f"CAPTURING {len(accumulator.frames)}/{args.frames_per_pose}"
                        status_colour = _GREEN
                        if accumulator.ready(args.frames_per_pose, now, 0.6):
                            pose = average_pose(accumulator, detector, fk, session.next_id())
                            accumulator.reset()
                            force_capture = False
                            if pose is None:
                                status = "AVERAGING FAILED (corners changed); hold still"
                            else:
                                session.add(pose)
                                flash_until = now + 0.6
                                diversity = motion_diversity(
                                    [p.world_from_hand[:3, :3] for p in session.poses]
                                )
                                print(
                                    f"captured {pose.sample_id}: corners {pose.ids.size}, "
                                    f"PnP {pose.observation.reprojection_rms_pixels:.2f} px, "
                                    f"view {pose.view_angle_deg:.0f} deg, "
                                    f"{pose.observation.camera_from_target[2, 3]:.2f} m, "
                                    f"q std {pose.q_std_max * 1e3:.2f} mrad | "
                                    f"{len(session.poses)} poses, max rot "
                                    f"{diversity.max_relative_rotation_deg:.0f} deg, "
                                    f"spread {diversity.axis_spread:.2f}",
                                    flush=True,
                                )
                                print("\a", end="", flush=True)
            # ----- overlay
            if detection is not None:
                draw_detection(
                    image, detection, camera.camera_matrix, _GREEN if capturable else _YELLOW
                )
            if overlay is not None and state is not None and state.publishing:
                draw_skeleton(
                    image, fk, state.q, arm.name, invert_transform(overlay), camera.camera_matrix
                )
            if now < flash_until:
                cv2.rectangle(image, (0, 0), (image.shape[1] - 1, image.shape[0] - 1), _GREEN, 12)
            n = len(session.poses)
            lines.append(
                (f"{arm.name} arm  poses {n}/{args.target_samples}   {status}", status_colour)
            )
            if state is not None and state.publishing:
                lines.append(
                    (
                        f"driver: q age {state.age_ms:.0f} ms  "
                        f"|dq|max {float(np.max(np.abs(state.dq))):.3f} rad/s  "
                        f"hand {np.round(state.hand, 3).tolist()}",
                        _WHITE,
                    )
                )
            else:
                lines.append((reason[:110], _RED))
            if detection is not None:
                lines.append(
                    (
                        f"board: {detection.ids.size} corners  "
                        f"PnP {detection.observation.reprojection_rms_pixels:.2f} px  "
                        f"view {detection.view_angle_deg:.0f} deg  {detection.distance_m:.2f} m",
                        _WHITE,
                    )
                )
            if n >= 2:
                diversity = motion_diversity([p.world_from_hand[:3, :3] for p in session.poses])
                lines.append(
                    (
                        f"diversity: max rot {diversity.max_relative_rotation_deg:.0f} deg "
                        f"(need >15), axis spread {diversity.axis_spread:.2f} (need >0.05)   "
                        f"{rotation_hint(session.poses)}",
                        _WHITE,
                    )
                )
            else:
                lines.append((rotation_hint(session.poses), _WHITE))
            if result is not None:
                lines.append(
                    (
                        f"solved: RMS {result['reprojection_rms_pixels']:.2f} px, camera at "
                        f"{np.round(result['camera_position_world_metres'], 3).tolist()} m; "
                        "cyan = arm from FK through it",
                        _CYAN,
                    )
                )
            lines.append(
                ("space: capture anyway   u: undo   s: solve   q: quit and solve", (180, 180, 180))
            )
            draw_panel(image, lines)
            cv2.imshow(WINDOW, image)
            if now - last_status >= 5.0:
                last_status = now
                print(f"{status}  poses={n}", flush=True)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                force_capture = True
            elif key == ord("u"):
                undone = session.undo()
                print(f"removed {undone.sample_id}" if undone else "nothing to undo")
            elif key == ord("s"):
                result = solve_and_save(
                    session.directory,
                    holdout_every=args.holdout_every,
                    minimum_samples=args.min_samples,
                )
                if result is not None:
                    overlay = np.asarray(result["world_from_camera"], dtype=np.float64)
    except KeyboardInterrupt:
        print("\ninterrupted by operator")
    finally:
        camera.close()
        shm.close()
        cv2.destroyAllWindows()
    if not args.no_solve and len(session.poses) >= args.min_samples:
        result = solve_and_save(
            session.directory, holdout_every=args.holdout_every, minimum_samples=args.min_samples
        )
    elif len(session.poses) < args.min_samples:
        print(
            f"{len(session.poses)} poses saved in {session.directory}; "
            f"at least {args.min_samples} are needed to solve"
        )
    return 0 if result is not None or args.no_solve else 1


def run_solve(args: argparse.Namespace) -> int:
    result = solve_and_save(
        Path(args.session), holdout_every=args.holdout_every, minimum_samples=args.min_samples
    )
    return 0 if result is not None else 1


def run_check(args: argparse.Namespace) -> int:
    rig = load_cpf_rig(args.rig)
    extrinsic = load_extrinsic_file(args.calibration)
    camera_from_world = invert_transform(extrinsic.world_from_camera)
    arms = [rig.arm(name) for name in args.arms]
    fks = {
        arm.name: MujocoForwardKinematics(rig.model_xml, arm.profile.joint_names, arm.mujoco_body)
        for arm in arms
    }
    shm = SharedMemoryArms(rig.hw_dir, rig.shm_name)
    width, height = int(args.color[0]), int(args.color[1])
    camera = ColorCamera(
        rig.rig.camera_serial, width, height, int(args.color[2]) if len(args.color) > 2 else None
    )
    record = extrinsic.record
    board_arm = record.get("arm") if "flange_from_target" in record else None
    detector = None
    if board_arm in fks and "board" in record:
        detector = PatternDetector(pattern_from_record(record["board"]), camera.camera_matrix, 6)
        flange_from_target = np.asarray(record["flange_from_target"], dtype=np.float64)
    print(f"checking {extrinsic.path} ({record.get('method', '?')[:60]}...)")
    print(
        "cyan skeleton = arm from FK projected through the calibration; "
        "it should sit on the real arm"
    )
    last_print = 0.0
    last_snapshot = 0.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            image = camera.read()
            lines: list[tuple[str, tuple[int, int, int]]] = [
                (
                    f"check {extrinsic.path.name}: camera at "
                    f"{np.round(extrinsic.world_from_camera[:3, 3], 3).tolist()} m",
                    _CYAN,
                )
            ]
            states = {arm.name: shm.read(arm.shm_index) for arm in arms}
            for arm in arms:
                state = states[arm.name]
                if state is None or not state.publishing:
                    lines.append((shm.why_not_publishing(state, arm.name)[:110], _RED))
                    continue
                draw_skeleton(
                    image, fks[arm.name], state.q, arm.name, camera_from_world, camera.camera_matrix
                )
                if detector is not None and arm.name == board_arm:
                    world_from_hand = fks[arm.name].world_from_body(state.q)
                    predicted_target = camera_from_world @ world_from_hand @ flange_from_target
                    corners_world = (
                        detector.object_points @ predicted_target[:3, :3].T
                        + predicted_target[:3, 3]
                    )
                    pixels = project_world_points(corners_world, np.eye(4), camera.camera_matrix)
                    if pixels is not None:
                        for u, v in np.rint(pixels).astype(np.int32):
                            cv2.circle(image, (int(u), int(v)), 5, _MAGENTA, 1, cv2.LINE_AA)
                    detection, _ = detector.detect(image, camera.undistort_points)
                    if detection is not None and pixels is not None:
                        draw_detection(image, detection, camera.camera_matrix, _GREEN)
                        error = np.linalg.norm(
                            pixels[detection.ids] - detection.corners_undistorted, axis=1
                        )
                        mm, deg = transform_difference(
                            predicted_target, detection.observation.camera_from_target
                        )
                        lines.append(
                            (
                                "board: predicted (magenta) vs detected (green) "
                                f"{float(np.mean(error)):.1f} px mean, "
                                f"pose gap {mm:.1f} mm / {deg:.2f} deg",
                                _WHITE,
                            )
                        )
                        if time.monotonic() - last_print >= 1.0:
                            last_print = time.monotonic()
                            print(
                                f"board prediction error {float(np.mean(error)):.1f} px, "
                                f"{mm:.1f} mm, {deg:.2f} deg"
                            )
            lines.append(("q: quit", (180, 180, 180)))
            draw_panel(image, lines)
            if args.snapshot_dir and time.monotonic() - last_snapshot >= args.snapshot_every:
                last_snapshot = time.monotonic()
                folder = Path(args.snapshot_dir)
                folder.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(folder / f"check_{datetime.now().strftime('%H%M%S')}.png"), image)
            cv2.imshow(WINDOW, image)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.close()
        shm.close()
        cv2.destroyAllWindows()
    return 0


# ---------------------------------------------------------------------------------------
# CLI


def _add_board_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--pattern",
        choices=("charuco", "aruco-grid"),
        default="charuco",
        help="charuco: chessboard with markers (rig default); aruco-grid: plain marker grid "
        "(ArUco or AprilTag)",
    )
    parser.add_argument(
        "--squares", type=int, nargs=2, metavar=("X", "Y"), help="charuco: squares along x and y"
    )
    parser.add_argument("--square-mm", type=float, help="charuco: MEASURED printed square in mm")
    parser.add_argument(
        "--marker-mm",
        type=float,
        help="outer edge of one black marker square in mm; aruco-grid: required and measured, "
        "charuco: default keeps the printed ratio",
    )
    parser.add_argument("--dictionary", help="OpenCV dictionary, e.g. DICT_4X4_50 (see identify)")
    parser.add_argument(
        "--legacy-pattern", action="store_true", help="charuco board printed by OpenCV < 4.6"
    )
    parser.add_argument(
        "--markers",
        type=int,
        nargs=2,
        metavar=("COLS", "ROWS"),
        help="aruco-grid: markers per row and number of rows (default 3 2)",
    )
    parser.add_argument(
        "--gap-mm",
        type=float,
        help="aruco-grid: MEASURED white space between two neighbouring black squares in mm",
    )
    parser.add_argument(
        "--ids",
        type=int,
        nargs="+",
        help="aruco-grid: marker ids row by row, left to right, in the print's own frame "
        "(identify prints them); default counts up from --first-id",
    )
    parser.add_argument(
        "--first-id", type=int, default=0, help="aruco-grid: id of the first marker"
    )


def _add_camera_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--color", type=int, nargs="+", default=[1280, 720], metavar="N",
        help="colour WIDTH HEIGHT [FPS]; fps defaults to 15 on USB 3, 6 on USB 2, 30 at 640x480",
    )  # fmt: skip


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--rig", default=str(DEFAULT_RIG_PATH), help="rig profile with a cpf: section"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    board = commands.add_parser(
        "board", help="write a printable ChArUco board or marker grid (PNG + PDF at exact scale)"
    )
    _add_board_arguments(board)
    board.add_argument("--margin-mm", type=float, default=12.0)
    board.add_argument(
        "--out",
        default=str(CALIBRATION_DIR / "charuco_board"),
        help="output path without extension",
    )
    board.set_defaults(run=run_board)

    identify = commands.add_parser(
        "identify", help="find a printed pattern's ArUco dictionary, marker ids and layout"
    )
    identify.add_argument("--image", help="photo of the pattern instead of the live camera")
    identify.add_argument("--seconds", type=float, default=5.0, help="live capture window")
    _add_camera_arguments(identify)
    identify.set_defaults(run=run_identify)

    capture = commands.add_parser("capture", help="hand-guided capture, then solve")
    capture.add_argument("--arm", required=True, choices=("right", "left"))
    _add_board_arguments(capture)
    _add_camera_arguments(capture)
    capture.add_argument("--target-samples", type=int, default=20, help="on-screen goal")
    capture.add_argument(
        "--min-samples", type=int, default=8, help="fewest poses the solver accepts"
    )
    capture.add_argument(
        "--min-rotation-deg",
        type=float,
        default=8.0,
        help="novelty gate against every captured pose",
    )
    capture.add_argument("--settle-s", type=float, default=0.8)
    capture.add_argument(
        "--dq-still", type=float, default=0.03, help="rad/s; every joint below this counts as still"
    )
    capture.add_argument("--frames-per-pose", type=int, default=6)
    capture.add_argument("--min-corners", type=int, default=12)
    capture.add_argument("--max-view-angle", type=float, default=65.0)
    capture.add_argument("--max-pnp-rms", type=float, default=1.5)
    capture.add_argument(
        "--holdout-every", type=int, default=4, help="every n-th pose is held out for validation"
    )
    capture.add_argument("--skip-fk-check", action="store_true")
    capture.add_argument("--no-solve", action="store_true", help="only record the session")
    capture.set_defaults(run=run_capture)

    solve = commands.add_parser("solve", help="re-solve a recorded session")
    solve.add_argument("--session", required=True)
    solve.add_argument("--min-samples", type=int, default=8)
    solve.add_argument("--holdout-every", type=int, default=4)
    solve.set_defaults(run=run_solve)

    check = commands.add_parser(
        "check", help="project the arm through a calibration onto the live image"
    )
    check.add_argument("--arms", nargs="+", default=["right"], choices=("right", "left"))
    check.add_argument(
        "--calibration", default=str(CURRENT_CALIBRATION), help="extrinsic JSON (hand-eye or tape)"
    )
    _add_camera_arguments(check)
    check.add_argument("--snapshot-dir", help="also save the annotated frame here periodically")
    check.add_argument(
        "--snapshot-every", type=float, default=2.0, help="seconds between snapshots"
    )
    check.set_defaults(run=run_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.run(args))
    except (RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
