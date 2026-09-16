"""Robot-agnostic hand-eye calibration from ChArUco observations and forward kinematics.

Nothing here knows which robot produced the data. A sample is the robot's forward
kinematics ``base_from_flange`` (from any URDF/TF source) plus the calibration board as
the camera saw it at that instant. Two mounts are supported:

- ``EYE_TO_HAND``: the camera is fixed and the board rides on the flange. The unknown
  ``X`` is ``base_from_camera``; the constant board offset ``Y`` is ``flange_from_target``.
- ``EYE_IN_HAND``: the camera rides on the flange and the board is fixed. ``X`` is
  ``flange_from_camera``; ``Y`` is ``base_from_target``.

Every sample obeys ``camera_from_target = X^-1 · M · Y``, where ``M`` is
``base_from_flange`` (eye-to-hand) or its inverse (eye-in-hand). The pipeline:

1. reject rotation-poor data (:func:`motion_diversity`), because ``X`` is only
   observable from rotations about at least two non-parallel axes;
2. solve ``AX = XB`` in closed form with all five OpenCV methods;
3. refine ``X`` and ``Y`` jointly by minimizing board-corner reprojection error with a
   Huber loss, drop samples whose residual is far above the median, and refine again;
4. judge the result on samples that were not used to solve it (:func:`validate_hand_eye`).

Matrices are 4x4 ``numpy`` arrays named ``a_from_b``: they map a point expressed in frame
``b`` into frame ``a``, the same convention as :class:`RigidTransform3D`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import FrameId
from vision_pipeline.geometry.spatial import RigidTransform3D

from .charuco import CharucoObservation, CharucoTarget

Matrix = NDArray[np.float64]

CLOSED_FORM_METHODS: dict[str, int] = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


class HandEyeMount(StrEnum):
    EYE_TO_HAND = "eye_to_hand"
    EYE_IN_HAND = "eye_in_hand"


class HandEyeDegenerateError(ValueError):
    """The robot motions cannot determine the hand-eye transform."""


# ---------------------------------------------------------------------------------------
# Rigid-transform helpers


def make_transform(rotation: Any, translation: Any) -> Matrix:
    """4x4 transform from a 3x3 rotation and a length-3 translation (array-likes)."""

    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


def invert_transform(matrix: Matrix) -> Matrix:
    rotation = matrix[:3, :3]
    return make_transform(rotation.T, -rotation.T @ matrix[:3, 3])


def rotation_angle_deg(rotation: NDArray[np.float64]) -> float:
    """Geodesic angle of a rotation, accurate near 0 and 180 degrees (atan2, not acos)."""

    cosine = (float(np.trace(rotation)) - 1.0) / 2.0
    sine = float(np.linalg.norm(rotation - rotation.T)) / (2.0 * math.sqrt(2.0))
    return math.degrees(math.atan2(sine, cosine))


def _exp(delta: NDArray[np.float64]) -> Matrix:
    rotation, _ = cv2.Rodrigues(np.asarray(delta[:3], dtype=np.float64))
    return make_transform(rotation, delta[3:])


def _project_to_rotation(matrix: Any) -> NDArray[np.float64]:
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] = -u[:, -1]
        rotation = u @ vt
    return cast(NDArray[np.float64], rotation)


def average_transforms(matrices: Sequence[Matrix]) -> Matrix:
    """Chordal L2 mean of the rotations and arithmetic mean of the translations."""

    if not matrices:
        raise ValueError("at least one transform is required")
    stacked = np.stack(matrices)
    return make_transform(
        _project_to_rotation(stacked[:, :3, :3].mean(axis=0)), stacked[:, :3, 3].mean(axis=0)
    )


def matrix_from_rigid_transform(transform: RigidTransform3D) -> Matrix:
    columns = [transform.apply_vector(axis) for axis in ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))]
    return make_transform(np.array(columns).T, transform.translation_metres)


def rigid_transform_from_matrix(
    matrix: Matrix, *, source_frame: FrameId, target_frame: FrameId
) -> RigidTransform3D:
    """Convert ``target_from_source`` into the repository's transform contract."""

    r = _project_to_rotation(matrix[:3, :3])
    # Bar-Itzhack: the unit quaternion is the dominant eigenvector of this symmetric matrix.
    k = (
        np.array(
            [
                [
                    r[0, 0] - r[1, 1] - r[2, 2],
                    r[0, 1] + r[1, 0],
                    r[0, 2] + r[2, 0],
                    r[2, 1] - r[1, 2],
                ],
                [
                    r[0, 1] + r[1, 0],
                    r[1, 1] - r[0, 0] - r[2, 2],
                    r[1, 2] + r[2, 1],
                    r[0, 2] - r[2, 0],
                ],
                [
                    r[0, 2] + r[2, 0],
                    r[1, 2] + r[2, 1],
                    r[2, 2] - r[0, 0] - r[1, 1],
                    r[1, 0] - r[0, 1],
                ],
                [
                    r[2, 1] - r[1, 2],
                    r[0, 2] - r[2, 0],
                    r[1, 0] - r[0, 1],
                    r[0, 0] + r[1, 1] + r[2, 2],
                ],
            ]
        )
        / 3.0
    )
    values, vectors = np.linalg.eigh(k)
    quaternion = vectors[:, int(np.argmax(values))]
    if quaternion[3] < 0:
        quaternion = -quaternion
    quaternion = quaternion / np.linalg.norm(quaternion)
    return RigidTransform3D(
        source_frame=source_frame,
        target_frame=target_frame,
        translation_metres=(float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3])),
        rotation_xyzw=(
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        ),
    )


# ---------------------------------------------------------------------------------------
# Board observations


@dataclass(frozen=True, slots=True)
class BoardObservation:
    """Identified board corners in one image and the board pose PnP derived from them."""

    object_points: NDArray[np.float64]
    image_points: NDArray[np.float64]
    camera_from_target: Matrix
    reprojection_rms_pixels: float

    def __post_init__(self) -> None:
        if self.object_points.ndim != 2 or self.object_points.shape[1] != 3:
            raise ValueError("object_points must have [N, 3] shape")
        if self.image_points.shape != (self.object_points.shape[0], 2):
            raise ValueError("image_points must have [N, 2] shape matching object_points")
        if self.camera_from_target.shape != (4, 4):
            raise ValueError("camera_from_target must be a 4x4 transform")
        if not math.isfinite(self.reprojection_rms_pixels) or self.reprojection_rms_pixels < 0:
            raise ValueError("reprojection_rms_pixels must be finite and non-negative")

    @property
    def corner_count(self) -> int:
        return int(self.object_points.shape[0])


def _project(
    object_points: NDArray[np.float64],
    camera_from_target: Matrix,
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
) -> NDArray[np.float64]:
    rvec, _ = cv2.Rodrigues(camera_from_target[:3, :3])
    projected, _ = cv2.projectPoints(
        object_points, rvec, camera_from_target[:3, 3], camera_matrix, distortion
    )
    return cast(NDArray[np.float64], projected.reshape(-1, 2))


def board_observation_from_points(
    object_points: NDArray[np.float64],
    image_points: NDArray[np.float64],
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    *,
    minimum_corners: int = 10,
) -> BoardObservation | None:
    """Planar PnP (IPPE, then Levenberg-Marquardt) for identified board corners."""

    object_points = np.ascontiguousarray(object_points, dtype=np.float64)
    image_points = np.ascontiguousarray(image_points, dtype=np.float64)
    if object_points.shape[0] < max(4, minimum_corners):
        return None
    # Collinear corners (one row of the board) do not determine a pose.
    centred = object_points[:, :2] - object_points[:, :2].mean(axis=0)
    if np.linalg.svd(centred, compute_uv=False)[1] < 1e-6:
        return None
    ok, rvec, tvec = cv2.solvePnP(
        object_points, image_points, camera_matrix, distortion, flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        return None
    rvec, tvec = cv2.solvePnPRefineLM(
        object_points, image_points, camera_matrix, distortion, rvec, tvec
    )
    rotation, _ = cv2.Rodrigues(rvec)
    camera_from_target = make_transform(rotation, tvec.reshape(3))
    residual = _project(object_points, camera_from_target, camera_matrix, distortion) - image_points
    return BoardObservation(
        object_points=object_points,
        image_points=image_points,
        camera_from_target=camera_from_target,
        reprojection_rms_pixels=float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
    )


def estimate_board_pose(
    target: CharucoTarget,
    observation: CharucoObservation,
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    *,
    minimum_corners: int = 10,
) -> BoardObservation | None:
    """Pose of a detected ChArUco board in the camera frame, or ``None`` if unusable."""

    ids = observation.ids.reshape(-1)
    object_points = np.asarray(target.board.getChessboardCorners(), dtype=np.float64)[ids]
    image_points = observation.corners.reshape(-1, 2).astype(np.float64)
    return board_observation_from_points(
        object_points, image_points, camera_matrix, distortion, minimum_corners=minimum_corners
    )


def board_view_angle_deg(camera_from_target: Matrix) -> float:
    """Angle between the board normal and the camera's line of sight to the board."""

    normal = camera_from_target[:3, 2]
    sight = camera_from_target[:3, 3] / np.linalg.norm(camera_from_target[:3, 3])
    return math.degrees(math.acos(min(1.0, abs(float(normal @ sight)))))


# ---------------------------------------------------------------------------------------
# Samples, motion diversity, and pose selection


@dataclass(frozen=True, slots=True)
class HandEyeSample:
    """Forward kinematics and the board observation captured at the same robot state."""

    sample_id: str
    base_from_flange: Matrix
    board: BoardObservation

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be a non-empty string")
        if self.base_from_flange.shape != (4, 4) or not np.isfinite(self.base_from_flange).all():
            raise ValueError("base_from_flange must be a finite 4x4 transform")


@dataclass(frozen=True, slots=True)
class MotionDiversity:
    """How well a set of flange rotations constrains the hand-eye rotation.

    ``axis_spread`` is the ratio of the second to the first eigenvalue of the summed
    outer products of relative rotation axes: 0 when every relative rotation shares one
    axis (``X`` is then unobservable), approaching 1 when axes are spread over a plane or
    more.
    """

    samples: int
    max_relative_rotation_deg: float
    median_relative_rotation_deg: float
    axis_spread: float


def motion_diversity(
    flange_rotations: Sequence[NDArray[np.float64]], *, minimum_angle_deg: float = 2.0
) -> MotionDiversity:
    angles: list[float] = []
    scatter = np.zeros((3, 3))
    for i, first in enumerate(flange_rotations):
        for second in flange_rotations[i + 1 :]:
            rvec, _ = cv2.Rodrigues(first.T @ second)
            angle = float(np.linalg.norm(rvec))
            angles.append(math.degrees(angle))
            if math.degrees(angle) >= minimum_angle_deg:
                axis = np.asarray(rvec, dtype=np.float64).reshape(3) / angle
                scatter = scatter + np.outer(axis, axis).astype(np.float64)
    eigenvalues = np.sort(np.linalg.eigvalsh(scatter))[::-1]
    spread = float(eigenvalues[1] / eigenvalues[0]) if eigenvalues[0] > 0 else 0.0
    return MotionDiversity(
        samples=len(flange_rotations),
        max_relative_rotation_deg=max(angles, default=0.0),
        median_relative_rotation_deg=float(np.median(angles)) if angles else 0.0,
        axis_spread=spread,
    )


def select_diverse_pose(
    candidates: Sequence[Matrix],
    captured: Sequence[Matrix],
    *,
    acceptable: Callable[[Matrix], bool] | None = None,
) -> int | None:
    """Index of the candidate flange pose whose rotation differs most from all captured ones.

    This greedy max-min rule follows the Tsai-Lenz guidance (large inter-station angles)
    and is the "next best pose" step of automatic calibration. ``acceptable`` filters
    candidates, e.g. board predicted in view or collision-free.
    """

    best_index: int | None = None
    best_score = -1.0
    for index, candidate in enumerate(candidates):
        if acceptable is not None and not acceptable(candidate):
            continue
        score = min(
            (rotation_angle_deg(candidate[:3, :3].T @ pose[:3, :3]) for pose in captured),
            default=180.0,
        )
        if score > best_score:
            best_index, best_score = index, score
    return best_index


# ---------------------------------------------------------------------------------------
# Solving


def _chain(mount: HandEyeMount, base_from_flange: Matrix) -> Matrix:
    return (
        base_from_flange
        if mount is HandEyeMount.EYE_TO_HAND
        else invert_transform(base_from_flange)
    )


def predict_camera_from_target(
    mount: HandEyeMount, camera_to_robot: Matrix, target_offset: Matrix, base_from_flange: Matrix
) -> Matrix:
    return invert_transform(camera_to_robot) @ _chain(mount, base_from_flange) @ target_offset


def solve_closed_form(samples: Sequence[HandEyeSample], mount: HandEyeMount) -> dict[str, Matrix]:
    """``X`` from every OpenCV ``AX = XB`` method.

    For eye-to-hand the robot poses are inverted before the call, which makes OpenCV's
    ``cam2gripper`` output equal to ``base_from_camera``.
    """

    robot = [
        sample.base_from_flange
        if mount is HandEyeMount.EYE_IN_HAND
        else invert_transform(sample.base_from_flange)
        for sample in samples
    ]
    rotations_robot = [pose[:3, :3] for pose in robot]
    translations_robot = [pose[:3, 3].reshape(3, 1) for pose in robot]
    rotations_board = [sample.board.camera_from_target[:3, :3] for sample in samples]
    translations_board = [
        sample.board.camera_from_target[:3, 3].reshape(3, 1) for sample in samples
    ]
    solutions: dict[str, Matrix] = {}
    for name, method in CLOSED_FORM_METHODS.items():
        rotation, translation = cv2.calibrateHandEye(
            rotations_robot, translations_robot, rotations_board, translations_board, method=method
        )
        solution = make_transform(_project_to_rotation(rotation), translation.reshape(3))
        if np.isfinite(solution).all():
            solutions[name] = solution
    return solutions


def estimate_target_offset(
    samples: Sequence[HandEyeSample], mount: HandEyeMount, camera_to_robot: Matrix
) -> Matrix:
    """Average the constant board offset ``Y`` implied by each sample for a given ``X``."""

    return average_transforms(
        [
            invert_transform(_chain(mount, sample.base_from_flange))
            @ camera_to_robot
            @ sample.board.camera_from_target
            for sample in samples
        ]
    )


def _residuals(
    samples: Sequence[HandEyeSample],
    mount: HandEyeMount,
    camera_to_robot: Matrix,
    target_offset: Matrix,
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
) -> list[NDArray[np.float64]]:
    return [
        _project(
            sample.board.object_points,
            predict_camera_from_target(
                mount, camera_to_robot, target_offset, sample.base_from_flange
            ),
            camera_matrix,
            distortion,
        )
        - sample.board.image_points
        for sample in samples
    ]


def refine_hand_eye(
    samples: Sequence[HandEyeSample],
    mount: HandEyeMount,
    camera_to_robot: Matrix,
    target_offset: Matrix,
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    *,
    huber_pixels: float = 1.0,
    iterations: int = 50,
) -> tuple[Matrix, Matrix]:
    """Levenberg-Marquardt on ``X`` and ``Y`` minimizing Huber-weighted corner reprojection.

    Both transforms are updated by right-multiplying small se(3) increments, so the
    parameterization never meets a rotation-vector singularity.
    """

    x, y = camera_to_robot.copy(), target_offset.copy()
    damping = 1e-3

    def stacked(x_: Matrix, y_: Matrix) -> NDArray[np.float64]:
        return np.concatenate(
            [r.reshape(-1) for r in _residuals(samples, mount, x_, y_, camera_matrix, distortion)]
        )

    def weights(residual: NDArray[np.float64]) -> NDArray[np.float64]:
        norms = np.linalg.norm(residual.reshape(-1, 2), axis=1)
        per_corner = np.where(norms <= huber_pixels, 1.0, huber_pixels / np.maximum(norms, 1e-12))
        return cast(NDArray[np.float64], np.repeat(per_corner, 2))

    residual = stacked(x, y)
    for _ in range(iterations):
        w = weights(residual)
        cost = float(np.sum(w * residual**2))
        jacobian = np.empty((residual.size, 12))
        step = 1e-6
        for k in range(12):
            delta: NDArray[np.float64] = np.zeros(12, dtype=np.float64)
            delta[k] = step
            jacobian[:, k] = (stacked(x @ _exp(delta[:6]), y @ _exp(delta[6:])) - residual) / step
        normal = jacobian.T @ (w[:, None] * jacobian)
        gradient = jacobian.T @ (w * residual)
        while True:
            try:
                update: NDArray[np.float64] = np.asarray(
                    -np.linalg.solve(normal + damping * np.diag(np.diag(normal)), gradient),
                    dtype=np.float64,
                )
            except np.linalg.LinAlgError:
                damping *= 10
                continue
            x_new, y_new = x @ _exp(update[:6]), y @ _exp(update[6:])
            candidate = stacked(x_new, y_new)
            if float(np.sum(w * candidate**2)) < cost:
                x, y, residual = x_new, y_new, candidate
                damping = max(damping / 3, 1e-9)
                break
            damping *= 5
            if damping > 1e8:
                return x, y
        if np.linalg.norm(update) < 1e-10:
            break
    return x, y


@dataclass(frozen=True, slots=True)
class HandEyeCalibration:
    """A refined hand-eye result with the evidence needed to trust or reject it."""

    mount: HandEyeMount
    camera_to_robot: Matrix
    target_offset: Matrix
    reprojection_rms_pixels: float
    sample_ids: tuple[str, ...]
    rejected_sample_ids: tuple[str, ...]
    per_sample_rms_pixels: Mapping[str, float]
    closed_form: Mapping[str, Matrix]
    closed_form_deviation: Mapping[str, tuple[float, float]]
    diversity: MotionDiversity

    def as_rigid_transform(
        self, *, robot_frame: FrameId, camera_frame: FrameId
    ) -> RigidTransform3D:
        """``camera_to_robot`` as a transform whose source is the camera frame."""

        return rigid_transform_from_matrix(
            self.camera_to_robot, source_frame=camera_frame, target_frame=robot_frame
        )


def _rms(residuals: Sequence[NDArray[np.float64]]) -> float:
    corners = np.concatenate([r.reshape(-1, 2) for r in residuals])
    return float(np.sqrt(np.mean(np.sum(corners**2, axis=1))))


def calibrate_hand_eye(
    samples: Sequence[HandEyeSample],
    mount: HandEyeMount,
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
    *,
    minimum_samples: int = 8,
    minimum_rotation_deg: float = 15.0,
    minimum_axis_spread: float = 0.05,
    outlier_factor: float = 3.0,
    minimum_outlier_pixels: float = 2.0,
    max_rejected_fraction: float = 0.2,
) -> HandEyeCalibration:
    """Closed-form initialization, robust refinement, one round of outlier rejection."""

    if len(samples) < minimum_samples:
        raise HandEyeDegenerateError(f"{len(samples)} samples; at least {minimum_samples} required")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("sample_id values must be unique")
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
    distortion = np.asarray(distortion, dtype=np.float64)
    diversity = motion_diversity([sample.base_from_flange[:3, :3] for sample in samples])
    if diversity.max_relative_rotation_deg < minimum_rotation_deg:
        raise HandEyeDegenerateError(
            f"largest rotation between poses is {diversity.max_relative_rotation_deg:.1f} deg; "
            f"at least {minimum_rotation_deg:.1f} deg required"
        )
    if diversity.axis_spread < minimum_axis_spread:
        raise HandEyeDegenerateError(
            f"rotation axes are nearly parallel (spread {diversity.axis_spread:.3f}); "
            "rotate the flange about at least two different axes"
        )

    closed_form = solve_closed_form(samples, mount)
    if not closed_form:
        raise HandEyeDegenerateError("no closed-form method produced a finite solution")
    initial_scores = {}
    for name, solution in closed_form.items():
        offset = estimate_target_offset(samples, mount, solution)
        initial_scores[name] = (
            _rms(_residuals(samples, mount, solution, offset, camera_matrix, distortion)),
            offset,
        )
    best = min(initial_scores, key=lambda name: initial_scores[name][0])
    x, y = refine_hand_eye(
        samples, mount, closed_form[best], initial_scores[best][1], camera_matrix, distortion
    )

    per_sample = {
        sample.sample_id: float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
        for sample, residual in zip(
            samples, _residuals(samples, mount, x, y, camera_matrix, distortion), strict=True
        )
    }
    threshold = max(
        minimum_outlier_pixels, outlier_factor * float(np.median(list(per_sample.values())))
    )
    worst_first = sorted(per_sample, key=per_sample.__getitem__, reverse=True)
    allowed = int(max_rejected_fraction * len(samples))
    rejected = tuple(sid for sid in worst_first[:allowed] if per_sample[sid] > threshold)
    kept = [sample for sample in samples if sample.sample_id not in rejected]
    if rejected:
        x, y = refine_hand_eye(kept, mount, x, y, camera_matrix, distortion)

    deviation = {
        name: (
            float(np.linalg.norm(solution[:3, 3] - x[:3, 3]) * 1e3),
            rotation_angle_deg(solution[:3, :3].T @ x[:3, :3]),
        )
        for name, solution in closed_form.items()
    }
    return HandEyeCalibration(
        mount=mount,
        camera_to_robot=x,
        target_offset=y,
        reprojection_rms_pixels=_rms(_residuals(kept, mount, x, y, camera_matrix, distortion)),
        sample_ids=tuple(sample.sample_id for sample in kept),
        rejected_sample_ids=rejected,
        per_sample_rms_pixels=per_sample,
        closed_form=closed_form,
        closed_form_deviation=deviation,
        diversity=diversity,
    )


# ---------------------------------------------------------------------------------------
# Validation


@dataclass(frozen=True, slots=True)
class HandEyeValidation:
    """Agreement between the calibration's predictions and held-out board observations.

    Corner errors compare predicted board corners in the camera frame with the corners
    placed by each sample's own PnP pose, so they include PnP depth noise.
    """

    samples: int
    reprojection_rms_pixels: float
    corner_error_mean_mm: float
    corner_error_p95_mm: float
    corner_error_max_mm: float
    rotation_error_mean_deg: float
    rotation_error_max_deg: float


def validate_hand_eye(
    calibration: HandEyeCalibration,
    samples: Sequence[HandEyeSample],
    camera_matrix: NDArray[np.float64],
    distortion: NDArray[np.float64],
) -> HandEyeValidation:
    if not samples:
        raise ValueError("validation needs at least one held-out sample")
    reused = set(calibration.sample_ids) & {sample.sample_id for sample in samples}
    if reused:
        raise ValueError(f"validation samples were used to calibrate: {sorted(reused)}")
    residuals = _residuals(
        samples,
        calibration.mount,
        calibration.camera_to_robot,
        calibration.target_offset,
        np.asarray(camera_matrix, dtype=np.float64),
        np.asarray(distortion, dtype=np.float64),
    )
    corner_errors: list[float] = []
    rotation_errors: list[float] = []
    for sample in samples:
        predicted = predict_camera_from_target(
            calibration.mount,
            calibration.camera_to_robot,
            calibration.target_offset,
            sample.base_from_flange,
        )
        homogeneous = np.c_[sample.board.object_points, np.ones(sample.board.corner_count)]
        difference = (predicted @ homogeneous.T - sample.board.camera_from_target @ homogeneous.T)[
            :3
        ]
        corner_errors.extend((np.linalg.norm(difference, axis=0) * 1e3).tolist())
        rotation_errors.append(
            rotation_angle_deg(predicted[:3, :3].T @ sample.board.camera_from_target[:3, :3])
        )
    return HandEyeValidation(
        samples=len(samples),
        reprojection_rms_pixels=_rms(residuals),
        corner_error_mean_mm=float(np.mean(corner_errors)),
        corner_error_p95_mm=float(np.percentile(corner_errors, 95)),
        corner_error_max_mm=float(np.max(corner_errors)),
        rotation_error_mean_deg=float(np.mean(rotation_errors)),
        rotation_error_max_deg=float(np.max(rotation_errors)),
    )


def relative_base_error(
    left_base_from_camera: Matrix,
    right_base_from_camera: Matrix,
    expected_left_base_from_right_base: Matrix,
) -> tuple[float, float]:
    """Translation (mm) and rotation (deg) error of the arm-to-arm transform.

    Two independently calibrated eye-to-hand arms imply ``left_base_from_right_base``;
    comparing it with the nominal mounting (e.g. from a shared URDF body link) exposes
    mounting or calibration error that neither arm's own residuals can show.
    """

    measured = left_base_from_camera @ invert_transform(right_base_from_camera)
    error = invert_transform(expected_left_base_from_right_base) @ measured
    return float(np.linalg.norm(error[:3, 3]) * 1e3), rotation_angle_deg(error[:3, :3])


__all__ = [
    "CLOSED_FORM_METHODS",
    "BoardObservation",
    "HandEyeCalibration",
    "HandEyeDegenerateError",
    "HandEyeMount",
    "HandEyeSample",
    "HandEyeValidation",
    "MotionDiversity",
    "average_transforms",
    "board_observation_from_points",
    "board_view_angle_deg",
    "calibrate_hand_eye",
    "estimate_board_pose",
    "estimate_target_offset",
    "invert_transform",
    "make_transform",
    "matrix_from_rigid_transform",
    "motion_diversity",
    "predict_camera_from_target",
    "refine_hand_eye",
    "relative_base_error",
    "rigid_transform_from_matrix",
    "rotation_angle_deg",
    "select_diverse_pose",
    "solve_closed_form",
    "validate_hand_eye",
]
