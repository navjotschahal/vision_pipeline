"""Robust point-to-plane registration of depth points to a posed robot model.

One rigid transform ``world_from_camera`` is solved for all views at once. A view is a
depth cloud in the camera frame plus the model surface (points and normals, world frame)
for the joint configuration that was current when the depth was captured. Optional plane
points, for example the corners of a marker board lying on the table, are constrained to
a known world height; the table is the robot's own reference plane, so this pins roll,
pitch and height far better than the thin arm links can.

The solver is iteratively re-weighted Gauss-Newton on se(3) with a Huber loss and a
correspondence distance that anneals from ``max_distance_m`` to ``final_distance_m``.
The information matrix at the solution is reported so a poorly observed direction is
visible instead of silently absorbed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from .hand_eye import Matrix, make_transform

_AXIS_LABELS = ("roll (about x)", "pitch (about y)", "yaw (about z)", "x", "y", "z")
# Rotations are compared with translations through a lever arm of about the working
# distance, so the six eigenvalues share a unit.
_LEVER_M = 0.7


@dataclass(frozen=True, slots=True)
class RegistrationView:
    """Depth points (camera frame) with the model surface (world frame) for one capture."""

    name: str
    points_camera: NDArray[np.float64]
    model_points_world: NDArray[np.float64]
    model_normals_world: NDArray[np.float64]
    plane_points_camera: NDArray[np.float64] | None = None
    plane_height_world: float = 0.0

    def __post_init__(self) -> None:
        for name in ("points_camera", "model_points_world", "model_normals_world"):
            value = getattr(self, name)
            if value.ndim != 2 or value.shape[1] != 3:
                raise ValueError(f"{name} must have [N, 3] shape")
        if self.model_points_world.shape != self.model_normals_world.shape:
            raise ValueError("model points and normals must match")
        if self.plane_points_camera is not None and (
            self.plane_points_camera.ndim != 2 or self.plane_points_camera.shape[1] != 3
        ):
            raise ValueError("plane_points_camera must have [K, 3] shape")


@dataclass(frozen=True, slots=True)
class ViewFit:
    name: str
    candidates: int
    inliers: int
    rms_mm: float
    median_mm: float
    model_coverage: float


@dataclass(frozen=True, slots=True)
class RegistrationResult:
    world_from_camera: Matrix
    iterations: int
    converged: bool
    views: tuple[ViewFit, ...]
    inliers: int
    rms_mm: float
    plane_points: int
    plane_rms_mm: float | None
    # Eigenvalues of the scaled information matrix, smallest first, normalised by the
    # largest, and the axis label of the weakest direction.
    observability: tuple[float, ...]
    weakest: str


class RegistrationError(ValueError):
    """The registration could not be carried out on this data."""


def _exp_increment(delta: NDArray[np.float64]) -> Matrix:
    rotation, _ = cv2.Rodrigues(np.asarray(delta[:3], dtype=np.float64))
    return make_transform(rotation, delta[3:])


def _transform(points: NDArray[np.float64], matrix: Matrix) -> NDArray[np.float64]:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def _huber(residual: NDArray[np.float64], threshold: float) -> NDArray[np.float64]:
    magnitude = np.abs(residual)
    return np.where(magnitude <= threshold, 1.0, threshold / np.maximum(magnitude, 1e-12))


class _ViewData:
    def __init__(self, view: RegistrationView) -> None:
        self.view = view
        self.tree = cKDTree(view.model_points_world)

    def correspondences(
        self, world_from_camera: Matrix, distance: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """World points, matched model points and normals within ``distance``."""

        world = _transform(self.view.points_camera, world_from_camera)
        dist, index = self.tree.query(world, distance_upper_bound=distance)
        keep = np.isfinite(dist)
        matched = index[keep]
        return (
            world[keep],
            self.view.model_points_world[matched],
            self.view.model_normals_world[matched],
        )


def _fit(view_data: _ViewData, world_from_camera: Matrix, distance: float) -> ViewFit:
    world, model, normals = view_data.correspondences(world_from_camera, distance)
    residual = np.sum(normals * (world - model), axis=1)
    coverage = 0.0
    if world.shape[0] > 0:
        depth_tree = cKDTree(world)
        hits, _ = depth_tree.query(view_data.view.model_points_world, distance_upper_bound=distance)
        coverage = float(np.mean(np.isfinite(hits)))
    return ViewFit(
        name=view_data.view.name,
        candidates=int(view_data.view.points_camera.shape[0]),
        inliers=int(world.shape[0]),
        rms_mm=float(np.sqrt(np.mean(residual**2)) * 1e3) if residual.size else math.nan,
        median_mm=float(np.median(np.abs(residual)) * 1e3) if residual.size else math.nan,
        model_coverage=coverage,
    )


def evaluate_view(
    view: RegistrationView, world_from_camera: Matrix, *, distance_m: float
) -> ViewFit:
    """Point-to-plane statistics of one view at a fixed camera pose (for held-out checks)."""

    return _fit(_ViewData(view), world_from_camera, distance_m)


def register_views(
    views: list[RegistrationView],
    world_from_camera: Matrix,
    *,
    max_distance_m: float = 0.04,
    final_distance_m: float = 0.01,
    iterations: int = 40,
    huber_m: float = 0.006,
    depth_sigma_m: float = 0.004,
    plane_sigma_m: float = 0.0015,
    min_inliers: int = 300,
    prior_translation_m: float | None = 0.03,
    prior_rotation_deg: float | None = 3.0,
) -> RegistrationResult:
    """Solve ``world_from_camera`` from all views starting at the given estimate.

    The prior keeps the solution near the start along directions the depth does not
    constrain (a black arm returns little depth, a square column has no yaw signal);
    ``None`` or zero disables it. The report's observability says which directions the
    data itself fixed.
    """

    if not views:
        raise RegistrationError("at least one view is required")
    data = [_ViewData(view) for view in views]
    x0 = np.asarray(world_from_camera, dtype=np.float64).copy()
    x = x0.copy()
    anneal_steps = max(1, int(0.6 * iterations))
    converged = False
    iteration = 0
    depth_weight = 1.0 / depth_sigma_m**2
    plane_weight = 1.0 / plane_sigma_m**2
    prior_t_weight = 1.0 / prior_translation_m**2 if prior_translation_m else 0.0
    prior_r_weight = 1.0 / math.radians(prior_rotation_deg) ** 2 if prior_rotation_deg else 0.0
    z_axis = np.array([0.0, 0.0, 1.0])
    information = np.zeros((6, 6))
    for iteration in range(1, iterations + 1):
        fraction = min(1.0, (iteration - 1) / anneal_steps)
        distance = max_distance_m + (final_distance_m - max_distance_m) * fraction
        jacobians: list[NDArray[np.float64]] = []
        residuals: list[NDArray[np.float64]] = []
        weights: list[NDArray[np.float64]] = []
        total_inliers = 0
        for item in data:
            world, model, normals = item.correspondences(x, distance)
            if world.shape[0] == 0:
                continue
            residual = np.sum(normals * (world - model), axis=1)
            jacobian = np.hstack((np.cross(world, normals), normals))
            jacobians.append(jacobian)
            residuals.append(residual)
            weights.append(depth_weight * _huber(residual, huber_m))
            total_inliers += int(world.shape[0])
            plane = item.view.plane_points_camera
            if plane is not None and plane.shape[0] > 0:
                board = _transform(plane, x)
                plane_residual = board[:, 2] - item.view.plane_height_world
                z_rows = np.broadcast_to(z_axis, board.shape)
                plane_jacobian = np.hstack((np.cross(board, z_rows), z_rows))
                jacobians.append(plane_jacobian)
                residuals.append(plane_residual)
                weights.append(np.full(plane_residual.shape[0], plane_weight))
        if total_inliers < min_inliers:
            raise RegistrationError(
                f"only {total_inliers} depth points within {distance * 1e3:.0f} mm of the model at "
                f"iteration {iteration}; bad initial estimate or model not in view"
            )
        if prior_t_weight > 0.0 or prior_r_weight > 0.0:
            # Left-multiplied increment: t' = t + w x t + v, rotvec(R R0^T)' = rotvec + w.
            translation = x[:3, 3]
            skew = np.array(
                [
                    [0.0, -translation[2], translation[1]],
                    [translation[2], 0.0, -translation[0]],
                    [-translation[1], translation[0], 0.0],
                ]
            )
            rotvec, _ = cv2.Rodrigues(x[:3, :3] @ x0[:3, :3].T)
            prior_jacobian = np.zeros((6, 6))
            prior_jacobian[:3, :3] = -skew
            prior_jacobian[:3, 3:] = np.eye(3)
            prior_jacobian[3:, :3] = np.eye(3)
            jacobians.append(prior_jacobian)
            residuals.append(
                np.concatenate((translation - x0[:3, 3], np.asarray(rotvec).reshape(3)))
            )
            weights.append(np.array([prior_t_weight] * 3 + [prior_r_weight] * 3))
        jacobian_all = np.concatenate(jacobians)
        residual_all = np.concatenate(residuals)
        weight_all = np.concatenate(weights)
        information = jacobian_all.T @ (weight_all[:, None] * jacobian_all)
        gradient = jacobian_all.T @ (weight_all * residual_all)
        damping = 1e-6 * float(np.trace(information)) / 6.0
        try:
            delta = -np.linalg.solve(information + damping * np.eye(6), gradient)
        except np.linalg.LinAlgError as error:
            raise RegistrationError(
                "singular normal equations; the views do not constrain the pose"
            ) from error
        x = _exp_increment(np.asarray(delta, dtype=np.float64)) @ x
        if (
            np.linalg.norm(delta[:3]) < 1e-6
            and np.linalg.norm(delta[3:]) < 1e-6
            and fraction >= 1.0
        ):
            converged = True
            break

    fits = tuple(_fit(item, x, final_distance_m) for item in data)
    all_residuals = []
    for item in data:
        world, model, normals = item.correspondences(x, final_distance_m)
        all_residuals.append(np.sum(normals * (world - model), axis=1))
    residual_final = np.concatenate(all_residuals) if all_residuals else np.zeros(0)
    plane_residuals = [
        _transform(item.view.plane_points_camera, x)[:, 2] - item.view.plane_height_world
        for item in data
        if item.view.plane_points_camera is not None and item.view.plane_points_camera.shape[0] > 0
    ]
    plane_all = np.concatenate(plane_residuals) if plane_residuals else np.zeros(0)
    # Observability of the DATA alone: the information without the prior rows.
    data_information = information.copy()
    if prior_t_weight > 0.0 or prior_r_weight > 0.0:
        data_information -= prior_jacobian.T @ (
            np.array([prior_t_weight] * 3 + [prior_r_weight] * 3)[:, None] * prior_jacobian
        )
    scale = np.array([_LEVER_M, _LEVER_M, _LEVER_M, 1.0, 1.0, 1.0])
    scaled = data_information / np.outer(scale, scale)
    eigenvalues, eigenvectors = np.linalg.eigh(scaled)
    largest = max(float(eigenvalues[-1]), 1e-12)
    weakest_axis = int(np.argmax(np.abs(eigenvectors[:, 0])))
    return RegistrationResult(
        world_from_camera=x,
        iterations=iteration,
        converged=converged,
        views=fits,
        inliers=int(residual_final.shape[0]),
        rms_mm=float(np.sqrt(np.mean(residual_final**2)) * 1e3)
        if residual_final.size
        else math.nan,
        plane_points=int(plane_all.shape[0]),
        plane_rms_mm=float(np.sqrt(np.mean(plane_all**2)) * 1e3) if plane_all.size else None,
        observability=tuple(float(max(value, 0.0) / largest) for value in eigenvalues),
        weakest=_AXIS_LABELS[weakest_axis],
    )


__all__ = [
    "RegistrationError",
    "RegistrationResult",
    "RegistrationView",
    "ViewFit",
    "evaluate_view",
    "register_views",
]
