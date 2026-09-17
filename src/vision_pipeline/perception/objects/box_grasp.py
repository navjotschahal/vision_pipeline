"""Gravity-aligned box estimate in the robot world frame, for CPF's opposed grasp.

CPF's grasp planner (``cpf/src/openarm_hw/tools/grasp_planner.py``) needs only the box
centre ``object_pos`` in its world frame (``openarm_body_link0``: x forward, y left, z up)
and ``half_width``, the half-extent along world y, which is the axis the two hands squeeze
along (``OpposedGraspTargets(axis=1)``). It does not model box yaw, so a box must face
the squeeze axis; this module measures the yaw and flags a misaligned box instead of
silently returning a width that would be wrong for a rotated one.

The fit uses the selected object's metric cloud expressed in the world frame:

- **up** is world z (gravity), not PCA, so a near-cubic box cannot tip the axes;
- **bottom** is the support plane height under the box when a plane was accepted,
  otherwise a low percentile of the points;
- **top** is a high percentile of point heights;
- **footprint** is the minimum-area rectangle of the points above the bottom, projected
  onto the horizontal plane, after trimming the outer 1 % along its principal axes. With
  the top face in view this recovers the far edges the side faces cannot show.

Single-view limits are reported rather than hidden: ``top_points`` counts the points near
the top face, and a camera that cannot see the top leaves the far half of the footprint
unobserved.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import TimePoint

from .selected_geometry import SelectedObjectGeometry
from .tabletop import SupportPlane

Matrix = NDArray[np.float64]


class BoxFitError(ValueError):
    """The selected cloud cannot support a box estimate on this frame."""


@dataclass(frozen=True, slots=True)
class WorldBoxEstimate:
    """One frame's box in the world frame. Axis ``a`` is the footprint side nearest world x."""

    frameset_id: str
    selection_id: int
    depth_captured_at: TimePoint | None
    centre_world: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    yaw_deg: float
    top_z: float
    bottom_z: float
    bottom_from_support_plane: bool
    support_tilt_deg: float | None
    point_count: int
    top_points: int

    @property
    def half_width_along_y(self) -> float:
        """CPF ``half_width``: the half-extent of the box side facing world y."""

        return self.half_extents[1]


def _plane_in_world(
    plane: SupportPlane, world_from_camera: Matrix
) -> tuple[NDArray[np.float64], float]:
    rotation = world_from_camera[:3, :3]
    translation = world_from_camera[:3, 3]
    normal = rotation @ np.asarray(plane.normal, dtype=np.float64)
    return normal, float(plane.offset_metres - normal @ translation)


def fit_world_box(
    geometry: SelectedObjectGeometry,
    world_from_camera: Matrix,
    *,
    min_height_above_support: float = 0.01,
    top_band_metres: float = 0.015,
    minimum_points: int = 50,
) -> WorldBoxEstimate:
    """Fit the box to a selected object's cloud, expressed in the world frame first."""

    xyz_camera = geometry.cloud.xyz.detach().to("cpu").numpy().astype(np.float64)
    points = xyz_camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    plane = (
        None
        if geometry.support_plane is None
        else _plane_in_world(geometry.support_plane, world_from_camera)
    )
    return fit_box_to_world_points(
        points,
        support_plane_world=plane,
        frameset_id=geometry.frameset_id,
        selection_id=geometry.selection_id,
        depth_captured_at=geometry.depth_source.captured_at,
        min_height_above_support=min_height_above_support,
        top_band_metres=top_band_metres,
        minimum_points=minimum_points,
    )


def fit_box_to_world_points(
    points: NDArray[np.float64],
    *,
    support_plane_world: tuple[NDArray[np.float64], float] | None,
    frameset_id: str,
    selection_id: int,
    depth_captured_at: TimePoint | None,
    min_height_above_support: float = 0.01,
    top_band_metres: float = 0.015,
    minimum_points: int = 50,
) -> WorldBoxEstimate:
    """Gravity-aligned box from world-frame points (z up) and an optional support plane.

    ``support_plane_world`` is ``(normal, offset)`` with ``normal . p + offset = 0``.
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have [N, 3] shape")
    if points.shape[0] < minimum_points:
        raise BoxFitError(f"{points.shape[0]} points; at least {minimum_points} required")
    heights = points[:, 2]
    top_z = float(np.percentile(heights, 97))

    support_tilt: float | None = None
    bottom_from_plane = False
    bottom_z = float(np.percentile(heights, 2))
    if support_plane_world is not None:
        normal, offset = support_plane_world
        support_tilt = math.degrees(math.acos(min(1.0, abs(float(normal[2])))))
        if abs(normal[2]) > math.cos(math.radians(30.0)):
            centre_xy = np.median(points[:, :2], axis=0)
            plane_z = -(offset + normal[0] * centre_xy[0] + normal[1] * centre_xy[1]) / normal[2]
            if plane_z < top_z:
                bottom_z = float(plane_z)
                bottom_from_plane = True

    above = points[heights > bottom_z + min_height_above_support]
    if above.shape[0] < minimum_points:
        raise BoxFitError("too few points above the support to fit a footprint")
    top_face = points[heights > top_z - top_band_metres]
    # The top face is sampled uniformly over the whole footprint when it is in view; side
    # faces only add dense lines along the visible edges, which bias the rectangle's angle.
    footprint = (top_face if top_face.shape[0] >= minimum_points else above)[:, :2]
    (cx, cy), (width, height), angle = cv2.minAreaRect(footprint.astype(np.float32))

    # OpenCV's angle is for the `width` side. Pick whichever side lies nearest world x.
    side_angle = math.radians(angle)
    candidates = [(side_angle, width, height), (side_angle + math.pi / 2, height, width)]
    best = min(candidates, key=lambda item: abs(math.sin(item[0])))
    yaw = math.atan2(math.sin(best[0]), math.cos(best[0]))
    if yaw > math.pi / 2:
        yaw -= math.pi
    elif yaw < -math.pi / 2:
        yaw += math.pi
    half_a, half_b = best[1] / 2.0, best[2] / 2.0
    half_z = (top_z - bottom_z) / 2.0
    if min(half_a, half_b, half_z) <= 0:
        raise BoxFitError("degenerate box extents")
    top_points = int(top_face.shape[0])
    return WorldBoxEstimate(
        frameset_id=frameset_id,
        selection_id=selection_id,
        depth_captured_at=depth_captured_at,
        centre_world=(float(cx), float(cy), (top_z + bottom_z) / 2.0),
        half_extents=(float(half_a), float(half_b), float(half_z)),
        yaw_deg=math.degrees(yaw),
        top_z=top_z,
        bottom_z=bottom_z,
        bottom_from_support_plane=bottom_from_plane,
        support_tilt_deg=support_tilt,
        point_count=int(points.shape[0]),
        top_points=top_points,
    )


def box_corners_world(estimate: WorldBoxEstimate) -> NDArray[np.float64]:
    """The 8 corners, ordered like ``select_object._BOX_EDGES`` (x, then y, then z sign)."""

    yaw = math.radians(estimate.yaw_deg)
    rotation = np.array(
        [[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0, 0, 1.0]]
    )
    ha, hb, hz = estimate.half_extents
    local = np.array([(sx, sy, sz) for sx in (-ha, ha) for sy in (-hb, hb) for sz in (-hz, hz)])
    return np.asarray(local @ rotation.T + np.asarray(estimate.centre_world), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class StableBox:
    """Median over a window of per-frame estimates, with its spread."""

    selection_id: int
    samples: int
    centre_world: tuple[float, float, float]
    half_extents: tuple[float, float, float]
    yaw_deg: float
    centre_std_mm: tuple[float, float, float]
    half_width_std_mm: float
    yaw_std_deg: float
    bottom_from_support_plane_fraction: float
    support_tilt_deg: float | None
    last_frameset_id: str


@dataclass(slots=True)
class BoxEstimateWindow:
    """Rolling window of estimates for one selection; resets when the selection changes."""

    size: int = 30
    _items: deque[WorldBoxEstimate] = field(default_factory=deque)

    def add(self, estimate: WorldBoxEstimate) -> None:
        if self._items and self._items[-1].selection_id != estimate.selection_id:
            self._items.clear()
        self._items.append(estimate)
        while len(self._items) > self.size:
            self._items.popleft()

    def clear(self) -> None:
        self._items.clear()

    def summary(self) -> StableBox | None:
        if not self._items:
            return None
        items = list(self._items)
        centres = np.array([item.centre_world for item in items])
        extents = np.array([item.half_extents for item in items])
        yaws = np.array([item.yaw_deg for item in items])
        tilts = [item.support_tilt_deg for item in items if item.support_tilt_deg is not None]
        centre = np.median(centres, axis=0)
        extent = np.median(extents, axis=0)
        return StableBox(
            selection_id=items[-1].selection_id,
            samples=len(items),
            centre_world=(float(centre[0]), float(centre[1]), float(centre[2])),
            half_extents=(float(extent[0]), float(extent[1]), float(extent[2])),
            yaw_deg=float(np.median(yaws)),
            centre_std_mm=tuple(float(v) for v in centres.std(axis=0) * 1e3),  # type: ignore[arg-type]
            half_width_std_mm=float(extents[:, 1].std() * 1e3),
            yaw_std_deg=float(yaws.std()),
            bottom_from_support_plane_fraction=float(
                np.mean([item.bottom_from_support_plane for item in items])
            ),
            support_tilt_deg=float(np.median(tilts)) if tilts else None,
            last_frameset_id=items[-1].frameset_id,
        )


@dataclass(frozen=True, slots=True)
class CpfGraspLimits:
    """Planner defaults from grasp_planner.py (checked 2026-09-16); override to match your run."""

    palm_offset: float = 0.093
    workspace_min: tuple[float, float, float] = (0.05, -0.60, 0.05)
    workspace_max: tuple[float, float, float] = (0.75, 0.60, 0.85)
    max_yaw_deg: float = 8.0
    max_centre_std_mm: float = 5.0
    minimum_samples: int = 15


def cpf_handoff(
    box: StableBox, limits: CpfGraspLimits, *, extrinsic: dict[str, Any]
) -> dict[str, Any]:
    """The record CPF's planner needs, with every reason it should not be trusted yet."""

    x, y, z = box.centre_world
    half_width = box.half_extents[1]
    reach = half_width + limits.palm_offset
    attractors = {"left": (x, y + reach, z), "right": (x, y - reach, z)}
    problems: list[str] = []
    if abs(box.yaw_deg) > limits.max_yaw_deg:
        problems.append(
            f"box yaw {box.yaw_deg:+.1f} deg exceeds {limits.max_yaw_deg} deg; CPF squeezes along "
            "world y only -- rotate the box square to the robot"
        )
    if box.samples < limits.minimum_samples:
        problems.append(f"only {box.samples} samples; wait for {limits.minimum_samples}")
    if max(box.centre_std_mm) > limits.max_centre_std_mm:
        problems.append(f"centre spread {max(box.centre_std_mm):.1f} mm; hold the scene still")
    low, high = np.asarray(limits.workspace_min), np.asarray(limits.workspace_max)
    for side, point in attractors.items():
        if not np.all((np.asarray(point) >= low) & (np.asarray(point) <= high)):
            problems.append(
                f"{side} attractor {np.round(point, 3).tolist()} is outside the planner workspace"
            )
    return {
        "schema": "cpf-box-grasp-handoff-v1",
        "world_frame": extrinsic.get("world_frame"),
        "object_pos": [x, y, z],
        "half_width": half_width,
        "box_half_extents": list(box.half_extents),
        "yaw_deg": box.yaw_deg,
        "palm_offset": limits.palm_offset,
        "attractors_before_squeeze": {side: list(point) for side, point in attractors.items()},
        "samples": box.samples,
        "centre_std_mm": list(box.centre_std_mm),
        "half_width_std_mm": box.half_width_std_mm,
        "support_tilt_deg": box.support_tilt_deg,
        "last_frameset_id": box.last_frameset_id,
        "ready": not problems,
        "problems": problems,
        "extrinsic": extrinsic,
        "grasp_planner_args": planner_arguments(box),
    }


def planner_arguments(box: StableBox) -> list[str]:
    x, y, z = box.centre_world
    return [
        "--object-pos",
        f"{x:.4f}",
        f"{y:.4f}",
        f"{z:.4f}",
        "--half-width",
        f"{box.half_extents[1]:.4f}",
    ]


__all__ = [
    "BoxEstimateWindow",
    "BoxFitError",
    "CpfGraspLimits",
    "StableBox",
    "WorldBoxEstimate",
    "box_corners_world",
    "cpf_handoff",
    "fit_box_to_world_points",
    "fit_world_box",
    "planner_arguments",
]
