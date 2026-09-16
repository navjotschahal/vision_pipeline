"""Shared-memory record layout for the phase-1 tabletop box viewer.

The estimator process (producer) and the viewer process (consumer) both import this
module, so the byte layout lives in exactly one place -- this file is the contract
between them (the same rule phase 2 applies to the camera/ROS boundary). Only this
fixed-size record crosses the process boundary, carried by the seqlock in
``shm_seqlock.py``; nothing here is pickled, and no image, depth, or cloud buffer is
serialized (design rule 3).

Point categories distinguish what each stage of ``tabletop.py`` decided, so a bad crop
or a bad plane fit is visible in the viewer rather than inferred from the final box:
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .shm_seqlock import SeqlockReader, SeqlockWriter

CHANNEL_NAME = "vision-pipeline-tabletop-box-view"

CATEGORY_OUTSIDE_WORKSPACE = 0
CATEGORY_PLANE = 1
CATEGORY_ABOVE_PLANE = 2
CATEGORY_CLUSTER = 3


def _record_dtype(width: int, height: int, max_points: int) -> np.dtype:
    return np.dtype(
        [
            ("frame_index", np.uint64),
            ("captured_at_ns", np.int64),
            ("host_written_at_ns", np.int64),
            ("point_count", np.uint32),
            ("has_observation", np.uint8),
            ("_reserved", np.uint8, 3),
            ("box_position_metres", np.float64, 3),
            ("box_orientation_xyzw", np.float64, 4),
            ("box_size_metres", np.float64, 3),
            ("cluster_point_count", np.uint32),
            ("plane_normal", np.float64, 3),
            ("plane_offset_metres", np.float64),
            ("estimator_frames_submitted", np.uint64),
            ("estimator_frames_failed", np.uint64),
            ("color_bgr", np.uint8, (height, width, 3)),
            ("points_xyz", np.float32, (max_points, 3)),
            ("points_rgb", np.uint8, (max_points, 3)),
            ("points_category", np.uint8, (max_points,)),
        ]
    )


@dataclass(frozen=True, slots=True)
class BoxViewGeometry:
    """Fixed capacities the channel is sized for; both sides must agree on these."""

    width: int
    height: int
    max_points: int

    def __post_init__(self) -> None:
        for name in ("width", "height", "max_points"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def dtype(self) -> np.dtype:
        return _record_dtype(self.width, self.height, self.max_points)

    @property
    def payload_size(self) -> int:
        return int(self.dtype.itemsize)


@dataclass(frozen=True, slots=True)
class BoxViewFrame:
    """One decoded snapshot read from the channel."""

    sequence: int
    frame_index: int
    captured_at_ns: int
    host_written_at_ns: int
    color_bgr: np.ndarray
    points_xyz: np.ndarray
    points_rgb: np.ndarray
    points_category: np.ndarray
    has_observation: bool
    box_position_metres: tuple[float, float, float]
    box_orientation_xyzw: tuple[float, float, float, float]
    box_size_metres: tuple[float, float, float]
    cluster_point_count: int
    plane_normal: tuple[float, float, float]
    plane_offset_metres: float
    estimator_frames_submitted: int
    estimator_frames_failed: int


class BoxViewWriter:
    """Producer side: publishes one full frame per call, never blocking on a reader."""

    def __init__(self, geometry: BoxViewGeometry, *, name: str = CHANNEL_NAME) -> None:
        if not isinstance(geometry, BoxViewGeometry):
            raise TypeError("geometry must be a BoxViewGeometry")
        self._geometry = geometry
        self._writer = SeqlockWriter(name, geometry.payload_size)
        self._record = np.zeros((), dtype=geometry.dtype)

    @property
    def name(self) -> str:
        return self._writer.name

    def publish(
        self,
        *,
        frame_index: int,
        captured_at_ns: int,
        host_written_at_ns: int,
        color_bgr: np.ndarray,
        points_xyz: np.ndarray,
        points_rgb: np.ndarray,
        points_category: np.ndarray,
        has_observation: bool,
        box_position_metres: tuple[float, float, float] = (0.0, 0.0, 0.0),
        box_orientation_xyzw: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
        box_size_metres: tuple[float, float, float] = (0.0, 0.0, 0.0),
        cluster_point_count: int = 0,
        plane_normal: tuple[float, float, float] = (0.0, 0.0, 0.0),
        plane_offset_metres: float = 0.0,
        estimator_frames_submitted: int = 0,
        estimator_frames_failed: int = 0,
    ) -> None:
        geometry = self._geometry
        point_count = int(points_xyz.shape[0])
        if point_count > geometry.max_points:
            raise ValueError(
                f"{point_count} points exceeds this channel's capacity of "
                f"{geometry.max_points}"
            )
        if color_bgr.shape != (geometry.height, geometry.width, 3):
            raise ValueError("color_bgr shape does not match the channel geometry")

        record = self._record
        record["frame_index"] = frame_index
        record["captured_at_ns"] = captured_at_ns
        record["host_written_at_ns"] = host_written_at_ns
        record["point_count"] = point_count
        record["has_observation"] = 1 if has_observation else 0
        record["box_position_metres"] = box_position_metres
        record["box_orientation_xyzw"] = box_orientation_xyzw
        record["box_size_metres"] = box_size_metres
        record["cluster_point_count"] = cluster_point_count
        record["plane_normal"] = plane_normal
        record["plane_offset_metres"] = plane_offset_metres
        record["estimator_frames_submitted"] = estimator_frames_submitted
        record["estimator_frames_failed"] = estimator_frames_failed
        record["color_bgr"] = color_bgr
        record["points_xyz"][:point_count] = points_xyz
        record["points_rgb"][:point_count] = points_rgb
        record["points_category"][:point_count] = points_category
        if point_count < geometry.max_points:
            record["points_xyz"][point_count:] = 0
            record["points_rgb"][point_count:] = 0
            record["points_category"][point_count:] = 0
        self._writer.write(record.tobytes())

    def close(self) -> None:
        self._writer.close()


class BoxViewReader:
    """Consumer side: reads the newest frame, tolerating a torn read by retrying."""

    def __init__(self, geometry: BoxViewGeometry, *, name: str = CHANNEL_NAME) -> None:
        if not isinstance(geometry, BoxViewGeometry):
            raise TypeError("geometry must be a BoxViewGeometry")
        self._geometry = geometry
        self._reader = SeqlockReader(name, geometry.payload_size)

    def read(self) -> BoxViewFrame | None:
        result = self._reader.read()
        if result is None:
            return None
        sequence, payload = result
        record = np.frombuffer(payload, dtype=self._geometry.dtype)[0]
        point_count = int(record["point_count"])
        return BoxViewFrame(
            sequence=sequence,
            frame_index=int(record["frame_index"]),
            captured_at_ns=int(record["captured_at_ns"]),
            host_written_at_ns=int(record["host_written_at_ns"]),
            color_bgr=np.array(record["color_bgr"]),
            points_xyz=np.array(record["points_xyz"][:point_count]),
            points_rgb=np.array(record["points_rgb"][:point_count]),
            points_category=np.array(record["points_category"][:point_count]),
            has_observation=bool(record["has_observation"]),
            box_position_metres=tuple(float(v) for v in record["box_position_metres"]),
            box_orientation_xyzw=tuple(float(v) for v in record["box_orientation_xyzw"]),
            box_size_metres=tuple(float(v) for v in record["box_size_metres"]),
            cluster_point_count=int(record["cluster_point_count"]),
            plane_normal=tuple(float(v) for v in record["plane_normal"]),
            plane_offset_metres=float(record["plane_offset_metres"]),
            estimator_frames_submitted=int(record["estimator_frames_submitted"]),
            estimator_frames_failed=int(record["estimator_frames_failed"]),
        )

    def close(self) -> None:
        self._reader.close()


def box_yaw_radians(orientation_xyzw: tuple[float, float, float, float]) -> float:
    """Yaw about +Z from a quaternion, for the operator-facing print/overlay."""

    x, y, z, w = orientation_xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


__all__ = [
    "CATEGORY_ABOVE_PLANE",
    "CATEGORY_CLUSTER",
    "CATEGORY_OUTSIDE_WORKSPACE",
    "CATEGORY_PLANE",
    "CHANNEL_NAME",
    "BoxViewFrame",
    "BoxViewGeometry",
    "BoxViewReader",
    "BoxViewWriter",
    "box_yaw_radians",
]
