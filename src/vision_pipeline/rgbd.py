"""Vendor-neutral payloads for depth registered to a color measurement."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import MeasurementKind, SensorSample
from vision_pipeline.geometry.camera import PinholeIntrinsics
from vision_pipeline.image import ImageFrame


@dataclass(frozen=True, slots=True)
class DepthFrame:
    """A host-readable metric depth plane.

    Values are metres along the camera optical Z axis. Zero and non-finite values are
    invalid measurements; consumers choose the range and confidence policy appropriate
    for their task. The dataclass does not imply ownership, so source adapters must copy
    SDK-owned storage before advertising host-owned memory.
    """

    data: NDArray[np.float32]

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if self.data.dtype != np.dtype(np.float32):
            raise ValueError("metric depth data must use float32 elements")
        if self.data.ndim != 2:
            raise ValueError("metric depth data must have HxW shape")
        if self.data.shape[0] <= 0 or self.data.shape[1] <= 0:
            raise ValueError("depth dimensions must be positive")

    @property
    def width(self) -> int:
        return int(self.data.shape[1])

    @property
    def height(self) -> int:
        return int(self.data.shape[0])

    @property
    def row_stride_bytes(self) -> int:
        return int(self.data.strides[0])

    @property
    def size_bytes(self) -> int:
        return int(self.data.nbytes)


def _intrinsics_match(left: PinholeIntrinsics, right: PinholeIntrinsics) -> bool:
    if (left.width, left.height) != (right.width, right.height):
        return False
    return all(
        math.isclose(left_value, right_value, rel_tol=1e-7, abs_tol=1e-9)
        for left_value, right_value in (
            (left.fx, right.fx),
            (left.fy, right.fy),
            (left.cx, right.cx),
            (left.cy, right.cy),
        )
    )


@dataclass(frozen=True, slots=True)
class AlignedRgbdFrame:
    """One SDK-synchronized color/depth pair registered to color pixel geometry.

    Color and depth retain separate headers because their frame numbers and acquisition
    timestamps can differ. Registration only establishes a common optical coordinate
    frame and pixel geometry; it does not claim simultaneous exposure.
    """

    frameset_id: str
    color: SensorSample[ImageFrame]
    depth: SensorSample[DepthFrame]
    color_intrinsics: PinholeIntrinsics
    aligned_depth_intrinsics: PinholeIntrinsics
    synchronization_method: str
    alignment_method: str

    def __post_init__(self) -> None:
        if not isinstance(self.frameset_id, str) or not self.frameset_id.strip():
            raise ValueError("frameset_id must be a non-empty string")
        if self.color.header.measurement_kind is not MeasurementKind.RGB_IMAGE:
            raise ValueError("color must carry an RGB_IMAGE header")
        if self.depth.header.measurement_kind is not MeasurementKind.DEPTH_IMAGE:
            raise ValueError("depth must carry a DEPTH_IMAGE header")
        if not isinstance(self.color.payload, ImageFrame):
            raise TypeError("color payload must be an ImageFrame")
        if not isinstance(self.depth.payload, DepthFrame):
            raise TypeError("depth payload must be a DepthFrame")
        if not isinstance(self.color_intrinsics, PinholeIntrinsics):
            raise TypeError("color_intrinsics must be PinholeIntrinsics")
        if not isinstance(self.aligned_depth_intrinsics, PinholeIntrinsics):
            raise TypeError("aligned_depth_intrinsics must be PinholeIntrinsics")
        color_size = (self.color.payload.width, self.color.payload.height)
        depth_size = (self.depth.payload.width, self.depth.payload.height)
        if color_size != depth_size:
            raise ValueError("aligned color and depth dimensions must match")
        if color_size != (self.color_intrinsics.width, self.color_intrinsics.height):
            raise ValueError("color dimensions must match color intrinsics")
        if depth_size != (
            self.aligned_depth_intrinsics.width,
            self.aligned_depth_intrinsics.height,
        ):
            raise ValueError("depth dimensions must match aligned-depth intrinsics")
        if not _intrinsics_match(self.color_intrinsics, self.aligned_depth_intrinsics):
            raise ValueError("aligned depth must use the color camera's pixel geometry")
        if self.color.header.frame_id != self.depth.header.frame_id:
            raise ValueError("aligned color and depth must use the same optical frame")
        for value, name in (
            (self.synchronization_method, "synchronization_method"),
            (self.alignment_method, "alignment_method"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")


__all__ = ["AlignedRgbdFrame", "DepthFrame"]
