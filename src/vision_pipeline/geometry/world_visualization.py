"""OpenCV rendering of an accumulated colored world point cloud."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class WorldCloudViewConfig:
    width: int = 1440
    height: int = 900
    point_size: int = 2

    def __post_init__(self) -> None:
        for name in ("width", "height", "point_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class WorldCloudRenderer:
    """Render a world map from an interactive orthographic viewpoint."""

    def __init__(self, config: WorldCloudViewConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.yaw_radians = math.radians(-25.0)
        self.pitch_radians = math.radians(20.0)
        self.roll_radians = 0.0
        self.zoom = 1.0

    def handle_key(self, key: int) -> bool:
        """Apply a keyboard command and return True when the map should clear."""

        if key == ord("j"):
            self.yaw_radians -= math.radians(5)
        elif key == ord("l"):
            self.yaw_radians += math.radians(5)
        elif key == ord("i"):
            self.pitch_radians += math.radians(5)
        elif key == ord("k"):
            self.pitch_radians -= math.radians(5)
        elif key == ord("u"):
            self.roll_radians -= math.radians(5)
        elif key == ord("o"):
            self.roll_radians += math.radians(5)
        elif key in (ord("+"), ord("=")):
            self.zoom = min(8.0, self.zoom * 1.15)
        elif key in (ord("-"), ord("_")):
            self.zoom = max(0.1, self.zoom / 1.15)
        elif key == ord("r"):
            self.reset()
        return key == ord("c")

    def render(
        self,
        xyz_world: NDArray[np.float32],
        colors_rgb: NDArray[np.uint8],
        *,
        status_lines: tuple[str, ...] = (),
        title: str = "WORLD CLOUD",
        vertical_axis_up: bool = True,
    ) -> NDArray[np.uint8]:
        xyz = np.asarray(xyz_world, dtype=np.float32)
        colors = np.asarray(colors_rgb, dtype=np.uint8)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz_world must have shape [N, 3]")
        if colors.shape != xyz.shape:
            raise ValueError("colors_rgb must have shape [N, 3]")
        canvas = np.full(
            (self.config.height, self.config.width, 3), 18, dtype=np.uint8
        )
        if len(xyz):
            center = np.median(xyz, axis=0)
            rotated = (xyz - center) @ self._rotation().T
            ranges = np.percentile(rotated, (2, 98), axis=0)
            extent = max(
                float(ranges[1, 0] - ranges[0, 0]),
                float(ranges[1, 1] - ranges[0, 1]),
                0.25,
            )
            scale = 0.78 * min(self.config.width, self.config.height) / extent * self.zoom
            pixel_x = np.rint(self.config.width / 2 + rotated[:, 0] * scale).astype(
                np.int32
            )
            vertical_sign = -1.0 if vertical_axis_up else 1.0
            pixel_y = np.rint(
                self.config.height / 2 + vertical_sign * rotated[:, 1] * scale
            ).astype(np.int32)
            inside = (
                (pixel_x >= 0)
                & (pixel_x < self.config.width)
                & (pixel_y >= 0)
                & (pixel_y < self.config.height)
            )
            visible = np.flatnonzero(inside)
            order = visible[np.argsort(rotated[visible, 2])[::-1]]
            canvas[pixel_y[order], pixel_x[order]] = colors[order, ::-1]
            if self.config.point_size > 1:
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (self.config.point_size, self.config.point_size),
                )
                canvas = cast(NDArray[np.uint8], cv2.dilate(canvas, kernel))

        lines = (
            f"{title}  points={len(xyz)}",
            *status_lines,
            "J/L yaw  I/K pitch  U/O roll  +/- zoom  R view reset  C clear  Q quit",
        )
        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (18, 34 + index * 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
        return canvas

    def _rotation(self) -> NDArray[np.float32]:
        cy, sy = math.cos(self.yaw_radians), math.sin(self.yaw_radians)
        cp, sp = math.cos(self.pitch_radians), math.sin(self.pitch_radians)
        cr, sr = math.cos(self.roll_radians), math.sin(self.roll_radians)
        yaw = np.array(((cy, 0, sy), (0, 1, 0), (-sy, 0, cy)), dtype=np.float32)
        pitch = np.array(((1, 0, 0), (0, cp, -sp), (0, sp, cp)), dtype=np.float32)
        roll = np.array(((cr, -sr, 0), (sr, cr, 0), (0, 0, 1)), dtype=np.float32)
        return cast(NDArray[np.float32], roll @ pitch @ yaw)


__all__ = ["WorldCloudRenderer", "WorldCloudViewConfig"]
