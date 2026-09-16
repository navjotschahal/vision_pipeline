"""A lightweight OpenCV point-cloud viewer for hands-on experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from .pointcloud import PointCloud


@dataclass(frozen=True, slots=True)
class PointCloudViewConfig:
    width: int = 1280
    height: int = 720
    point_size: int = 2

    def __post_init__(self) -> None:
        for name in ("width", "height", "point_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


class PointCloudRenderer:
    """Render a colored cloud from a controllable virtual orthographic view."""

    def __init__(self, config: PointCloudViewConfig) -> None:
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.yaw_radians = math.radians(-18.0)
        self.pitch_radians = math.radians(12.0)
        self.roll_radians = 0.0
        self.zoom = 1.0

    def rotate(
        self,
        *,
        yaw_degrees: float = 0.0,
        pitch_degrees: float = 0.0,
        roll_degrees: float = 0.0,
    ) -> None:
        self.yaw_radians += math.radians(yaw_degrees)
        self.pitch_radians = min(
            math.radians(85),
            max(math.radians(-85), self.pitch_radians + math.radians(pitch_degrees)),
        )
        self.roll_radians = (self.roll_radians + math.radians(roll_degrees)) % (2 * math.pi)

    def change_zoom(self, factor: float) -> None:
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError("zoom factor must be positive and finite")
        self.zoom = min(8.0, max(0.15, self.zoom * factor))

    def render(
        self,
        cloud: PointCloud[torch.Tensor],
        *,
        title: str = "MONOCULAR PSEUDO-CLOUD - learned depth",
        extra_status_lines: tuple[str, ...] = (),
    ) -> NDArray[np.uint8]:
        xyz = cloud.xyz.detach().to(device="cpu", dtype=torch.float32).numpy()
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("point cloud XYZ values must have [N, 3] shape")
        canvas = np.full(
            (self.config.height, self.config.width, 3),
            (18, 18, 18),
            dtype=np.uint8,
        )
        if xyz.shape[0] == 0:
            self._draw_status(canvas, cloud, 0, title, extra_status_lines)
            return canvas

        center = np.median(xyz, axis=0)
        centered = xyz - center
        rotation = self._rotation()
        rotated = centered @ rotation.T

        ranges = np.percentile(rotated, (2, 98), axis=0)
        visible_extent = max(
            float(ranges[1, 0] - ranges[0, 0]),
            float(ranges[1, 1] - ranges[0, 1]),
            float(ranges[1, 2] - ranges[0, 2]) * 0.35,
            1e-6,
        )
        scale = 0.82 * min(self.config.width, self.config.height) / visible_extent * self.zoom
        pixel_x = np.rint(self.config.width / 2 + rotated[:, 0] * scale).astype(np.int32)
        pixel_y = np.rint(self.config.height / 2 + rotated[:, 1] * scale).astype(np.int32)
        inside = (
            (pixel_x >= 0)
            & (pixel_x < self.config.width)
            & (pixel_y >= 0)
            & (pixel_y < self.config.height)
        )
        indices = np.flatnonzero(inside)
        # The camera looks along +Z: paint far points first and near points last.
        ordering = indices[np.argsort(rotated[indices, 2])[::-1]]
        colors = self._bgr_colors(cloud, xyz.shape[0])
        canvas[pixel_y[ordering], pixel_x[ordering]] = colors[ordering]
        if self.config.point_size > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (self.config.point_size, self.config.point_size),
            )
            canvas = cast(NDArray[np.uint8], cv2.dilate(canvas, kernel))

        self._draw_axes(canvas, rotation, scale, visible_extent)
        self._draw_status(canvas, cloud, len(ordering), title, extra_status_lines)
        return canvas

    def _rotation(self) -> NDArray[np.float32]:
        yaw_cos = math.cos(self.yaw_radians)
        yaw_sin = math.sin(self.yaw_radians)
        pitch_cos = math.cos(self.pitch_radians)
        pitch_sin = math.sin(self.pitch_radians)
        roll_cos = math.cos(self.roll_radians)
        roll_sin = math.sin(self.roll_radians)
        yaw = np.array(
            ((yaw_cos, 0, yaw_sin), (0, 1, 0), (-yaw_sin, 0, yaw_cos)),
            dtype=np.float32,
        )
        pitch = np.array(
            ((1, 0, 0), (0, pitch_cos, -pitch_sin), (0, pitch_sin, pitch_cos)),
            dtype=np.float32,
        )
        roll = np.array(
            ((roll_cos, -roll_sin, 0), (roll_sin, roll_cos, 0), (0, 0, 1)),
            dtype=np.float32,
        )
        return cast(NDArray[np.float32], roll @ pitch @ yaw)

    @staticmethod
    def _bgr_colors(cloud: PointCloud[torch.Tensor], count: int) -> NDArray[np.uint8]:
        if cloud.colors_rgb is None:
            return np.full((count, 3), (220, 220, 220), dtype=np.uint8)
        rgb = cloud.colors_rgb.detach().to(device="cpu", dtype=torch.uint8).numpy()
        if rgb.shape != (count, 3):
            raise ValueError("point colors must have [N, 3] shape")
        return np.ascontiguousarray(rgb[:, ::-1])

    def _draw_axes(
        self,
        canvas: NDArray[np.uint8],
        rotation: NDArray[np.float32],
        scale: float,
        visible_extent: float,
    ) -> None:
        origin = np.array((90.0, self.config.height - 90.0))
        length = visible_extent * 0.13
        axes = np.eye(3, dtype=np.float32) * length
        projected = axes @ rotation.T
        colors = ((0, 0, 255), (0, 255, 0), (255, 100, 0))
        labels = ("X", "Y", "Z")
        origin_px = (round(origin[0]), round(origin[1]))
        for axis, color, label in zip(projected, colors, labels, strict=True):
            end = (
                round(origin[0] + axis[0] * scale),
                round(origin[1] + axis[1] * scale),
            )
            cv2.arrowedLine(canvas, origin_px, end, color, 2, cv2.LINE_AA, tipLength=0.15)
            cv2.putText(
                canvas,
                label,
                end,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    def _draw_status(
        self,
        canvas: NDArray[np.uint8],
        cloud: PointCloud[torch.Tensor],
        visible_count: int,
        title: str,
        extra_status_lines: tuple[str, ...],
    ) -> None:
        intrinsics_kind = (
            f"calibrated:{cloud.intrinsics.calibration_id}"
            if cloud.intrinsics.calibration_id is not None
            else "approximate-FOV"
        )
        lines = (
            f"{title}; intrinsics={intrinsics_kind}",
            f"points={cloud.xyz.shape[0]} visible={visible_count} "
            f"stride={cloud.sampling_stride} geometry={cloud.geometry_kind.value}",
            f"view yaw={math.degrees(self.yaw_radians):.0f} "
            f"pitch={math.degrees(self.pitch_radians):.0f} "
            f"roll={math.degrees(self.roll_radians):.0f} zoom={self.zoom:.2f}",
            *extra_status_lines,
            "controls: J/L yaw  I/K pitch  U/O roll  +/- zoom  R reset  Q quit",
        )
        for index, line in enumerate(lines):
            cv2.putText(
                canvas,
                line,
                (18, 32 + index * 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (240, 240, 240),
                2,
                cv2.LINE_AA,
            )


__all__ = ["PointCloudRenderer", "PointCloudViewConfig"]
