"""Bounded voxelized world-map accumulation for live RGB-D experiments."""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float32Array = NDArray[np.float32]
UInt8Array = NDArray[np.uint8]
VoxelKey = tuple[int, int, int]
VoxelValue = tuple[float, float, float, float, float, float]


@dataclass(frozen=True, slots=True)
class VoxelMapConfig:
    voxel_size_metres: float = 0.02
    max_voxels: int = 250_000
    update_alpha: float = 0.25

    def __post_init__(self) -> None:
        if (
            isinstance(self.voxel_size_metres, bool)
            or not isinstance(self.voxel_size_metres, int | float)
            or not math.isfinite(self.voxel_size_metres)
            or self.voxel_size_metres <= 0
        ):
            raise ValueError("voxel_size_metres must be positive and finite")
        if (
            isinstance(self.max_voxels, bool)
            or not isinstance(self.max_voxels, int)
            or self.max_voxels <= 0
        ):
            raise ValueError("max_voxels must be a positive integer")
        if (
            isinstance(self.update_alpha, bool)
            or not isinstance(self.update_alpha, int | float)
            or not math.isfinite(self.update_alpha)
            or not 0 < self.update_alpha <= 1
        ):
            raise ValueError("update_alpha must be in (0, 1]")


class VoxelWorldMap:
    """Fuse colored world points while bounding spatial density and memory."""

    def __init__(self, config: VoxelMapConfig) -> None:
        self.config = config
        self._voxels: OrderedDict[VoxelKey, VoxelValue] = OrderedDict()

    def clear(self) -> None:
        self._voxels.clear()

    def __len__(self) -> int:
        return len(self._voxels)

    def integrate(self, xyz_world: Float32Array, colors_rgb: UInt8Array) -> int:
        """Integrate one cloud and return the number of finite sampled voxels."""

        xyz = np.asarray(xyz_world, dtype=np.float32)
        colors = np.asarray(colors_rgb, dtype=np.uint8)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz_world must have shape [N, 3]")
        if colors.shape != xyz.shape:
            raise ValueError("colors_rgb must have shape [N, 3]")
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        colors = colors[finite]
        if len(xyz) == 0:
            return 0

        voxel_indices = np.floor(xyz / self.config.voxel_size_metres).astype(np.int64)
        # One representative per voxel per frame prevents dense near-camera areas
        # from receiving a disproportionate number of updates.
        _, representatives = np.unique(voxel_indices, axis=0, return_index=True)
        alpha = float(self.config.update_alpha)
        inverse_alpha = 1.0 - alpha
        for index in representatives:
            key_array = voxel_indices[index]
            key = (int(key_array[0]), int(key_array[1]), int(key_array[2]))
            point = xyz[index]
            color = colors[index]
            previous = self._voxels.get(key)
            if previous is None:
                value: VoxelValue = (
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    float(color[0]),
                    float(color[1]),
                    float(color[2]),
                )
            else:
                value = (
                    inverse_alpha * previous[0] + alpha * float(point[0]),
                    inverse_alpha * previous[1] + alpha * float(point[1]),
                    inverse_alpha * previous[2] + alpha * float(point[2]),
                    inverse_alpha * previous[3] + alpha * float(color[0]),
                    inverse_alpha * previous[4] + alpha * float(color[1]),
                    inverse_alpha * previous[5] + alpha * float(color[2]),
                )
                self._voxels.move_to_end(key)
            self._voxels[key] = value

        overflow = len(self._voxels) - self.config.max_voxels
        for _ in range(max(0, overflow)):
            self._voxels.popitem(last=False)
        return len(representatives)

    def snapshot(self) -> tuple[Float32Array, UInt8Array]:
        if not self._voxels:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint8),
            )
        values = np.asarray(tuple(self._voxels.values()), dtype=np.float32)
        xyz = np.ascontiguousarray(values[:, :3], dtype=np.float32)
        colors = np.ascontiguousarray(np.clip(values[:, 3:], 0, 255), dtype=np.uint8)
        return xyz, colors


__all__ = ["VoxelMapConfig", "VoxelWorldMap"]
