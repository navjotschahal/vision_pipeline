"""Typed image payloads used by camera sources and perception capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray


class PixelFormat(StrEnum):
    """In-memory pixel layout, including channel order and element width."""

    BGR8 = "bgr8"
    RGB8 = "rgb8"


@dataclass(frozen=True, slots=True)
class ImageFrame:
    """A zero-copy view of an interleaved 8-bit, three-channel image.

    The dataclass is immutable, but NumPy owns the underlying memory. Consumers must not
    mutate it unless a later ownership contract explicitly allows that operation.
    """

    data: NDArray[np.uint8]
    pixel_format: PixelFormat

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("data must be a numpy.ndarray")
        if not isinstance(self.pixel_format, PixelFormat):
            raise TypeError("pixel_format must be a PixelFormat")
        if self.data.dtype != np.dtype(np.uint8):
            raise ValueError("RGB/BGR image data must use uint8 elements")
        if self.data.ndim != 3 or self.data.shape[2] != 3:
            raise ValueError("RGB/BGR image data must have HxWx3 shape")
        if self.data.shape[0] <= 0 or self.data.shape[1] <= 0:
            raise ValueError("image dimensions must be positive")

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
