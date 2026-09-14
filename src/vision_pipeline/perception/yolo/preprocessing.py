"""Explicit image preprocessing for YOLO detection models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LetterboxTransform:
    """Geometry needed to invert preprocessing coordinates."""

    scale: float
    pad_left: int
    pad_top: int
    input_width: int
    input_height: int


def letterbox_bgr(
    image: NDArray[np.uint8],
    image_size: int,
    stride: int,
) -> tuple[NDArray[np.uint8], LetterboxTransform]:
    """Resize without distortion and add minimal stride-aligned padding."""

    source_height, source_width = image.shape[:2]
    scale = min(image_size / source_height, image_size / source_width)
    resized_width = round(source_width * scale)
    resized_height = round(source_height * scale)

    pad_width = (image_size - resized_width) % stride
    pad_height = (image_size - resized_height) % stride
    left = round(pad_width / 2 - 0.1)
    right = round(pad_width / 2 + 0.1)
    top = round(pad_height / 2 - 0.1)
    bottom = round(pad_height / 2 + 0.1)

    resized = (
        cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        if (resized_width, resized_height) != (source_width, source_height)
        else image
    )
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    transform = LetterboxTransform(
        scale=scale,
        pad_left=left,
        pad_top=top,
        input_width=int(padded.shape[1]),
        input_height=int(padded.shape[0]),
    )
    return cast(NDArray[np.uint8], padded), transform


def preprocess_bgr(
    image: NDArray[np.uint8],
    image_size: int,
    stride: int,
    device: torch.device,
) -> tuple[torch.Tensor, LetterboxTransform]:
    """Convert a host BGR8 HWC image to normalized RGB NCHW model input."""

    padded, transform = letterbox_bgr(image, image_size, stride)
    rgb_chw = np.ascontiguousarray(padded[..., ::-1].transpose(2, 0, 1))
    tensor = torch.from_numpy(rgb_chw).to(device=device, dtype=torch.float32)
    return tensor.unsqueeze(0).div_(255.0), transform
