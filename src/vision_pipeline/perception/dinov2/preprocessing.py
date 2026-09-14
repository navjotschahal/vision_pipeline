"""Geometry-preserving preprocessing for DINOv2 image backbones."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from vision_pipeline.image import ImageFrame, PixelFormat

IMAGENET_MEAN = np.array((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.array((0.229, 0.224, 0.225), dtype=np.float32)


@dataclass(frozen=True, slots=True)
class DinoV2Transform:
    """Geometry required to map a patch grid back into source-image pixels."""

    source_width: int
    source_height: int
    input_width: int
    input_height: int
    content_left: int
    content_top: int
    content_width: int
    content_height: int
    patch_width: int
    patch_height: int

    @property
    def grid_width(self) -> int:
        return self.input_width // self.patch_width

    @property
    def grid_height(self) -> int:
        return self.input_height // self.patch_height


def preprocess_image(
    image: ImageFrame,
    long_edge: int,
    patch_size: tuple[int, int],
    device: torch.device,
) -> tuple[torch.Tensor, DinoV2Transform]:
    """Produce RGB ImageNet-normalized NCHW input without aspect distortion.

    Padding is the ImageNet mean, which becomes zero after normalization. Only enough
    padding to reach a whole patch is added; camera and model resolutions stay separate.
    """

    patch_height, patch_width = patch_size
    if long_edge <= 0 or patch_width <= 0 or patch_height <= 0:
        raise ValueError("long edge and patch dimensions must be positive")

    source_width = image.width
    source_height = image.height
    scale = long_edge / max(source_width, source_height)
    content_width = max(1, round(source_width * scale))
    content_height = max(1, round(source_height * scale))

    resized = (
        cv2.resize(
            image.data,
            (content_width, content_height),
            interpolation=cv2.INTER_LINEAR,
        )
        if (content_width, content_height) != (source_width, source_height)
        else image.data
    )
    rgb = resized[..., ::-1] if image.pixel_format is PixelFormat.BGR8 else resized
    normalized = rgb.astype(np.float32) / 255.0
    normalized = (normalized - IMAGENET_MEAN) / IMAGENET_STD

    input_width = math.ceil(content_width / patch_width) * patch_width
    input_height = math.ceil(content_height / patch_height) * patch_height
    content_left = (input_width - content_width) // 2
    content_top = (input_height - content_height) // 2
    canvas = np.zeros((input_height, input_width, 3), dtype=np.float32)
    canvas[
        content_top : content_top + content_height,
        content_left : content_left + content_width,
    ] = normalized

    chw = np.ascontiguousarray(canvas.transpose(2, 0, 1))
    tensor = torch.from_numpy(chw).unsqueeze(0).to(device=device)
    transform = DinoV2Transform(
        source_width=source_width,
        source_height=source_height,
        input_width=input_width,
        input_height=input_height,
        content_left=content_left,
        content_top=content_top,
        content_width=content_width,
        content_height=content_height,
        patch_width=patch_width,
        patch_height=patch_height,
    )
    return tensor, transform
