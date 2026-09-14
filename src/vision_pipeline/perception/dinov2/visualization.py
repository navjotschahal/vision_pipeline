"""Diagnostic visualization of dense DINOv2 patch features."""

from __future__ import annotations

from typing import cast

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from vision_pipeline.image import ImageFrame
from vision_pipeline.perception.contracts import FeatureBatch


class DinoPcaVisualizer:
    """Project patch descriptors to stable colors using a first-frame PCA basis.

    PCA is a diagnostic view, not a semantic segmentation result. Fitting once avoids
    arbitrary component sign changes and color flicker from frame to frame.
    """

    def __init__(self) -> None:
        self._mean: NDArray[np.float32] | None = None
        self._basis: NDArray[np.float32] | None = None
        self._low: NDArray[np.float32] | None = None
        self._high: NDArray[np.float32] | None = None

    def reset(self) -> None:
        """Make the next rendered frame the shared PCA/color reference."""

        self._mean = None
        self._basis = None
        self._low = None
        self._high = None

    def render(
        self,
        image: ImageFrame,
        batch: FeatureBatch[torch.Tensor],
    ) -> NDArray[np.uint8]:
        geometry = batch.geometry
        if (geometry.source_width, geometry.source_height) != (image.width, image.height):
            raise ValueError("feature geometry does not match the source image")

        spatial = batch.spatial_features
        expected_shape = (1, geometry.grid_height, geometry.grid_width)
        if spatial.ndim != 4 or (
            spatial.shape[0], spatial.shape[2], spatial.shape[3]
        ) != expected_shape:
            raise ValueError("spatial features do not match the declared feature grid")

        descriptors = (
            spatial[0]
            .detach()
            .to(device="cpu", dtype=torch.float32)
            .permute(1, 2, 0)
            .reshape(-1, spatial.shape[1])
            .numpy()
        )
        norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
        normalized = descriptors / np.maximum(norms, np.float32(1e-12))
        self._ensure_basis(normalized)
        assert self._mean is not None and self._basis is not None

        projected = (normalized - self._mean) @ self._basis
        projected = projected.reshape(geometry.grid_height, geometry.grid_width, 3)
        colors = self._normalize_colors(projected)
        model_view = cv2.resize(
            colors,
            (geometry.input_width, geometry.input_height),
            interpolation=cv2.INTER_NEAREST,
        )
        content = model_view[
            geometry.content_top : geometry.content_top + geometry.content_height,
            geometry.content_left : geometry.content_left + geometry.content_width,
        ]
        source_view = cv2.resize(
            content,
            (geometry.source_width, geometry.source_height),
            interpolation=cv2.INTER_NEAREST,
        )
        # Treat PCA components as RGB for presentation; OpenCV windows consume BGR.
        return np.ascontiguousarray(source_view[..., ::-1])

    def _ensure_basis(self, descriptors: NDArray[np.float32]) -> None:
        if self._basis is not None and self._basis.shape[0] == descriptors.shape[1]:
            return
        mean = descriptors.mean(axis=0, keepdims=True, dtype=np.float32)
        centered = descriptors - mean
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        component_count = min(3, right_vectors.shape[0])
        basis = np.zeros((descriptors.shape[1], 3), dtype=np.float32)
        basis[:, :component_count] = right_vectors[:component_count].T
        for component in range(component_count):
            pivot = int(np.argmax(np.abs(basis[:, component])))
            if basis[pivot, component] < 0:
                basis[:, component] *= -1
        self._mean = mean
        self._basis = basis

    def _normalize_colors(self, values: NDArray[np.float32]) -> NDArray[np.uint8]:
        flat = values.reshape(-1, 3)
        if self._low is None or self._high is None:
            self._low = np.asarray(np.percentile(flat, 2, axis=0), dtype=np.float32)
            self._high = np.asarray(np.percentile(flat, 98, axis=0), dtype=np.float32)
        scale = np.maximum(self._high - self._low, 1e-6)
        normalized = np.clip((values - self._low) / scale, 0.0, 1.0)
        return cast(NDArray[np.uint8], (normalized * 255).astype(np.uint8))


__all__ = ["DinoPcaVisualizer"]
