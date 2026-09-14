"""Metric monocular depth using Meta's official DINOv2 linear depth heads."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, cast

import torch
import torch.nn.functional as functional

from vision_pipeline.contracts import ClockKind, SampleHeader
from vision_pipeline.image import ImageFrame

from . import DinoV2Error
from .preprocessing import preprocess_image
from .tasks import DenseTaskTiming, DepthPrediction
from .torch_support import (
    normalize_device,
    official_checkpoint_safe_globals,
    placement,
    synchronize,
)


@dataclass(frozen=True, slots=True)
class DinoV2DepthConfig:
    model: str = "dinov2_vits14_ld"
    weights: str = "NYU"
    device: str = "cpu"
    image_size: int = 518

    def __post_init__(self) -> None:
        if not self.model.strip() or not self.device.strip():
            raise ValueError("model and device must be non-empty")
        if self.weights not in {"NYU", "KITTI"}:
            raise ValueError("weights must be NYU or KITTI")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")


class DepthDecodeHead(Protocol):
    min_depth: float
    max_depth: float


class DepthModel(Protocol):
    backbone: object
    decode_head: DepthDecodeHead

    def to(self, device: torch.device) -> DepthModel: ...

    def eval(self) -> DepthModel: ...

    def forward_dummy(self, image: torch.Tensor) -> torch.Tensor: ...


DepthModelLoader = Callable[[str, str], DepthModel]
ClockReader = Callable[[], int]


def _default_model_loader(model_name: str, weights: str) -> DepthModel:
    try:
        # Meta's older head files contain NumPy scalar metadata. Keep PyTorch's
        # weights-only loader enabled and narrowly allow-list those scalar types.
        with official_checkpoint_safe_globals():
            model = torch.hub.load(  # type: ignore[no-untyped-call]
                "facebookresearch/dinov2",
                model_name,
                trust_repo=True,
                weights=weights,
            )
    except Exception as error:
        raise DinoV2Error(
            f"could not load official DINOv2 depth model {model_name!r}: {error}"
        ) from error
    return cast(DepthModel, model)


def _depth_patch_size(model: DepthModel) -> tuple[int, int]:
    backbone = getattr(model, "backbone", None)
    value = getattr(backbone, "patch_size", None)
    if isinstance(value, int) and value > 0:
        return value, value
    raise DinoV2Error(f"depth backbone exposes an unsupported patch size {value!r}")


class TorchDinoV2DepthEstimator:
    """DINOv2 backbone plus the official NYU or KITTI metric-depth head."""

    def __init__(
        self,
        config: DinoV2DepthConfig,
        *,
        model_loader: DepthModelLoader = _default_model_loader,
        clock_ns: ClockReader = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._clock_ns = clock_ns
        self._device = normalize_device(config.device)
        self._min_depth = 0.001
        self._max_depth = 10.0 if config.weights == "NYU" else 80.0
        try:
            self._model = model_loader(config.model, config.weights).to(self._device).eval()
            self._patch_size = _depth_patch_size(self._model)
            # Meta's current Hub builder passes the dataset range into its helper,
            # but the linear-head helper hard-codes KITTI's 80 m maximum. Set the
            # released head's range explicitly before it constructs its depth bins.
            self._model.decode_head.min_depth = self._min_depth
            self._model.decode_head.max_depth = self._max_depth
        except DinoV2Error:
            raise
        except Exception as error:
            raise DinoV2Error(f"could not initialize DINOv2 depth model: {error}") from error

    def estimate(
        self, image: ImageFrame, source: SampleHeader
    ) -> DepthPrediction[torch.Tensor]:
        if source.received_at.clock.kind is not ClockKind.HOST_MONOTONIC:
            raise DinoV2Error("depth timing requires a host-monotonic receipt clock")

        started_ns = self._clock_ns()
        try:
            tensor, transform = preprocess_image(
                image,
                self.config.image_size,
                self._patch_size,
                self._device,
            )
            synchronize(self._device)
            preprocessed_ns = self._clock_ns()
            with torch.inference_mode():
                depth = self._model.forward_dummy(tensor)
            synchronize(self._device)
            inferred_ns = self._clock_ns()

            if depth.ndim != 4 or depth.shape[:2] != (1, 1):
                raise DinoV2Error(
                    f"depth model returned {tuple(depth.shape)}; expected [1, 1, H, W]"
                )
            depth = depth[
                :,
                :,
                transform.content_top : transform.content_top + transform.content_height,
                transform.content_left : transform.content_left + transform.content_width,
            ]
            depth = functional.interpolate(
                depth,
                size=(transform.source_height, transform.source_width),
                mode="bilinear",
                align_corners=False,
            ).clamp_(self._min_depth, self._max_depth)
            synchronize(self._device)
            finished_ns = self._clock_ns()
        except DinoV2Error:
            raise
        except Exception as error:
            raise DinoV2Error(f"DINOv2 depth estimation failed: {error}") from error

        produced_on, memory = placement(depth.device)
        return DepthPrediction(
            source=source,
            depth_metres=depth,
            min_depth_metres=self._min_depth,
            max_depth_metres=self._max_depth,
            model_id=f"{self.config.model}:{self.config.weights.lower()}",
            produced_on=produced_on,
            memory=memory,
            timing=DenseTaskTiming(
                preprocess_ms=(preprocessed_ns - started_ns) / 1_000_000,
                inference_ms=(inferred_ns - preprocessed_ns) / 1_000_000,
                postprocess_ms=(finished_ns - inferred_ns) / 1_000_000,
                wall_ms=(finished_ns - started_ns) / 1_000_000,
            ),
        )


__all__ = ["DinoV2DepthConfig", "TorchDinoV2DepthEstimator"]
