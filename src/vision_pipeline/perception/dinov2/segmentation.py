"""ADE20K semantic segmentation using Meta's DINOv2 linear head."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

import torch
import torch.nn as nn
import torch.nn.functional as functional

from vision_pipeline.contracts import ClockKind, SampleHeader
from vision_pipeline.image import ImageFrame

from . import DinoV2Error
from .ade20k import ADE20K_CLASS_NAMES
from .preprocessing import preprocess_image
from .tasks import DenseTaskTiming, SemanticSegmentation
from .torch_support import (
    normalize_device,
    official_checkpoint_safe_globals,
    placement,
    synchronize,
)

_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
_SUPPORTED_MODELS = {
    "dinov2_vits14",
    "dinov2_vitb14",
    "dinov2_vitl14",
    "dinov2_vitg14",
}


@dataclass(frozen=True, slots=True)
class DinoV2SegmentationConfig:
    model: str = "dinov2_vits14"
    device: str = "cpu"
    image_size: int = 518

    def __post_init__(self) -> None:
        if self.model not in _SUPPORTED_MODELS:
            raise ValueError(
                "the ADE20K head requires a matching non-register DINOv2 model; "
                f"supported models are {sorted(_SUPPORTED_MODELS)}"
            )
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")


class SegmentationBackbone(Protocol):
    patch_embed: object
    embed_dim: int
    num_register_tokens: int

    def to(self, device: torch.device) -> SegmentationBackbone: ...

    def eval(self) -> SegmentationBackbone: ...

    def forward_features(self, image: torch.Tensor) -> object: ...


BackboneLoader = Callable[[str], SegmentationBackbone]
HeadLoader = Callable[[str], Mapping[str, torch.Tensor]]
ClockReader = Callable[[], int]


def _default_backbone_loader(model_name: str) -> SegmentationBackbone:
    try:
        model = torch.hub.load(  # type: ignore[no-untyped-call]
            "facebookresearch/dinov2",
            model_name,
            trust_repo=True,
        )
    except Exception as error:
        raise DinoV2Error(f"could not load DINOv2 backbone {model_name!r}: {error}") from error
    return cast(SegmentationBackbone, model)


def _default_head_loader(model_name: str) -> Mapping[str, torch.Tensor]:
    url = f"{_BASE_URL}/{model_name}/{model_name}_ade20k_linear_head.pth"
    try:
        with official_checkpoint_safe_globals():
            checkpoint: Any = torch.hub.load_state_dict_from_url(
                url,
                map_location="cpu",
                weights_only=True,
            )
    except Exception as error:
        raise DinoV2Error(
            f"could not load official ADE20K head for {model_name!r}: {error}"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise DinoV2Error("ADE20K checkpoint is not a mapping")
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, Mapping):
        raise DinoV2Error("ADE20K checkpoint has no state_dict mapping")
    state: dict[str, torch.Tensor] = {}
    for key, value in raw_state.items():
        if (
            isinstance(key, str)
            and key.startswith("decode_head.")
            and isinstance(value, torch.Tensor)
        ):
            state[key.removeprefix("decode_head.")] = value
    return state


class _LinearSegmentationHead(nn.Module):
    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(embedding_size)
        self.conv_seg = nn.Conv2d(embedding_size, len(ADE20K_CLASS_NAMES), kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv_seg(self.bn(features)))


def _patch_size(model: SegmentationBackbone) -> tuple[int, int]:
    value = getattr(model.patch_embed, "patch_size", None)
    if isinstance(value, int) and value > 0:
        return value, value
    if (
        isinstance(value, tuple | list)
        and len(value) == 2
        and all(isinstance(item, int) and item > 0 for item in value)
    ):
        return int(value[0]), int(value[1])
    raise DinoV2Error(f"segmentation backbone has unsupported patch size {value!r}")


class TorchDinoV2SemanticSegmenter:
    """Frozen DINOv2 backbone with Meta's ADE20K linear probe."""

    def __init__(
        self,
        config: DinoV2SegmentationConfig,
        *,
        backbone_loader: BackboneLoader = _default_backbone_loader,
        head_loader: HeadLoader = _default_head_loader,
        clock_ns: ClockReader = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._clock_ns = clock_ns
        self._device = normalize_device(config.device)
        try:
            self._backbone = backbone_loader(config.model).to(self._device).eval()
            self._patch_size = _patch_size(self._backbone)
            self._head = _LinearSegmentationHead(self._backbone.embed_dim)
            self._head.load_state_dict(dict(head_loader(config.model)), strict=True)
            self._head = self._head.to(self._device).eval()
        except DinoV2Error:
            raise
        except Exception as error:
            raise DinoV2Error(f"could not initialize DINOv2 segmentation: {error}") from error

    def segment(
        self, image: ImageFrame, source: SampleHeader
    ) -> SemanticSegmentation[torch.Tensor]:
        if source.received_at.clock.kind is not ClockKind.HOST_MONOTONIC:
            raise DinoV2Error("segmentation timing requires a host-monotonic receipt clock")

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
                output = self._backbone.forward_features(tensor)
                if not isinstance(output, Mapping):
                    raise DinoV2Error("DINOv2 backbone output is not a mapping")
                patch_tokens = output.get("x_norm_patchtokens")
                if not isinstance(patch_tokens, torch.Tensor) or patch_tokens.ndim != 3:
                    raise DinoV2Error(
                        "DINOv2 backbone did not return [B, patches, C] "
                        "x_norm_patchtokens"
                    )
                expected = transform.grid_height * transform.grid_width
                if patch_tokens.shape[1] != expected:
                    raise DinoV2Error(
                        f"backbone returned {patch_tokens.shape[1]} patches; expected {expected}"
                    )
                spatial = patch_tokens.transpose(1, 2).reshape(
                    1,
                    patch_tokens.shape[2],
                    transform.grid_height,
                    transform.grid_width,
                )
                logits = self._head(spatial)
                logits = functional.interpolate(
                    logits,
                    size=(transform.input_height, transform.input_width),
                    mode="bilinear",
                    align_corners=False,
                )
                logits = logits[
                    :,
                    :,
                    transform.content_top : transform.content_top + transform.content_height,
                    transform.content_left : transform.content_left + transform.content_width,
                ]
                probabilities = logits.softmax(dim=1)
                confidence, class_ids = probabilities.max(dim=1)
            synchronize(self._device)
            inferred_ns = self._clock_ns()

            class_ids = functional.interpolate(
                class_ids[:, None].to(dtype=torch.float32),
                size=(transform.source_height, transform.source_width),
                mode="nearest",
            )[:, 0].to(dtype=torch.int64)
            confidence = functional.interpolate(
                confidence[:, None],
                size=(transform.source_height, transform.source_width),
                mode="bilinear",
                align_corners=False,
            )[:, 0]
            synchronize(self._device)
            finished_ns = self._clock_ns()
        except DinoV2Error:
            raise
        except Exception as error:
            raise DinoV2Error(f"DINOv2 semantic segmentation failed: {error}") from error

        produced_on, memory = placement(class_ids.device)
        return SemanticSegmentation(
            source=source,
            class_ids=class_ids,
            confidence=confidence,
            class_names=ADE20K_CLASS_NAMES,
            model_id=f"{self.config.model}:ade20k-linear",
            produced_on=produced_on,
            memory=memory,
            timing=DenseTaskTiming(
                preprocess_ms=(preprocessed_ns - started_ns) / 1_000_000,
                inference_ms=(inferred_ns - preprocessed_ns) / 1_000_000,
                postprocess_ms=(finished_ns - inferred_ns) / 1_000_000,
                wall_ms=(finished_ns - started_ns) / 1_000_000,
            ),
        )


__all__ = ["DinoV2SegmentationConfig", "TorchDinoV2SemanticSegmenter"]
