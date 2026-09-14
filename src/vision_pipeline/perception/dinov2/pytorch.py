"""Direct PyTorch DINOv2 feature extraction using Meta's official architecture."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Protocol, cast

import torch

from vision_pipeline.contracts import (
    ClockKind,
    SampleHeader,
    TimePoint,
)
from vision_pipeline.image import ImageFrame
from vision_pipeline.perception.contracts import (
    FeatureBatch,
    FeatureGeometry,
    FeatureTiming,
)
from vision_pipeline.perception.dinov2 import DinoV2Config, DinoV2Error
from vision_pipeline.perception.dinov2.preprocessing import preprocess_image
from vision_pipeline.perception.dinov2.torch_support import (
    normalize_device,
    placement,
    synchronize,
)


class DinoV2Model(Protocol):
    patch_embed: object

    def to(self, device: torch.device) -> DinoV2Model: ...

    def eval(self) -> DinoV2Model: ...

    def forward_features(self, image: torch.Tensor) -> object: ...


ModelLoader = Callable[[str], DinoV2Model]
ClockReader = Callable[[], int]


def _default_model_loader(model_name: str) -> DinoV2Model:
    try:
        model = torch.hub.load(  # type: ignore[no-untyped-call]
            "facebookresearch/dinov2",
            model_name,
            trust_repo=True,
        )
    except Exception as error:
        raise DinoV2Error(
            f"could not load official DINOv2 model {model_name!r} through Torch Hub: {error}"
        ) from error
    return cast(DinoV2Model, model)


def _patch_size(model: DinoV2Model) -> tuple[int, int]:
    value = getattr(model.patch_embed, "patch_size", None)
    if isinstance(value, int):
        size = (value, value)
    elif (
        isinstance(value, tuple | list)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    ):
        size = (int(value[0]), int(value[1]))
    else:
        raise DinoV2Error(f"model exposes an unsupported patch size {value!r}")
    if size[0] <= 0 or size[1] <= 0:
        raise DinoV2Error(f"model exposes an invalid patch size {size!r}")
    return size


def _require_tensor(
    features: Mapping[str, object], key: str, *, required: bool = True
) -> torch.Tensor | None:
    value = features.get(key)
    if value is None and not required:
        return None
    if not isinstance(value, torch.Tensor):
        raise DinoV2Error(f"model output {key!r} is not a tensor")
    return value


class TorchDinoV2FeatureExtractor:
    """Research adapter that exposes DINOv2 global, register, and patch features."""

    def __init__(
        self,
        config: DinoV2Config,
        *,
        model_loader: ModelLoader = _default_model_loader,
        clock_ns: ClockReader = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._clock_ns = clock_ns
        self._device = normalize_device(config.device)
        try:
            self._model = model_loader(config.model).to(self._device).eval()
            self._patch_size = _patch_size(self._model)
        except DinoV2Error:
            raise
        except Exception as error:
            raise DinoV2Error(
                f"could not initialize DINOv2 model {config.model!r}: {error}"
            ) from error

    def extract(
        self, image: ImageFrame, source: SampleHeader
    ) -> FeatureBatch[torch.Tensor]:
        if source.received_at.clock.kind is not ClockKind.HOST_MONOTONIC:
            raise DinoV2Error("feature timing requires a host-monotonic receipt clock")

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
                raw_features = self._model.forward_features(tensor)
            synchronize(self._device)
            inferred_ns = self._clock_ns()

            if not isinstance(raw_features, Mapping):
                raise DinoV2Error(
                    f"model returned unsupported feature type {type(raw_features).__name__}"
                )
            global_features = _require_tensor(raw_features, "x_norm_clstoken")
            patch_tokens = _require_tensor(raw_features, "x_norm_patchtokens")
            extra_tokens = _require_tensor(
                raw_features,
                "x_norm_regtokens",
                required=False,
            )
            assert global_features is not None and patch_tokens is not None
            if global_features.ndim != 2 or global_features.shape[0] != 1:
                raise DinoV2Error(
                    "x_norm_clstoken must have single-image [1, embedding] shape"
                )
            if patch_tokens.ndim != 3 or patch_tokens.shape[0] != 1:
                raise DinoV2Error(
                    "x_norm_patchtokens must have single-image [1, patches, embedding] shape"
                )
            expected_patches = transform.grid_height * transform.grid_width
            if patch_tokens.shape[1] != expected_patches:
                raise DinoV2Error(
                    f"model returned {patch_tokens.shape[1]} patches; expected {expected_patches} "
                    f"for {transform.input_width}x{transform.input_height} input"
                )
            spatial_features = patch_tokens.transpose(1, 2).reshape(
                1,
                patch_tokens.shape[2],
                transform.grid_height,
                transform.grid_width,
            )
            finished_ns = self._clock_ns()
        except DinoV2Error:
            raise
        except Exception as error:
            raise DinoV2Error(f"DINOv2 feature extraction failed: {error}") from error

        produced_on, memory = placement(spatial_features.device)
        return FeatureBatch(
            source=source,
            global_features=global_features,
            spatial_features=spatial_features,
            extra_tokens=extra_tokens,
            geometry=FeatureGeometry(
                source_width=transform.source_width,
                source_height=transform.source_height,
                input_width=transform.input_width,
                input_height=transform.input_height,
                content_left=transform.content_left,
                content_top=transform.content_top,
                content_width=transform.content_width,
                content_height=transform.content_height,
                patch_width=transform.patch_width,
                patch_height=transform.patch_height,
                grid_width=transform.grid_width,
                grid_height=transform.grid_height,
            ),
            model_id=self.config.model,
            implementation=f"pytorch-direct/{torch.__version__}+official-dinov2",
            produced_on=produced_on,
            memory=memory,
            started_at=TimePoint(started_ns, source.received_at.clock),
            finished_at=TimePoint(finished_ns, source.received_at.clock),
            timing=FeatureTiming(
                preprocess_ms=(preprocessed_ns - started_ns) / 1_000_000,
                inference_ms=(inferred_ns - preprocessed_ns) / 1_000_000,
                postprocess_ms=(finished_ns - inferred_ns) / 1_000_000,
                wall_ms=(finished_ns - started_ns) / 1_000_000,
            ),
        )


__all__ = ["TorchDinoV2FeatureExtractor"]
