from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from vision_pipeline.contracts import (
    ClockDomain,
    ClockKind,
    ComputeKind,
    ComputePlacement,
    FrameId,
    MeasurementKind,
    MemoryKind,
    SampleHeader,
    TimePoint,
)
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception.dinov2 import DinoV2Config, DinoV2Error
from vision_pipeline.perception.dinov2.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    preprocess_image,
)
from vision_pipeline.perception.dinov2.pytorch import TorchDinoV2FeatureExtractor


def source_header() -> SampleHeader:
    clock = ClockDomain("host/test", ClockKind.HOST_MONOTONIC)
    return SampleHeader(
        sample_id="run/camera/7",
        source_id="camera",
        sequence_number=7,
        measurement_kind=MeasurementKind.RGB_IMAGE,
        captured_at=None,
        received_at=TimePoint(900, clock),
        frame_id=FrameId("camera_optical"),
        calibration=None,
        producer="test-camera",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "test"),
    )


def test_preprocessing_converts_bgr_and_applies_reference_normalization() -> None:
    bgr = np.zeros((14, 14, 3), dtype=np.uint8)
    bgr[:, :] = (10, 20, 30)

    tensor, transform = preprocess_image(
        ImageFrame(bgr, PixelFormat.BGR8),
        long_edge=14,
        patch_size=(14, 14),
        device=torch.device("cpu"),
    )

    expected_rgb = np.array((30, 20, 10), dtype=np.float32) / 255.0
    expected = (expected_rgb - IMAGENET_MEAN) / IMAGENET_STD
    assert tensor.shape == (1, 3, 14, 14)
    assert tensor.dtype is torch.float32
    assert tensor[0, :, 0, 0].tolist() == pytest.approx(expected.tolist())
    assert transform.grid_width == 1
    assert transform.grid_height == 1


def test_preprocessing_preserves_aspect_and_only_pads_to_whole_patches() -> None:
    image = ImageFrame(np.zeros((1080, 1920, 3), dtype=np.uint8), PixelFormat.RGB8)

    tensor, transform = preprocess_image(
        image,
        long_edge=518,
        patch_size=(14, 14),
        device=torch.device("cpu"),
    )

    assert transform.content_width == 518
    assert transform.content_height == 291
    assert transform.input_width == 518
    assert transform.input_height == 294
    assert transform.content_top == 1
    assert (transform.grid_width, transform.grid_height) == (37, 21)
    assert tensor.shape == (1, 3, 294, 518)
    assert torch.count_nonzero(tensor[:, :, 0, :]) == 0


@dataclass
class FakePatchEmbed:
    patch_size: tuple[int, int] = (14, 14)


class FakeDinoModel:
    def __init__(self, *, patch_count: int = 2) -> None:
        self.patch_embed = FakePatchEmbed()
        self.patch_count = patch_count
        self.device = torch.device("cpu")
        self.received: torch.Tensor | None = None

    def to(self, device: torch.device) -> FakeDinoModel:
        self.device = device
        return self

    def eval(self) -> FakeDinoModel:
        return self

    def forward_features(self, image: torch.Tensor) -> object:
        self.received = image
        return {
            "x_norm_clstoken": torch.tensor([[1.0, 2.0, 3.0]], device=self.device),
            "x_norm_regtokens": torch.ones((1, 4, 3), device=self.device),
            "x_norm_patchtokens": torch.arange(
                self.patch_count * 3,
                dtype=torch.float32,
                device=self.device,
            ).reshape(1, self.patch_count, 3),
        }


def test_torch_adapter_keeps_features_on_device_and_records_geometry() -> None:
    model = FakeDinoModel()
    clock_values = iter((1_000, 2_000, 5_000, 8_000))
    extractor = TorchDinoV2FeatureExtractor(
        DinoV2Config(model="test-dino", device="cpu", image_size=28),
        model_loader=lambda _name: model,
        clock_ns=lambda: next(clock_values),
    )
    image = ImageFrame(np.zeros((14, 28, 3), dtype=np.uint8), PixelFormat.BGR8)
    source = source_header()

    result = extractor.extract(image, source)

    assert result.source is source
    assert result.global_features.shape == (1, 3)
    assert result.extra_tokens is not None
    assert result.extra_tokens.shape == (1, 4, 3)
    assert result.spatial_features.shape == (1, 3, 1, 2)
    assert result.spatial_features.device.type == "cpu"
    assert result.geometry.grid_width == 2
    assert result.geometry.grid_height == 1
    assert result.geometry.scale_x == 1
    assert result.produced_on.kind is ComputeKind.HOST_CPU
    assert result.memory.kind is MemoryKind.HOST
    assert result.timing.preprocess_ms == pytest.approx(0.001)
    assert result.timing.inference_ms == pytest.approx(0.003)
    assert result.timing.postprocess_ms == pytest.approx(0.003)
    assert model.received is not None
    assert model.received.shape == (1, 3, 14, 28)


def test_torch_adapter_rejects_patch_count_that_disagrees_with_input() -> None:
    extractor = TorchDinoV2FeatureExtractor(
        DinoV2Config(model="test-dino", device="cpu", image_size=28),
        model_loader=lambda _name: FakeDinoModel(patch_count=3),
    )
    image = ImageFrame(np.zeros((14, 28, 3), dtype=np.uint8), PixelFormat.BGR8)

    with pytest.raises(DinoV2Error, match="returned 3 patches; expected 2"):
        extractor.extract(image, source_header())
