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
    MemoryPlacement,
    SampleHeader,
    TimePoint,
)
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception.contracts import (
    FeatureBatch,
    FeatureGeometry,
    FeatureTiming,
)
from vision_pipeline.perception.dinov2.depth import (
    DinoV2DepthConfig,
    TorchDinoV2DepthEstimator,
)
from vision_pipeline.perception.dinov2.matching import ImagePoint, match_reference_patch
from vision_pipeline.perception.dinov2.retrieval import DinoFeatureIndex
from vision_pipeline.perception.dinov2.segmentation import (
    DinoV2SegmentationConfig,
    TorchDinoV2SemanticSegmenter,
)


def source_header(sequence: int = 1) -> SampleHeader:
    clock = ClockDomain("host/test", ClockKind.HOST_MONOTONIC)
    return SampleHeader(
        sample_id=f"camera/{sequence}",
        source_id="camera",
        sequence_number=sequence,
        measurement_kind=MeasurementKind.RGB_IMAGE,
        captured_at=None,
        received_at=TimePoint(100, clock),
        frame_id=FrameId("camera_optical"),
        calibration=None,
        producer="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
    )


def feature_batch(
    spatial: torch.Tensor,
    global_features: torch.Tensor,
    *,
    sequence: int = 1,
) -> FeatureBatch[torch.Tensor]:
    grid_height = int(spatial.shape[2])
    grid_width = int(spatial.shape[3])
    clock = source_header().received_at.clock
    return FeatureBatch(
        source=source_header(sequence),
        global_features=global_features,
        spatial_features=spatial,
        extra_tokens=None,
        geometry=FeatureGeometry(
            source_width=grid_width * 14,
            source_height=grid_height * 14,
            input_width=grid_width * 14,
            input_height=grid_height * 14,
            content_left=0,
            content_top=0,
            content_width=grid_width * 14,
            content_height=grid_height * 14,
            patch_width=14,
            patch_height=14,
            grid_width=grid_width,
            grid_height=grid_height,
        ),
        model_id="dino-test",
        implementation="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
        memory=MemoryPlacement(MemoryKind.HOST),
        started_at=TimePoint(1, clock),
        finished_at=TimePoint(2, clock),
        timing=FeatureTiming(0, 0, 0, 0),
    )


def test_sparse_patch_matching_uses_cosine_similarity_and_geometry() -> None:
    reference = feature_batch(
        torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]]),
        torch.tensor([[1.0, 0.0]]),
    )
    target = feature_batch(
        torch.tensor([[[[0.0, 1.0]], [[1.0, 0.0]]]]),
        torch.tensor([[0.0, 1.0]]),
        sequence=2,
    )

    match = match_reference_patch(reference, target, ImagePoint(2.0, 3.0))

    assert match.reference_grid_xy == (0, 0)
    assert match.matched_grid_xy == (1, 0)
    assert match.reference_point == ImagePoint(7.0, 7.0)
    assert match.matched_point == ImagePoint(21.0, 7.0)
    assert match.cosine_similarity == pytest.approx(1.0)


def test_retrieval_index_ranks_global_descriptors_and_evicts_oldest() -> None:
    spatial = torch.ones((1, 2, 1, 1))
    index = DinoFeatureIndex(capacity=2)
    index.add("x", feature_batch(spatial, torch.tensor([[1.0, 0.0]])))
    index.add("y", feature_batch(spatial, torch.tensor([[0.0, 1.0]]), sequence=2))

    matches = index.query(
        feature_batch(spatial, torch.tensor([[0.1, 0.9]]), sequence=3),
        top_k=2,
    )

    assert [match.item_id for match in matches] == ["y", "x"]
    index.add("z", feature_batch(spatial, torch.tensor([[-1.0, 0.0]]), sequence=4))
    final_matches = index.query(feature_batch(spatial, torch.tensor([[1.0, 0.0]])))
    assert [match.item_id for match in final_matches] == ["y"]


@dataclass
class FakePatchEmbed:
    patch_size: int = 14


@dataclass
class FakeDecodeHead:
    min_depth: float = 0.0
    max_depth: float = 0.0


class FakeDepthModel:
    def __init__(self) -> None:
        self.backbone = type("Backbone", (), {"patch_size": 14})()
        self.decode_head = FakeDecodeHead()

    def to(self, _device: torch.device) -> FakeDepthModel:
        return self

    def eval(self) -> FakeDepthModel:
        return self

    def forward_dummy(self, image: torch.Tensor) -> torch.Tensor:
        return torch.full((1, 1, image.shape[2], image.shape[3]), 20.0)


def test_depth_adapter_restores_dataset_range_and_source_geometry() -> None:
    model = FakeDepthModel()
    clock = iter((1_000, 2_000, 3_000, 4_000))
    estimator = TorchDinoV2DepthEstimator(
        DinoV2DepthConfig(model="test", weights="NYU", image_size=28),
        model_loader=lambda _model, _weights: model,
        clock_ns=lambda: next(clock),
    )
    image = ImageFrame(np.zeros((14, 28, 3), dtype=np.uint8), PixelFormat.BGR8)

    result = estimator.estimate(image, source_header())

    assert result.depth_metres.shape == (1, 1, 14, 28)
    assert torch.all(result.depth_metres == 10.0)
    assert model.decode_head.max_depth == 10.0
    assert result.timing.wall_ms == pytest.approx(0.003)


class FakeSegmentationBackbone:
    patch_embed = FakePatchEmbed()
    embed_dim = 2
    num_register_tokens = 0

    def to(self, _device: torch.device) -> FakeSegmentationBackbone:
        return self

    def eval(self) -> FakeSegmentationBackbone:
        return self

    def forward_features(self, _image: torch.Tensor) -> object:
        return {
            "x_norm_patchtokens": torch.tensor([[[4.0, 0.0], [0.0, 4.0]]]),
        }


def segmentation_head_state(_model: str) -> dict[str, torch.Tensor]:
    state: dict[str, torch.Tensor] = {
        "bn.weight": torch.ones(2),
        "bn.bias": torch.zeros(2),
        "bn.running_mean": torch.zeros(2),
        "bn.running_var": torch.ones(2),
        "bn.num_batches_tracked": torch.tensor(0),
        "conv_seg.weight": torch.zeros((150, 2, 1, 1)),
        "conv_seg.bias": torch.zeros(150),
    }
    state["conv_seg.weight"][1, 0, 0, 0] = 1.0
    state["conv_seg.weight"][2, 1, 0, 0] = 1.0
    return state


def test_semantic_adapter_applies_official_linear_head_shape() -> None:
    clock = iter((1_000, 2_000, 3_000, 4_000))
    segmenter = TorchDinoV2SemanticSegmenter(
        DinoV2SegmentationConfig(model="dinov2_vits14", image_size=28),
        backbone_loader=lambda _model: FakeSegmentationBackbone(),
        head_loader=segmentation_head_state,
        clock_ns=lambda: next(clock),
    )
    image = ImageFrame(np.zeros((14, 28, 3), dtype=np.uint8), PixelFormat.BGR8)

    result = segmenter.segment(image, source_header())

    assert result.class_ids.shape == (1, 14, 28)
    assert torch.all(result.class_ids[:, :, :14] == 1)
    assert torch.all(result.class_ids[:, :, 14:] == 2)
    assert result.class_names[1] == "building"
    assert result.timing.wall_ms == pytest.approx(0.003)
