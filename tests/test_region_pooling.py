from __future__ import annotations

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
from vision_pipeline.perception.contracts import (
    BoundingBox2D,
    Detection2D,
    DetectionBatch,
    DetectionTiming,
    FeatureBatch,
    FeatureGeometry,
    FeatureTiming,
)
from vision_pipeline.perception.regions import pool_detection_regions


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
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "test"),
    )


def detection_batch(source: SampleHeader, *, include_boxes: bool = True) -> DetectionBatch:
    detections = (
        (
            Detection2D(BoundingBox2D(0, 0, 4, 4), 0, "left", 0.9),
            Detection2D(BoundingBox2D(4, 0, 8, 4), 1, "right", 0.8),
        )
        if include_boxes
        else ()
    )
    return DetectionBatch(
        source=source,
        image_width=8,
        image_height=4,
        detections=detections,
        model_id="yolo-test",
        implementation="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
        started_at=TimePoint(101, source.received_at.clock),
        finished_at=TimePoint(102, source.received_at.clock),
        timing=DetectionTiming(0, 0, 0, 0),
    )


def feature_batch(source: SampleHeader) -> FeatureBatch[torch.Tensor]:
    # The left patch is [3, 0], and the right patch is [0, 4].
    spatial = torch.tensor([[[[3.0, 0.0]], [[0.0, 4.0]]]])
    return FeatureBatch(
        source=source,
        global_features=torch.zeros((1, 2)),
        spatial_features=spatial,
        extra_tokens=None,
        geometry=FeatureGeometry(
            source_width=8,
            source_height=4,
            input_width=4,
            input_height=2,
            content_left=0,
            content_top=0,
            content_width=4,
            content_height=2,
            patch_width=2,
            patch_height=2,
            grid_width=2,
            grid_height=1,
        ),
        model_id="dino-test",
        implementation="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
        memory=MemoryPlacement(MemoryKind.HOST),
        started_at=TimePoint(103, source.received_at.clock),
        finished_at=TimePoint(104, source.received_at.clock),
        timing=FeatureTiming(0, 0, 0, 0),
    )


def test_boxes_map_to_patch_regions_and_produce_normalized_descriptors() -> None:
    source = source_header()

    result = pool_detection_regions(detection_batch(source), feature_batch(source))

    assert result.descriptors.shape == (2, 2)
    assert result.descriptors.device.type == "cpu"
    assert result.descriptors.numpy() == pytest.approx(np.eye(2, dtype=np.float32))
    assert (result.regions[0].x_min, result.regions[0].x_max) == (0, 1)
    assert (result.regions[1].x_min, result.regions[1].x_max) == (1, 2)


def test_empty_detection_batch_produces_empty_device_resident_matrix() -> None:
    source = source_header()

    result = pool_detection_regions(
        detection_batch(source, include_boxes=False),
        feature_batch(source),
    )

    assert result.descriptors.shape == (0, 2)
    assert result.regions == ()


def test_pooling_rejects_results_from_different_frames() -> None:
    left = source_header(1)
    right = source_header(2)

    with pytest.raises(ValueError, match="same source sample"):
        pool_detection_regions(detection_batch(left), feature_batch(right))
