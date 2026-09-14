from __future__ import annotations

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
from vision_pipeline.perception.regions import FeatureGridRegion, RegionDescriptorBatch
from vision_pipeline.perception.tracking import (
    AppearanceTrackerConfig,
    GreedyAppearanceTracker,
)

CLOCK = ClockDomain("host/test", ClockKind.HOST_MONOTONIC)


def region_batch(
    sequence: int,
    boxes: list[tuple[float, float, float, float]],
    descriptors: torch.Tensor,
    classes: list[int] | None = None,
) -> RegionDescriptorBatch[torch.Tensor]:
    source = SampleHeader(
        sample_id=f"camera/{sequence}",
        source_id="camera",
        sequence_number=sequence,
        measurement_kind=MeasurementKind.RGB_IMAGE,
        captured_at=None,
        received_at=TimePoint(sequence * 1_000_000, CLOCK),
        frame_id=FrameId("camera_optical"),
        calibration=None,
        producer="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "test"),
    )
    class_ids = classes or [0] * len(boxes)
    detections = tuple(
        Detection2D(BoundingBox2D(*box), class_id, f"class-{class_id}", 0.9)
        for box, class_id in zip(boxes, class_ids, strict=True)
    )
    detection_batch = DetectionBatch(
        source=source,
        image_width=100,
        image_height=100,
        detections=detections,
        model_id="detector",
        implementation="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
        started_at=TimePoint(sequence * 1_000_000 + 1, CLOCK),
        finished_at=TimePoint(sequence * 1_000_000 + 2, CLOCK),
        timing=DetectionTiming(0, 0, 0, 0),
    )
    feature_batch = FeatureBatch(
        source=source,
        global_features=torch.zeros((1, 2)),
        spatial_features=torch.zeros((1, 2, 1, 1)),
        extra_tokens=None,
        geometry=FeatureGeometry(
            source_width=100,
            source_height=100,
            input_width=14,
            input_height=14,
            content_left=0,
            content_top=0,
            content_width=14,
            content_height=14,
            patch_width=14,
            patch_height=14,
            grid_width=1,
            grid_height=1,
        ),
        model_id="features",
        implementation="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
        memory=MemoryPlacement(MemoryKind.HOST),
        started_at=TimePoint(sequence * 1_000_000 + 3, CLOCK),
        finished_at=TimePoint(sequence * 1_000_000 + 4, CLOCK),
        timing=FeatureTiming(0, 0, 0, 0),
    )
    regions = tuple(FeatureGridRegion(index, 0, 0, 1, 1) for index in range(len(boxes)))
    return RegionDescriptorBatch(
        detections=detection_batch,
        features=feature_batch,
        descriptors=descriptors,
        regions=regions,
        pooling="test",
    )


def test_appearance_preserves_identity_when_two_objects_swap_positions() -> None:
    tracker = GreedyAppearanceTracker()
    left = (0.0, 0.0, 20.0, 20.0)
    right = (80.0, 0.0, 100.0, 20.0)

    first = tracker.update(
        region_batch(1, [left, right], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    )
    second = tracker.update(
        region_batch(2, [left, right], torch.tensor([[0.0, 1.0], [1.0, 0.0]]))
    )

    assert [item.track_id for item in first.tracked_detections] == [1, 2]
    assert [item.track_id for item in second.tracked_detections] == [2, 1]
    assert all(item.association_score is not None for item in second.tracked_detections)


def test_class_aware_matching_rejects_identical_appearance_from_other_class() -> None:
    tracker = GreedyAppearanceTracker()
    box = (10.0, 10.0, 30.0, 30.0)

    tracker.update(region_batch(1, [box], torch.tensor([[1.0, 0.0]]), [0]))
    result = tracker.update(region_batch(2, [box], torch.tensor([[1.0, 0.0]]), [1]))

    assert result.tracked_detections[0].track_id == 2
    assert result.tracked_detections[0].association_score is None


def test_track_expires_after_configured_number_of_missed_frames() -> None:
    tracker = GreedyAppearanceTracker(AppearanceTrackerConfig(max_missed_frames=0))
    box = (10.0, 10.0, 30.0, 30.0)

    tracker.update(region_batch(1, [box], torch.tensor([[1.0, 0.0]])))
    empty = tracker.update(region_batch(2, [], torch.empty((0, 2))))
    result = tracker.update(region_batch(3, [box], torch.tensor([[1.0, 0.0]])))

    assert empty.active_track_count == 0
    assert result.tracked_detections[0].track_id == 2
