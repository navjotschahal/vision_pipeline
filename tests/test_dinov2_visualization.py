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
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception.contracts import (
    FeatureBatch,
    FeatureGeometry,
    FeatureTiming,
)
from vision_pipeline.perception.dinov2.visualization import DinoPcaVisualizer


def feature_batch(spatial: torch.Tensor) -> FeatureBatch[torch.Tensor]:
    clock = ClockDomain("host/test", ClockKind.HOST_MONOTONIC)
    source = SampleHeader(
        sample_id="camera/1",
        source_id="camera",
        sequence_number=1,
        measurement_kind=MeasurementKind.RGB_IMAGE,
        captured_at=None,
        received_at=TimePoint(100, clock),
        frame_id=FrameId("camera_optical"),
        calibration=None,
        producer="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "test"),
    )
    return FeatureBatch(
        source=source,
        global_features=torch.zeros((1, spatial.shape[1])),
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
        model_id="test",
        implementation="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "cpu"),
        memory=MemoryPlacement(MemoryKind.HOST),
        started_at=TimePoint(101, clock),
        finished_at=TimePoint(102, clock),
        timing=FeatureTiming(0, 0, 0, 0),
    )


def test_pca_visualizer_maps_grid_back_to_source_dimensions() -> None:
    spatial = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]]])
    image = ImageFrame(np.zeros((4, 8, 3), dtype=np.uint8), PixelFormat.BGR8)

    view = DinoPcaVisualizer().render(image, feature_batch(spatial))

    assert view.shape == image.data.shape
    assert view.dtype == np.uint8
    assert not np.array_equal(view[:, :4], view[:, 4:])


def test_pca_visualizer_rejects_wrong_source_geometry() -> None:
    spatial = torch.zeros((1, 3, 1, 2))
    image = ImageFrame(np.zeros((5, 8, 3), dtype=np.uint8), PixelFormat.BGR8)

    with pytest.raises(ValueError, match="does not match"):
        DinoPcaVisualizer().render(image, feature_batch(spatial))
