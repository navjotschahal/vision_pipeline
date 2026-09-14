from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from numpy.typing import NDArray

from vision_pipeline.contracts import (
    ClockDomain,
    ClockKind,
    ComputeKind,
    ComputePlacement,
    FrameId,
    MeasurementKind,
    SampleHeader,
    TimePoint,
)
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception.yolo import YoloConfig
from vision_pipeline.perception.yolo.ultralytics import UltralyticsYoloDetector


class FakeTensor:
    def __init__(self, values: NDArray[np.float32]) -> None:
        self.values = values
        self.device = "mps:0"

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> object:
        return self.values


@dataclass
class FakeBoxes:
    data: FakeTensor


@dataclass
class FakeResult:
    boxes: FakeBoxes
    names: dict[int, str]
    speed: dict[str, float]


class FakeModel:
    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    def predict(
        self,
        source: NDArray[np.uint8],
        *,
        device: str,
        conf: float,
        iou: float,
        imgsz: int,
        max_det: int,
        verbose: bool,
    ) -> list[object]:
        self.calls.append(
            {
                "source": source,
                "device": device,
                "conf": conf,
                "iou": iou,
                "imgsz": imgsz,
                "max_det": max_det,
                "verbose": verbose,
            }
        )
        return [self.result]


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


def test_yolo_adapter_emits_typed_detections_with_source_provenance() -> None:
    result = FakeResult(
        boxes=FakeBoxes(FakeTensor(np.array([[10, 20, 110, 220, 0.9, 0]], dtype=np.float32))),
        names={0: "person"},
        speed={"preprocess": 1.0, "inference": 5.0, "postprocess": 0.5},
    )
    model = FakeModel(result)
    clock_values = iter((1_000, 8_000_000))
    detector = UltralyticsYoloDetector(
        YoloConfig(model="test.pt", device="mps", confidence=0.4, image_size=512),
        model_factory=lambda _model: model,
        clock_ns=lambda: next(clock_values),
    )
    pixels = np.zeros((480, 640, 3), dtype=np.uint8)
    source = source_header()

    batch = detector.detect(ImageFrame(pixels, PixelFormat.BGR8), source)

    assert batch.source is source
    assert batch.image_width == 640
    assert batch.image_height == 480
    assert batch.model_id == "test.pt"
    assert batch.produced_on.kind is ComputeKind.HOST_GPU
    assert batch.produced_on.device_id == "apple-mps"
    assert batch.timing.inference_ms == 5.0
    assert batch.timing.wall_ms == pytest.approx(7.999)
    assert len(batch.detections) == 1
    detection = batch.detections[0]
    assert detection.label == "person"
    assert detection.confidence == pytest.approx(0.9)
    assert detection.box.x_max == 110
    assert model.calls[0]["source"] is pixels
    assert model.calls[0]["device"] == "mps"
    assert model.calls[0]["imgsz"] == 512
    assert model.calls[0]["conf"] == 0.4
