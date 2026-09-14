from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

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
    SensorSample,
    TimePoint,
)
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception.contracts import DetectionBatch, DetectionTiming
from vision_pipeline.runtime import LatestFrameDetectionPipeline


@dataclass
class ControlledDetector:
    entered: threading.Event
    release: threading.Event
    processed_sequences: list[int]

    def detect(self, image: ImageFrame, source: SampleHeader) -> DetectionBatch:
        del image
        self.processed_sequences.append(source.sequence_number)
        self.entered.set()
        self.release.wait(timeout=2)
        started_ns = source.received_at.nanoseconds + 1_000_000
        finished_ns = started_ns + 2_000_000
        return DetectionBatch(
            source=source,
            image_width=8,
            image_height=6,
            detections=(),
            model_id="controlled",
            implementation="test",
            produced_on=ComputePlacement(ComputeKind.HOST_CPU, "test"),
            started_at=TimePoint(started_ns, source.received_at.clock),
            finished_at=TimePoint(finished_ns, source.received_at.clock),
            timing=DetectionTiming(0, 2, 0, 2),
        )


def sample(sequence: int) -> SensorSample[ImageFrame]:
    clock = ClockDomain("test/monotonic", ClockKind.HOST_MONOTONIC)
    header = SampleHeader(
        sample_id=f"camera/{sequence}",
        source_id="camera",
        sequence_number=sequence,
        measurement_kind=MeasurementKind.RGB_IMAGE,
        captured_at=None,
        received_at=TimePoint(sequence * 10_000_000, clock),
        frame_id=FrameId("camera_optical"),
        calibration=None,
        producer="test",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "test"),
    )
    image = ImageFrame(np.zeros((6, 8, 3), dtype=np.uint8), PixelFormat.BGR8)
    return SensorSample(header, image, MemoryPlacement(MemoryKind.HOST))


def test_busy_pipeline_replaces_stale_pending_frame() -> None:
    entered = threading.Event()
    release = threading.Event()
    detector = ControlledDetector(entered, release, [])
    pipeline = LatestFrameDetectionPipeline(detector)
    pipeline.start()

    pipeline.submit(sample(0))
    assert entered.wait(timeout=1)
    pipeline.submit(sample(1))
    pipeline.submit(sample(2))
    release.set()
    pipeline.close(drain=True)

    assert detector.processed_sequences == [0, 2]
    assert pipeline.stats.submitted == 3
    assert pipeline.stats.processed == 2
    assert pipeline.stats.dropped_before_processing == 1
    result = pipeline.take_latest()
    assert result is not None
    assert result.sample.header.sequence_number == 2
    assert result.detections.source is result.sample.header
    assert result.queue_wait_ms == 1
    assert result.end_to_end_ms == 3
