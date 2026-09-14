"""A bounded latest-frame scheduler for live object detection."""

from __future__ import annotations

import threading
from dataclasses import dataclass

from vision_pipeline.contracts import SensorSample
from vision_pipeline.image import ImageFrame
from vision_pipeline.perception.contracts import DetectionBatch, Detector


class PipelineClosedError(RuntimeError):
    """An operation requires a running pipeline."""


class PipelineWorkerError(RuntimeError):
    """The detector worker failed."""


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """A source image kept together with detections derived from that exact image."""

    sample: SensorSample[ImageFrame]
    detections: DetectionBatch
    queue_wait_ms: float
    end_to_end_ms: float

    def __post_init__(self) -> None:
        if self.detections.source != self.sample.header:
            raise ValueError("detections must reference the processed source sample")
        if self.queue_wait_ms < 0 or self.end_to_end_ms < 0:
            raise ValueError("pipeline latency values must be non-negative")


@dataclass(frozen=True, slots=True)
class PipelineStats:
    submitted: int
    processed: int
    dropped_before_processing: int
    dropped_results: int
    worker_busy: bool
    frame_pending: bool


class LatestFrameDetectionPipeline:
    """Run one detector worker with a single replaceable pending-frame slot.

    A frame already being processed is never interrupted. If a newer frame arrives while
    another frame is pending, the stale pending frame is counted and replaced. Results
    also use a single slot so consumers cannot create an unbounded output backlog.
    """

    def __init__(self, detector: Detector, *, thread_name: str = "detection-worker") -> None:
        self._detector = detector
        self._condition = threading.Condition()
        self._thread_name = thread_name
        self._thread: threading.Thread | None = None
        self._pending: SensorSample[ImageFrame] | None = None
        self._latest_result: ProcessedFrame | None = None
        self._worker_error: BaseException | None = None
        self._stopping = False
        self._worker_busy = False
        self._submitted = 0
        self._processed = 0
        self._dropped_before_processing = 0
        self._dropped_results = 0

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive() and not self._stopping

    @property
    def stats(self) -> PipelineStats:
        with self._condition:
            return PipelineStats(
                submitted=self._submitted,
                processed=self._processed,
                dropped_before_processing=self._dropped_before_processing,
                dropped_results=self._dropped_results,
                worker_busy=self._worker_busy,
                frame_pending=self._pending is not None,
            )

    def start(self) -> None:
        with self._condition:
            if self._thread is not None:
                raise PipelineClosedError("pipeline instances cannot be restarted")
            self._thread = threading.Thread(
                target=self._run,
                name=self._thread_name,
                daemon=False,
            )
            self._thread.start()

    def submit(self, sample: SensorSample[ImageFrame]) -> None:
        with self._condition:
            self._raise_worker_error()
            if not self.is_running:
                raise PipelineClosedError("start() must be called before submit()")
            self._submitted += 1
            if self._pending is not None:
                self._dropped_before_processing += 1
            self._pending = sample
            self._condition.notify()

    def take_latest(self) -> ProcessedFrame | None:
        with self._condition:
            self._raise_worker_error()
            result = self._latest_result
            self._latest_result = None
            return result

    def close(self, *, drain: bool = True) -> None:
        with self._condition:
            thread = self._thread
            if thread is None:
                return
            if not self._stopping:
                self._stopping = True
                if not drain and self._pending is not None:
                    self._pending = None
                    self._dropped_before_processing += 1
                self._condition.notify_all()
        thread.join()
        with self._condition:
            self._raise_worker_error()

    def _raise_worker_error(self) -> None:
        if self._worker_error is not None:
            raise PipelineWorkerError("detector worker failed") from self._worker_error

    def _run(self) -> None:
        try:
            while True:
                with self._condition:
                    self._condition.wait_for(
                        lambda: self._pending is not None or self._stopping
                    )
                    if self._pending is None:
                        return
                    sample = self._pending
                    self._pending = None
                    self._worker_busy = True

                detections = self._detector.detect(sample.payload, sample.header)
                queue_wait_ns = detections.started_at - sample.header.received_at
                end_to_end_ns = detections.finished_at - sample.header.received_at
                result = ProcessedFrame(
                    sample=sample,
                    detections=detections,
                    queue_wait_ms=queue_wait_ns / 1_000_000,
                    end_to_end_ms=end_to_end_ns / 1_000_000,
                )

                with self._condition:
                    self._worker_busy = False
                    self._processed += 1
                    if self._latest_result is not None:
                        self._dropped_results += 1
                    self._latest_result = result
                    if self._stopping and self._pending is None:
                        return
        except BaseException as error:
            with self._condition:
                self._worker_busy = False
                self._worker_error = error
                self._stopping = True
                if self._pending is not None:
                    self._pending = None
                    self._dropped_before_processing += 1
                self._condition.notify_all()
