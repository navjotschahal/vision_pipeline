"""Ultralytics YOLO adapter that emits model-independent detection contracts."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import (
    ClockKind,
    ComputeKind,
    ComputePlacement,
    SampleHeader,
    TimePoint,
)
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception.contracts import (
    BoundingBox2D,
    Detection2D,
    DetectionBatch,
    DetectionTiming,
)
from vision_pipeline.perception.yolo import YoloConfig, YoloError


class TensorLike(Protocol):
    device: object

    def cpu(self) -> TensorLike: ...

    def numpy(self) -> object: ...


class BoxesLike(Protocol):
    data: TensorLike


class ResultLike(Protocol):
    boxes: BoxesLike | None
    names: Mapping[int, str]
    speed: Mapping[str, float]


class YoloModelLike(Protocol):
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
    ) -> Sequence[object]: ...


class YoloConstructor(Protocol):
    def __call__(self, model: str) -> YoloModelLike: ...


ModelFactory = Callable[[str], YoloModelLike]
ClockReader = Callable[[], int]


def _default_model_factory(model: str) -> YoloModelLike:
    try:
        module = import_module("ultralytics")
    except ModuleNotFoundError as error:
        raise YoloError(
            'YOLO dependencies are missing; install with: pip install -e ".[webcam,yolo]"'
        ) from error
    constructor = cast(YoloConstructor, module.YOLO)
    return constructor(model)


def _implementation_version() -> str:
    try:
        return f"ultralytics/{version('ultralytics')}"
    except PackageNotFoundError:
        return "ultralytics/unknown"


def _compute_placement(device: str | None) -> ComputePlacement:
    if device is None:
        return ComputePlacement(ComputeKind.UNKNOWN, "ultralytics-unreported")
    normalized = device.lower()
    if normalized == "cpu":
        return ComputePlacement(ComputeKind.HOST_CPU, "cpu")
    if normalized.startswith("mps"):
        return ComputePlacement(ComputeKind.HOST_GPU, "apple-mps")
    if normalized.startswith("cuda") or normalized.isdigit():
        return ComputePlacement(ComputeKind.HOST_GPU, device)
    return ComputePlacement(ComputeKind.UNKNOWN, device)


class UltralyticsYoloDetector:
    def __init__(
        self,
        config: YoloConfig,
        *,
        model_factory: ModelFactory = _default_model_factory,
        clock_ns: ClockReader = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._clock_ns = clock_ns
        try:
            self._model = model_factory(config.model)
        except YoloError:
            raise
        except Exception as error:
            raise YoloError(f"could not load YOLO model {config.model!r}: {error}") from error

    def detect(self, image: ImageFrame, source: SampleHeader) -> DetectionBatch:
        if image.pixel_format is not PixelFormat.BGR8:
            raise YoloError("Ultralytics adapter currently requires a BGR8 source image")
        if source.received_at.clock.kind is not ClockKind.HOST_MONOTONIC:
            raise YoloError("detector timing requires a host-monotonic receipt clock")

        started_ns = self._clock_ns()
        try:
            raw_results = self._model.predict(
                image.data,
                device=self.config.device,
                conf=self.config.confidence,
                iou=self.config.iou,
                imgsz=self.config.image_size,
                max_det=self.config.max_detections,
                verbose=False,
            )
            if len(raw_results) != 1:
                raise YoloError(f"expected one image result, received {len(raw_results)}")
            result = cast(ResultLike, raw_results[0])
            detections, actual_device = self._extract_detections(result)
        except YoloError:
            raise
        except Exception as error:
            raise YoloError(f"YOLO inference failed: {error}") from error
        finished_ns = self._clock_ns()

        speed = result.speed
        return DetectionBatch(
            source=source,
            image_width=image.width,
            image_height=image.height,
            detections=detections,
            model_id=self.config.model,
            implementation=_implementation_version(),
            produced_on=_compute_placement(actual_device),
            started_at=TimePoint(started_ns, source.received_at.clock),
            finished_at=TimePoint(finished_ns, source.received_at.clock),
            timing=DetectionTiming(
                preprocess_ms=float(speed.get("preprocess", 0.0)),
                inference_ms=float(speed.get("inference", 0.0)),
                postprocess_ms=float(speed.get("postprocess", 0.0)),
                wall_ms=(finished_ns - started_ns) / 1_000_000,
            ),
        )

    @staticmethod
    def _extract_detections(result: ResultLike) -> tuple[tuple[Detection2D, ...], str | None]:
        if result.boxes is None:
            return (), None
        actual_device = str(result.boxes.data.device)
        rows = np.asarray(result.boxes.data.cpu().numpy(), dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] < 6:
            raise YoloError(f"unexpected YOLO box matrix shape: {rows.shape}")

        detections = []
        for row in rows:
            class_id = int(row[5])
            label = result.names.get(class_id, f"class-{class_id}")
            detections.append(
                Detection2D(
                    box=BoundingBox2D(
                        x_min=max(0.0, float(row[0])),
                        y_min=max(0.0, float(row[1])),
                        x_max=float(row[2]),
                        y_max=float(row[3]),
                    ),
                    class_id=class_id,
                    label=label,
                    confidence=float(row[4]),
                )
            )
        return tuple(detections), actual_device
