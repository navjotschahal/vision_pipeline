"""Direct PyTorch YOLO inference for research and model-I/O exploration.

Ultralytics is used only to reconstruct the checkpoint's ``torch.nn.Module``. This
backend explicitly coordinates preprocessing, the forward call, prediction decoding,
NMS, and mapping boxes back to source-image coordinates.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Protocol, cast

import numpy as np
import torch
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
from vision_pipeline.perception.yolo.decoding import decode_predictions
from vision_pipeline.perception.yolo.preprocessing import LetterboxTransform, preprocess_bgr


class TorchDetectionModel(Protocol):
    names: Mapping[int, str]
    stride: torch.Tensor

    def to(self, device: torch.device) -> TorchDetectionModel: ...

    def eval(self) -> TorchDetectionModel: ...

    def __call__(self, image: torch.Tensor) -> object: ...


ModelLoader = Callable[[str], TorchDetectionModel]
ClockReader = Callable[[], int]


def _default_model_loader(checkpoint: str) -> TorchDetectionModel:
    """Load architecture and weights, but do not use the high-level predictor."""

    try:
        ultralytics = import_module("ultralytics")
    except ModuleNotFoundError as error:
        raise YoloError(
            'YOLO dependencies are missing; install with: pip install -e ".[webcam,yolo]"'
        ) from error

    wrapper = ultralytics.YOLO(checkpoint)
    if getattr(wrapper, "task", None) != "detect":
        raise YoloError(
            f"direct PyTorch YOLO currently supports detection checkpoints, got {wrapper.task!r}"
        )
    return cast(TorchDetectionModel, wrapper.model)


def _implementation_version() -> str:
    try:
        loader_version = version("ultralytics")
    except PackageNotFoundError:
        loader_version = "unknown"
    return f"pytorch-direct/{torch.__version__}+ultralytics-loader/{loader_version}"


def _normalize_device(name: str) -> torch.device:
    normalized = f"cuda:{name}" if name.isdigit() else name
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise YoloError(f"CUDA device {name!r} was requested but CUDA is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise YoloError("MPS was requested but is unavailable")
    return device


def _compute_placement(device: torch.device) -> ComputePlacement:
    if device.type == "cpu":
        return ComputePlacement(ComputeKind.HOST_CPU, "cpu")
    if device.type == "mps":
        return ComputePlacement(ComputeKind.HOST_GPU, "apple-mps")
    if device.type == "cuda":
        index = 0 if device.index is None else device.index
        return ComputePlacement(ComputeKind.HOST_GPU, f"cuda:{index}")
    return ComputePlacement(ComputeKind.UNKNOWN, str(device))


def _synchronize(device: torch.device) -> None:
    """Finish queued GPU work so phase timings measure execution, not dispatch."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _first_prediction(raw_output: object) -> torch.Tensor:
    candidate = raw_output[0] if isinstance(raw_output, tuple | list) else raw_output
    if not isinstance(candidate, torch.Tensor):
        raise YoloError(f"model returned unsupported prediction type {type(candidate).__name__}")
    return candidate


class TorchYoloDetector:
    """Research backend exposing the major operations hidden by ``YOLO.predict``."""

    def __init__(
        self,
        config: YoloConfig,
        *,
        model_loader: ModelLoader = _default_model_loader,
        clock_ns: ClockReader = time.monotonic_ns,
    ) -> None:
        self.config = config
        self._clock_ns = clock_ns
        self._device = _normalize_device(config.device)
        try:
            self._model = model_loader(config.model).to(self._device).eval()
            self._names = dict(self._model.names)
            self._stride = int(self._model.stride.max().item())
        except YoloError:
            raise
        except Exception as error:
            raise YoloError(f"could not load YOLO model {config.model!r}: {error}") from error
        if not self._names:
            raise YoloError("checkpoint has no class names")
        if self._stride <= 0:
            raise YoloError(f"checkpoint has invalid model stride {self._stride}")

    def detect(self, image: ImageFrame, source: SampleHeader) -> DetectionBatch:
        if image.pixel_format is not PixelFormat.BGR8:
            raise YoloError("direct PyTorch YOLO currently requires a BGR8 source image")
        if source.received_at.clock.kind is not ClockKind.HOST_MONOTONIC:
            raise YoloError("detector timing requires a host-monotonic receipt clock")

        started_ns = self._clock_ns()
        try:
            tensor, transform = preprocess_bgr(
                image.data, self.config.image_size, self._stride, self._device
            )
            _synchronize(self._device)
            preprocessed_ns = self._clock_ns()

            with torch.inference_mode():
                raw_output = self._model(tensor)
                prediction = _first_prediction(raw_output)
            _synchronize(self._device)
            inferred_ns = self._clock_ns()

            decoded = decode_predictions(
                prediction,
                class_count=len(self._names),
                confidence=self.config.confidence,
                iou=self.config.iou,
                max_detections=self.config.max_detections,
            )
            rows = decoded.detach().to(device="cpu", dtype=torch.float32).numpy()
            detections = self._to_source_detections(rows, transform, image.width, image.height)
            finished_ns = self._clock_ns()
        except YoloError:
            raise
        except Exception as error:
            raise YoloError(f"direct PyTorch YOLO inference failed: {error}") from error

        return DetectionBatch(
            source=source,
            image_width=image.width,
            image_height=image.height,
            detections=detections,
            model_id=self.config.model,
            implementation=_implementation_version(),
            produced_on=_compute_placement(prediction.device),
            started_at=TimePoint(started_ns, source.received_at.clock),
            finished_at=TimePoint(finished_ns, source.received_at.clock),
            timing=DetectionTiming(
                preprocess_ms=(preprocessed_ns - started_ns) / 1_000_000,
                inference_ms=(inferred_ns - preprocessed_ns) / 1_000_000,
                postprocess_ms=(finished_ns - inferred_ns) / 1_000_000,
                wall_ms=(finished_ns - started_ns) / 1_000_000,
            ),
        )

    def _to_source_detections(
        self,
        rows: NDArray[np.float32],
        transform: LetterboxTransform,
        source_width: int,
        source_height: int,
    ) -> tuple[Detection2D, ...]:
        detections: list[Detection2D] = []
        for row in rows:
            x_min = min(
                max((float(row[0]) - transform.pad_left) / transform.scale, 0.0),
                source_width,
            )
            y_min = min(
                max((float(row[1]) - transform.pad_top) / transform.scale, 0.0),
                source_height,
            )
            x_max = min(
                max((float(row[2]) - transform.pad_left) / transform.scale, 0.0),
                source_width,
            )
            y_max = min(
                max((float(row[3]) - transform.pad_top) / transform.scale, 0.0),
                source_height,
            )
            if x_max <= x_min or y_max <= y_min:
                continue
            class_id = int(row[5])
            detections.append(
                Detection2D(
                    box=BoundingBox2D(x_min, y_min, x_max, y_max),
                    class_id=class_id,
                    label=self._names.get(class_id, f"class-{class_id}"),
                    confidence=float(row[4]),
                )
            )
        return tuple(detections)
