"""Interactive webcam host for the five DINOv2 showcase capabilities."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from vision_pipeline.config import (
    DinoDemoAppConfig,
    FeatureExtractorAppConfig,
    PointCloudAppConfig,
    WebcamAppConfig,
)
from vision_pipeline.contracts import FrameId, SensorSample
from vision_pipeline.geometry import (
    GeometryKind,
    PinholeIntrinsics,
    depth_to_point_cloud,
)
from vision_pipeline.geometry.visualization import (
    PointCloudRenderer,
    PointCloudViewConfig,
)
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception.contracts import FeatureBatch
from vision_pipeline.perception.dinov2 import DinoV2Config
from vision_pipeline.perception.dinov2.depth import (
    DinoV2DepthConfig,
    TorchDinoV2DepthEstimator,
)
from vision_pipeline.perception.dinov2.matching import (
    ImagePoint,
    match_reference_patch,
)
from vision_pipeline.perception.dinov2.pytorch import TorchDinoV2FeatureExtractor
from vision_pipeline.perception.dinov2.retrieval import DinoFeatureIndex
from vision_pipeline.perception.dinov2.segmentation import (
    DinoV2SegmentationConfig,
    TorchDinoV2SemanticSegmenter,
)
from vision_pipeline.perception.dinov2.task_visualization import (
    draw_point,
    render_depth,
    render_semantic_segmentation,
)
from vision_pipeline.perception.dinov2.visualization import DinoPcaVisualizer
from vision_pipeline.sources.opencv_webcam import (
    OpenCvWebcam,
    WebcamBackend,
    WebcamConfig,
)

_SOURCE_WINDOW = "DINOv2 source"
_OUTPUT_WINDOW = "DINOv2 output"
_REFERENCE_WINDOW = "DINOv2 reference / best retrieval"


@dataclass(slots=True)
class _Reference:
    sample: SensorSample[ImageFrame]
    features: FeatureBatch[torch.Tensor]
    pca_view: NDArray[np.uint8] | None = None


@dataclass(slots=True)
class _PointerState:
    point: ImagePoint | None = None
    image_width: int = 0
    image_height: int = 0
    mirrored: bool = False

    def callback(self, event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN or self.image_width <= 0 or self.image_height <= 0:
            return
        point_x = self.image_width - 1 - x if self.mirrored else x
        self.point = ImagePoint(
            float(min(max(point_x, 0), self.image_width - 1)),
            float(min(max(y, 0), self.image_height - 1)),
        )


def _annotate(view: NDArray[np.uint8], text: str, *, y: int = 32) -> None:
    cv2.putText(
        view,
        text,
        (16, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _display_image(image: ImageFrame, mirror: bool) -> NDArray[np.uint8]:
    view = image.data.copy()
    if image.pixel_format is PixelFormat.RGB8:
        view = np.ascontiguousarray(view[..., ::-1])
    if mirror:
        view = cv2.flip(view, 1)  # type: ignore[assignment]
    return view


def _show(name: str, view: NDArray[np.uint8], mirror: bool) -> None:
    output = cv2.flip(view, 1) if mirror else view
    cv2.imshow(name, output)


def _create_windows(settings: DinoDemoAppConfig) -> None:
    flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
    for name in (_SOURCE_WINDOW, _OUTPUT_WINDOW, _REFERENCE_WINDOW):
        cv2.namedWindow(name, flags)
        cv2.resizeWindow(name, settings.window_width, settings.window_height)


def _new_reference(
    sample: SensorSample[ImageFrame],
    features: FeatureBatch[torch.Tensor],
    pca: DinoPcaVisualizer,
) -> _Reference:
    pca.reset()
    reference = _Reference(sample=sample, features=features)
    reference.pca_view = pca.render(sample.payload, features)
    return reference


def run_dino_webcam(
    webcam: WebcamAppConfig,
    features: FeatureExtractorAppConfig,
    demo: DinoDemoAppConfig,
    point_cloud: PointCloudAppConfig,
) -> None:
    """Run one DINO-only capability against live camera measurements."""

    camera = OpenCvWebcam(
        WebcamConfig(
            device_index=webcam.device_index,
            backend=WebcamBackend(webcam.backend),
            width=webcam.width,
            height=webcam.height,
            fps=webcam.fps,
            source_id=webcam.source_id,
            frame_id=FrameId(webcam.frame_id),
        )
    )

    depth_estimator: TorchDinoV2DepthEstimator | None = None
    segmenter: TorchDinoV2SemanticSegmenter | None = None
    extractor: TorchDinoV2FeatureExtractor | None = None
    if demo.mode in {"depth", "point-cloud"}:
        depth_estimator = TorchDinoV2DepthEstimator(
            DinoV2DepthConfig(
                model=demo.depth_model,
                weights=demo.depth_weights,
                device=features.device,
                image_size=features.image_size,
            )
        )
    elif demo.mode == "semantic":
        segmenter = TorchDinoV2SemanticSegmenter(
            DinoV2SegmentationConfig(
                model=features.model,
                device=features.device,
                image_size=features.image_size,
            )
        )
    else:
        extractor = TorchDinoV2FeatureExtractor(
            DinoV2Config(
                model=features.model,
                device=features.device,
                image_size=features.image_size,
            )
        )

    pca = DinoPcaVisualizer()
    cloud_renderer = PointCloudRenderer(
        PointCloudViewConfig(
            width=demo.window_width,
            height=demo.window_height,
            point_size=point_cloud.point_size,
        )
    )
    cloud_intrinsics: PinholeIntrinsics | None = None
    first_cloud_reported = False
    reference: _Reference | None = None
    pointer = _PointerState(mirrored=webcam.mirror_preview)
    index = DinoFeatureIndex(demo.gallery_capacity)
    gallery_images: OrderedDict[str, NDArray[np.uint8]] = OrderedDict()
    frame_count = 0
    timings: list[float] = []

    if not webcam.headless:
        _create_windows(demo)
        cv2.setMouseCallback(_REFERENCE_WINDOW, pointer.callback)

    print(
        f"DINO-only mode={demo.mode} device={features.device} "
        f"input-long-edge={features.image_size}; YOLO/tracking are not loaded"
    )
    if demo.mode in {"dense-match", "sparse-match"}:
        print("controls: r=capture new reference, q/esc=quit")
    elif demo.mode == "retrieval":
        print("controls: a=add current frame to gallery, q/esc=quit")
    elif demo.mode == "point-cloud":
        print("controls: j/l=yaw i/k=pitch u/o=roll +/-=zoom r=reset q/esc=quit")

    try:
        with camera:
            info = camera.info
            print(
                f"webcam backend={info.backend} "
                f"reported={info.actual.width}x{info.actual.height}@{info.actual.fps:g}"
            )
            while webcam.max_frames == 0 or frame_count < webcam.max_frames:
                sample = camera.read()
                frame_count += 1
                source_view = _display_image(sample.payload, webcam.mirror_preview)
                output_view: NDArray[np.uint8]
                reference_view: NDArray[np.uint8] | None = None
                current_features: FeatureBatch[torch.Tensor] | None = None

                if depth_estimator is not None:
                    depth_prediction = depth_estimator.estimate(sample.payload, sample.header)
                    timings.append(depth_prediction.timing.wall_ms)
                    if demo.mode == "point-cloud":
                        if cloud_intrinsics is None:
                            if point_cloud.has_calibrated_intrinsics:
                                assert point_cloud.fx is not None
                                assert point_cloud.fy is not None
                                assert point_cloud.cx is not None
                                assert point_cloud.cy is not None
                                cloud_intrinsics = PinholeIntrinsics(
                                    width=sample.payload.width,
                                    height=sample.payload.height,
                                    fx=point_cloud.fx,
                                    fy=point_cloud.fy,
                                    cx=point_cloud.cx,
                                    cy=point_cloud.cy,
                                    calibration_id="configured/webcam",
                                )
                            else:
                                cloud_intrinsics = PinholeIntrinsics.from_horizontal_fov(
                                    sample.payload.width,
                                    sample.payload.height,
                                    point_cloud.horizontal_fov_degrees,
                                )
                        cloud = depth_to_point_cloud(
                            depth_prediction.depth_metres,
                            sample.payload,
                            sample.header,
                            cloud_intrinsics,
                            min_depth_metres=point_cloud.min_depth_metres,
                            max_depth_metres=point_cloud.max_depth_metres,
                            sampling_stride=point_cloud.sampling_stride,
                            geometry_kind=GeometryKind.ESTIMATED_METRIC,
                            produced_on=depth_prediction.produced_on,
                            memory=depth_prediction.memory,
                        )
                        output_view = cloud_renderer.render(cloud)
                        if not first_cloud_reported:
                            calibration = (
                                cloud_intrinsics.calibration_id or "approximate-horizontal-FOV"
                            )
                            print(
                                f"first cloud points={cloud.xyz.shape[0]} "
                                f"frame={cloud.source.frame_id.value} "
                                f"intrinsics={calibration} "
                                f"fx={cloud_intrinsics.fx:.2f} fy={cloud_intrinsics.fy:.2f}"
                            )
                            first_cloud_reported = True
                    else:
                        output_view = render_depth(depth_prediction)
                        _annotate(
                            output_view,
                            f"DINOv2 + {demo.depth_weights} metric-depth head  "
                            f"{depth_prediction.timing.wall_ms:.1f} ms",
                            y=64,
                        )
                elif segmenter is not None:
                    semantic_prediction = segmenter.segment(sample.payload, sample.header)
                    output_view = render_semantic_segmentation(
                        sample.payload,
                        semantic_prediction,
                    )
                    timings.append(semantic_prediction.timing.wall_ms)
                    _annotate(
                        output_view,
                        f"DINOv2 + ADE20K linear semantic head  "
                        f"{semantic_prediction.timing.wall_ms:.1f} ms",
                        y=64,
                    )
                else:
                    assert extractor is not None
                    current_features = extractor.extract(sample.payload, sample.header)
                    timings.append(current_features.timing.wall_ms)
                    if demo.mode == "dense-match":
                        if reference is None:
                            reference = _new_reference(sample, current_features, pca)
                        assert reference.pca_view is not None
                        output_view = pca.render(sample.payload, current_features)
                        reference_view = reference.pca_view.copy()
                        _annotate(reference_view, "reference patch features (shared PCA colors)")
                        _annotate(
                            output_view,
                            "current patch features: equal colors ~= corresponding features",
                        )
                    elif demo.mode == "sparse-match":
                        if reference is None:
                            reference = _new_reference(sample, current_features, pca)
                            pointer.point = ImagePoint(
                                sample.payload.width / 2,
                                sample.payload.height / 2,
                            )
                        assert pointer.point is not None
                        match = match_reference_patch(
                            reference.features,
                            current_features,
                            pointer.point,
                        )
                        reference_view = draw_point(
                            reference.sample.payload.data,
                            match.reference_point,
                            color=(0, 255, 255),
                            label="selected patch",
                        )
                        output_view = draw_point(
                            sample.payload.data,
                            match.matched_point,
                            color=(0, 255, 255),
                            label=f"best cosine={match.cosine_similarity:.3f}",
                        )
                        _annotate(reference_view, "click a different reference point")
                    else:
                        matches = index.query(current_features)
                        output_view = sample.payload.data.copy()
                        if matches:
                            best = matches[0]
                            reference_view = gallery_images[best.item_id].copy()
                            _annotate(
                                output_view,
                                f"nearest={best.item_id} cosine={best.cosine_similarity:.3f}",
                            )
                            _annotate(reference_view, f"gallery match: {best.item_id}")
                        else:
                            _annotate(output_view, "gallery empty: press a to add this view")

                _annotate(
                    source_view,
                    f"source seq={sample.header.sequence_number} | mode={demo.mode}",
                )
                if not webcam.headless:
                    cv2.imshow(_SOURCE_WINDOW, source_view)
                    _show(
                        _OUTPUT_WINDOW,
                        output_view,
                        webcam.mirror_preview and demo.mode != "point-cloud",
                    )
                    if reference_view is not None:
                        _show(_REFERENCE_WINDOW, reference_view, webcam.mirror_preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
                    if demo.mode == "point-cloud":
                        if key == ord("j"):
                            cloud_renderer.rotate(yaw_degrees=-5)
                        elif key == ord("l"):
                            cloud_renderer.rotate(yaw_degrees=5)
                        elif key == ord("i"):
                            cloud_renderer.rotate(pitch_degrees=-5)
                        elif key == ord("k"):
                            cloud_renderer.rotate(pitch_degrees=5)
                        elif key == ord("u"):
                            cloud_renderer.rotate(roll_degrees=-5)
                        elif key == ord("o"):
                            cloud_renderer.rotate(roll_degrees=5)
                        elif key in (ord("+"), ord("=")):
                            cloud_renderer.change_zoom(1.12)
                        elif key in (ord("-"), ord("_")):
                            cloud_renderer.change_zoom(1 / 1.12)
                        elif key == ord("r"):
                            cloud_renderer.reset()
                    if (
                        key == ord("r")
                        and current_features is not None
                        and demo.mode in {"dense-match", "sparse-match"}
                    ):
                        reference = _new_reference(sample, current_features, pca)
                        pointer.image_width = sample.payload.width
                        pointer.image_height = sample.payload.height
                        pointer.point = ImagePoint(
                            sample.payload.width / 2,
                            sample.payload.height / 2,
                        )
                    if (
                        key == ord("a")
                        and current_features is not None
                        and demo.mode == "retrieval"
                    ):
                        item_id = f"frame-{sample.header.sequence_number}"
                        index.add(item_id, current_features)
                        gallery_images[item_id] = sample.payload.data.copy()
                        while len(gallery_images) > demo.gallery_capacity:
                            gallery_images.popitem(last=False)
                        print(f"gallery add {item_id}; size={len(index)}")
                elif demo.mode == "retrieval" and current_features is not None and len(index) == 0:
                    item_id = f"frame-{sample.header.sequence_number}"
                    index.add(item_id, current_features)
                    gallery_images[item_id] = sample.payload.data.copy()

                if reference is not None:
                    pointer.image_width = reference.sample.payload.width
                    pointer.image_height = reference.sample.payload.height
    finally:
        if not webcam.headless:
            cv2.destroyAllWindows()

    mean_ms = sum(timings) / len(timings) if timings else 0.0
    print(f"processed {frame_count} frames; DINO mean wall={mean_ms:.2f} ms")


__all__ = ["run_dino_webcam"]
