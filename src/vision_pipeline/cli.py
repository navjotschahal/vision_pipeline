"""Command-line applications for exercising the perception pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from vision_pipeline.config import (
    ConfigError,
    PipelineAppConfig,
    load_pipeline_config,
    merge_dino_demo_overrides,
    merge_feature_overrides,
    merge_fusion_experiment_overrides,
    merge_point_cloud_overrides,
    merge_runtime_overrides,
    merge_stereo_calibration_overrides,
    merge_tracking_overrides,
    merge_webcam_overrides,
    merge_yolo_overrides,
)
from vision_pipeline.perception.dinov2 import DinoV2Error
from vision_pipeline.sources.camera_profiles import (
    CameraDevice,
    CameraProbeError,
    probe_macos_cameras,
    render_macos_camera_report,
)

if TYPE_CHECKING:
    import torch

    from vision_pipeline.contracts import SensorSample
    from vision_pipeline.image import ImageFrame
    from vision_pipeline.perception.contracts import FeatureBatch, FeatureExtractor
    from vision_pipeline.perception.dinov2.visualization import DinoPcaVisualizer
    from vision_pipeline.perception.regions import RegionDescriptorBatch
    from vision_pipeline.perception.tracking import GreedyAppearanceTracker, TrackingBatch


def _format_requested(value: int | float | None) -> str:
    return "default" if value is None else f"{value:g}"


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * fraction)
    return ordered[index]


def _print_camera_devices(devices: Sequence[CameraDevice]) -> None:
    if not devices:
        print("No AVFoundation video cameras found.")
        return
    for device in devices:
        status = "connected" if device.is_connected else "disconnected"
        if device.is_suspended:
            status += ", suspended"
        print(f"[{device.index}] {device.name} ({status})")
        print(f"    id={device.unique_id}")
        seen: set[tuple[int, int, float, float, str]] = set()
        profiles = sorted(
            device.profiles,
            key=lambda profile: (profile.pixels, profile.max_fps),
            reverse=True,
        )
        for profile in profiles:
            identity = (
                profile.width,
                profile.height,
                profile.min_fps,
                profile.max_fps,
                profile.pixel_format,
            )
            if identity in seen:
                continue
            seen.add(identity)
            print(
                f"    {profile.width}x{profile.height} "
                f"{profile.min_fps:g}-{profile.max_fps:g} fps "
                f"native={profile.pixel_format}"
            )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vision-pipeline")
    commands = parser.add_subparsers(dest="command", required=True)

    pose = commands.add_parser("human-pose", help="preview selected human limb links")
    pose.add_argument("--camera", type=int, default=0)
    pose.add_argument("--model", default="yolo11n-pose.pt")
    pose.add_argument("--device", help="inference device, e.g. cpu, mps, cuda:0")
    pose.add_argument("--person-index", type=int, default=0)
    pose.add_argument(
        "--chain",
        action="append",
        default=[],
        metavar="NAME:JOINT,JOINT,...",
        help="repeat to select chains; defaults to both arms",
    )
    pose.add_argument("--max-frames", type=int, default=0)
    pose.add_argument("--headless", action="store_true")
    pose.add_argument("--output", type=Path, help="write selected links as JSONL")

    kuka = commands.add_parser(
        "kuka-webcam-sim", help="webcam planar teleop of CPF's KUKA MuJoCo arm"
    )
    kuka.add_argument("--poc-root", type=Path, help="path to Navjot_IS/poc")
    kuka.add_argument("--camera", type=int, default=0)
    kuka.add_argument("--model", default="yolo11n-pose.pt")
    kuka.add_argument("--device", help="pose inference device")
    kuka.add_argument("--headless", action="store_true")
    kuka.add_argument("--max-seconds", type=float, default=0)

    webcam = commands.add_parser("webcam", help="capture the laptop webcam")
    webcam.add_argument("--config", help="YAML configuration file")
    webcam.add_argument("--device", dest="device_index", type=int, help="OpenCV camera index")
    webcam.add_argument("--width", type=int)
    webcam.add_argument("--height", type=int)
    webcam.add_argument("--fps", type=float)
    webcam.add_argument(
        "--backend",
        choices=("auto", "avfoundation", "v4l2", "msmf"),
    )
    webcam.add_argument("--frame-id")
    webcam.add_argument("--source-id")
    webcam.add_argument(
        "--probe",
        action="store_true",
        default=None,
        help="list native AVFoundation formats and exit",
    )
    webcam.add_argument(
        "--probe-report",
        metavar="PATH",
        help="write a dated Markdown capability inventory and exit",
    )
    webcam.add_argument("--max-frames", type=int, help="zero means unlimited")
    webcam.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="capture without a window",
    )
    webcam.add_argument(
        "--mirror-preview",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="mirror only the preview; emitted measurements remain unmodified",
    )
    webcam.add_argument(
        "--detect",
        dest="enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run YOLO object detection",
    )
    webcam.add_argument(
        "--detector-backend",
        dest="detector_backend",
        choices=("torch", "ultralytics"),
        help="YOLO execution path",
    )
    webcam.add_argument("--model", help="YOLO model checkpoint or path")
    webcam.add_argument(
        "--inference-device",
        dest="device",
        help="inference device such as cpu, mps, or cuda:0",
    )
    webcam.add_argument("--confidence", type=float)
    webcam.add_argument("--iou", type=float)
    webcam.add_argument("--image-size", type=int)
    webcam.add_argument("--max-detections", type=int)
    webcam.add_argument(
        "--detection-mode",
        choices=("latest", "synchronous"),
        help="latest preserves freshness; synchronous is useful for benchmarks",
    )
    webcam.add_argument(
        "--extract-features",
        dest="features_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run dense image feature extraction",
    )
    webcam.add_argument(
        "--feature-backend",
        choices=("dinov2-torch",),
    )
    webcam.add_argument("--feature-model", help="feature backbone name")
    webcam.add_argument(
        "--feature-device",
        help="feature inference device such as cpu, mps, or cuda:0",
    )
    webcam.add_argument("--feature-image-size", type=int, help="model-input long edge")
    webcam.add_argument(
        "--feature-visualization",
        choices=("none", "pca"),
        help="diagnostic dense-feature visualization",
    )
    webcam.add_argument(
        "--track",
        dest="tracking_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="assign persistent IDs using YOLO boxes and DINO descriptors",
    )
    webcam.add_argument("--tracking-appearance-weight", type=float)
    webcam.add_argument("--tracking-minimum-cosine", type=float)
    webcam.add_argument("--tracking-minimum-iou", type=float)
    webcam.add_argument("--tracking-minimum-score", type=float)
    webcam.add_argument("--tracking-descriptor-momentum", type=float)
    webcam.add_argument("--tracking-max-missed-frames", type=int)
    webcam.add_argument(
        "--tracking-class-aware",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    dino = commands.add_parser(
        "dino-webcam",
        help="run detector-free DINOv2 capability demos on the laptop webcam",
    )
    dino.add_argument("--config", help="YAML configuration file")
    dino.add_argument(
        "--demo",
        dest="mode",
        choices=(
            "point-cloud",
            "depth",
            "semantic",
            "retrieval",
            "dense-match",
            "sparse-match",
        ),
    )
    dino.add_argument("--device", dest="device_index", type=int, help="OpenCV camera index")
    dino.add_argument("--width", type=int)
    dino.add_argument("--height", type=int)
    dino.add_argument("--fps", type=float)
    dino.add_argument("--backend", choices=("auto", "avfoundation", "v4l2", "msmf"))
    dino.add_argument("--frame-id")
    dino.add_argument("--source-id")
    dino.add_argument("--max-frames", type=int, help="zero means unlimited")
    dino.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    dino.add_argument(
        "--mirror-preview",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    dino.add_argument("--feature-model", help="DINOv2 backbone name")
    dino.add_argument("--feature-device", help="cpu, mps, cuda:0, or a CUDA index")
    dino.add_argument("--feature-image-size", type=int, help="model-input long edge")
    dino.add_argument("--depth-model", help="official DINOv2 depth model name")
    dino.add_argument("--depth-weights", choices=("NYU", "KITTI"))
    dino.add_argument("--gallery-capacity", type=int)
    dino.add_argument("--window-width", type=int)
    dino.add_argument("--window-height", type=int)
    dino.add_argument("--horizontal-fov", dest="horizontal_fov_degrees", type=float)
    dino.add_argument("--fx", type=float)
    dino.add_argument("--fy", type=float)
    dino.add_argument("--cx", type=float)
    dino.add_argument("--cy", type=float)
    dino.add_argument("--point-stride", dest="sampling_stride", type=int)
    dino.add_argument("--min-depth", dest="min_depth_metres", type=float)
    dino.add_argument("--max-depth", dest="max_depth_metres", type=float)
    dino.add_argument("--point-size", type=int)

    calibration = commands.add_parser(
        "calibrate-stereo",
        help="automatically calibrate a ChArUco-observed camera pair",
    )
    calibration.add_argument("--config", help="YAML configuration file")
    calibration.add_argument("--camera-a", dest="camera_a_device_index", type=int)
    calibration.add_argument("--camera-b", dest="camera_b_device_index", type=int)
    calibration.add_argument("--backend", choices=("auto", "avfoundation", "v4l2", "msmf"))
    calibration.add_argument("--width", type=int)
    calibration.add_argument("--height", type=int)
    calibration.add_argument("--fps", type=float)
    calibration.add_argument("--required-pairs", type=int)
    calibration.add_argument("--output", dest="output_path")
    calibration.add_argument(
        "--generate-board",
        action="store_true",
        help="generate the configured printable board and exit",
    )

    fusion = commands.add_parser(
        "fusion-sim",
        help="simulate IMU prediction plus two calibrated RGB-D corrections",
    )
    fusion.add_argument("--config", help="YAML configuration file")
    fusion.add_argument("--duration", dest="duration_seconds", type=float)
    fusion.add_argument("--seed", type=int)
    fusion.add_argument("--imu-hz", type=int)
    fusion.add_argument("--camera-a-hz", type=int)
    fusion.add_argument("--camera-b-hz", type=int)

    iphone = commands.add_parser(
        "iphone-receiver",
        help="receive, inspect, and optionally record iPhone IMU plus RGB-D",
    )
    iphone.add_argument("--bind", dest="bind_host", default="0.0.0.0")
    iphone.add_argument("--imu-port", type=int, default=5001)
    iphone.add_argument("--max-frames", type=int, default=0, help="zero means unlimited")
    iphone.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="disable the live RGB/depth preview",
    )
    iphone.add_argument("--record-dir", type=Path)
    iphone.add_argument(
        "--minimum-depth-confidence",
        type=int,
        choices=(0, 1, 2),
        default=1,
    )
    cloud_view = iphone.add_mutually_exclusive_group()
    cloud_view.add_argument(
        "--raw-cloud",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show only the latest unaccumulated camera-optical point cloud",
    )
    cloud_view.add_argument(
        "--world-cloud",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show a temporally accumulated ARKit-world point cloud",
    )
    iphone.add_argument("--point-stride", type=int, default=2)
    iphone.add_argument("--voxel-size", dest="voxel_size_metres", type=float, default=0.02)
    iphone.add_argument("--max-world-voxels", type=int, default=250_000)
    return parser


def _run_stereo_calibration(args: argparse.Namespace) -> int:
    try:
        base = load_pipeline_config(args.config) if args.config else PipelineAppConfig()
        settings = merge_stereo_calibration_overrides(
            base.stereo_calibration,
            {
                "camera_a_device_index": args.camera_a_device_index,
                "camera_b_device_index": args.camera_b_device_index,
                "backend": args.backend,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "required_pairs": args.required_pairs,
                "output_path": args.output_path,
            },
        )
        from vision_pipeline.apps.stereo_calibration import (
            generate_calibration_board,
            run_stereo_calibration,
        )
        from vision_pipeline.sources.opencv_webcam import WebcamError

        if args.generate_board:
            path = generate_calibration_board(settings)
            print(f"wrote printable ChArUco board: {path}")
            print(
                "print without scaling, measure one square, and update "
                "square_length_metres if it is not exactly the configured size"
            )
            return 0
        result = run_stereo_calibration(settings)
        if result is None:
            print("calibration aborted; no calibration file was written")
            return 1
        return 0
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except ModuleNotFoundError:
        print(
            "Calibration dependencies are missing. Install with: "
            'python3 -m pip install -e ".[webcam]"',
            file=sys.stderr,
        )
        return 2
    except (OSError, ValueError, WebcamError) as error:
        print(f"stereo calibration error: {error}", file=sys.stderr)
        return 2


def _run_fusion_simulation(args: argparse.Namespace) -> int:
    try:
        base = load_pipeline_config(args.config) if args.config else PipelineAppConfig()
        settings = merge_fusion_experiment_overrides(
            base.fusion_experiment,
            {
                "duration_seconds": args.duration_seconds,
                "seed": args.seed,
                "imu_hz": args.imu_hz,
                "camera_a_hz": args.camera_a_hz,
                "camera_b_hz": args.camera_b_hz,
            },
        )
        from vision_pipeline.apps.fusion_simulation import run_fusion_simulation

        run_fusion_simulation(settings)
        return 0
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2
    except ModuleNotFoundError:
        print(
            'Fusion dependencies are missing. Install with: python3 -m pip install -e ".[fusion]"',
            file=sys.stderr,
        )
        return 2
    except ValueError as error:
        print(f"fusion simulation error: {error}", file=sys.stderr)
        return 2


def _run_iphone_receiver(args: argparse.Namespace) -> int:
    try:
        from vision_pipeline.apps.iphone_receiver import (
            IphoneReceiverConfig,
            run_iphone_receiver,
        )

        return run_iphone_receiver(
            IphoneReceiverConfig(
                bind_host=args.bind_host,
                imu_port=args.imu_port,
                max_frames=args.max_frames,
                headless=args.headless,
                record_directory=args.record_dir,
                minimum_depth_confidence=args.minimum_depth_confidence,
                raw_cloud=args.raw_cloud,
                world_cloud=args.world_cloud,
                point_stride=args.point_stride,
                voxel_size_metres=args.voxel_size_metres,
                max_world_voxels=args.max_world_voxels,
            )
        )
    except ModuleNotFoundError:
        print(
            "iPhone receiver dependencies are missing. Install with: "
            'python3 -m pip install -e ".[iphone]"',
            file=sys.stderr,
        )
        return 2
    except (FileExistsError, OSError, ValueError) as error:
        print(f"iPhone receiver error: {error}", file=sys.stderr)
        return 2


def _run_dino_webcam(args: argparse.Namespace) -> int:
    try:
        base_config = load_pipeline_config(args.config) if args.config else PipelineAppConfig()
        webcam_settings = merge_webcam_overrides(
            base_config.webcam,
            {
                "device_index": args.device_index,
                "backend": args.backend,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "frame_id": args.frame_id,
                "source_id": args.source_id,
                "max_frames": args.max_frames,
                "headless": args.headless,
                "mirror_preview": args.mirror_preview,
            },
        )
        feature_settings = merge_feature_overrides(
            base_config.features,
            {
                "model": args.feature_model,
                "device": args.feature_device,
                "image_size": args.feature_image_size,
            },
        )
        demo_settings = merge_dino_demo_overrides(
            base_config.dino_demo,
            {
                "mode": args.mode,
                "depth_model": args.depth_model,
                "depth_weights": args.depth_weights,
                "gallery_capacity": args.gallery_capacity,
                "window_width": args.window_width,
                "window_height": args.window_height,
            },
        )
        point_cloud_settings = merge_point_cloud_overrides(
            base_config.point_cloud,
            {
                "horizontal_fov_degrees": args.horizontal_fov_degrees,
                "fx": args.fx,
                "fy": args.fy,
                "cx": args.cx,
                "cy": args.cy,
                "sampling_stride": args.sampling_stride,
                "min_depth_metres": args.min_depth_metres,
                "max_depth_metres": args.max_depth_metres,
                "point_size": args.point_size,
            },
        )
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    try:
        from vision_pipeline.apps.dino_webcam import run_dino_webcam
        from vision_pipeline.sources.opencv_webcam import WebcamError

        run_dino_webcam(
            webcam_settings,
            feature_settings,
            demo_settings,
            point_cloud_settings,
        )
    except ModuleNotFoundError:
        print(
            "DINO webcam dependencies are missing. Install with: "
            'python3 -m pip install -e ".[webcam,dino]"',
            file=sys.stderr,
        )
        return 2
    except (DinoV2Error, WebcamError, ValueError) as error:
        print(f"DINO webcam error: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        pass
    return 0


def _run_webcam(args: argparse.Namespace) -> int:
    try:
        base_config = load_pipeline_config(args.config) if args.config else PipelineAppConfig()
        settings = merge_webcam_overrides(
            base_config.webcam,
            {
                "device_index": args.device_index,
                "backend": args.backend,
                "width": args.width,
                "height": args.height,
                "fps": args.fps,
                "frame_id": args.frame_id,
                "source_id": args.source_id,
                "probe": args.probe,
                "probe_report": args.probe_report,
                "max_frames": args.max_frames,
                "headless": args.headless,
                "mirror_preview": args.mirror_preview,
            },
        )
        detector_settings = merge_yolo_overrides(
            base_config.detector,
            {
                "enabled": args.enabled,
                "backend": args.detector_backend,
                "model": args.model,
                "device": args.device,
                "confidence": args.confidence,
                "iou": args.iou,
                "image_size": args.image_size,
                "max_detections": args.max_detections,
            },
        )
        feature_settings = merge_feature_overrides(
            base_config.features,
            {
                "enabled": args.features_enabled,
                "backend": args.feature_backend,
                "model": args.feature_model,
                "device": args.feature_device,
                "image_size": args.feature_image_size,
                "visualization": args.feature_visualization,
            },
        )
        tracking_settings = merge_tracking_overrides(
            base_config.tracking,
            {
                "enabled": args.tracking_enabled,
                "appearance_weight": args.tracking_appearance_weight,
                "minimum_cosine_similarity": args.tracking_minimum_cosine,
                "minimum_iou": args.tracking_minimum_iou,
                "minimum_score": args.tracking_minimum_score,
                "descriptor_momentum": args.tracking_descriptor_momentum,
                "max_missed_frames": args.tracking_max_missed_frames,
                "class_aware": args.tracking_class_aware,
            },
        )
        runtime_settings = merge_runtime_overrides(
            base_config.runtime,
            {"detection_mode": args.detection_mode},
        )
    except ConfigError as error:
        print(f"configuration error: {error}", file=sys.stderr)
        return 2

    if (
        detector_settings.enabled
        and feature_settings.enabled
        and runtime_settings.detection_mode != "synchronous"
    ):
        print(
            "configuration error: YOLO-to-DINO region fusion requires "
            "runtime.detection_mode=synchronous so both results reference the same frame",
            file=sys.stderr,
        )
        return 2
    if tracking_settings.enabled and not (detector_settings.enabled and feature_settings.enabled):
        print(
            "configuration error: tracking requires both detector.enabled=true "
            "and features.enabled=true",
            file=sys.stderr,
        )
        return 2

    try:
        devices = probe_macos_cameras() if settings.probe or settings.probe_report else ()
        if settings.probe:
            _print_camera_devices(devices)
        if settings.probe_report:
            report_path = Path(settings.probe_report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(render_macos_camera_report(devices), encoding="utf-8")
            print(f"wrote camera capability report: {report_path}")
        if settings.probe or settings.probe_report:
            return 0
    except CameraProbeError as error:
        print(f"camera probe error: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"cannot write camera capability report: {error}", file=sys.stderr)
        return 2

    try:
        import cv2

        from vision_pipeline.contracts import FrameId
        from vision_pipeline.image import ImageFrame, PixelFormat
        from vision_pipeline.perception.contracts import DetectionBatch, Detector
        from vision_pipeline.perception.tracking.config import TrackingError
        from vision_pipeline.perception.tracking.visualization import draw_tracks
        from vision_pipeline.perception.visualization import draw_detections
        from vision_pipeline.perception.yolo import YoloConfig, YoloError
        from vision_pipeline.perception.yolo.ultralytics import UltralyticsYoloDetector
        from vision_pipeline.runtime import (
            LatestFrameDetectionPipeline,
            PipelineWorkerError,
            ProcessedFrame,
        )
        from vision_pipeline.sources.opencv_webcam import (
            OpenCvWebcam,
            WebcamBackend,
            WebcamConfig,
            WebcamError,
        )
    except ModuleNotFoundError:
        print(
            'Webcam dependencies are missing. Install with: python3 -m pip install -e ".[webcam]"',
            file=sys.stderr,
        )
        return 2

    detector: Detector | None = None
    if detector_settings.enabled:
        try:
            yolo_config = YoloConfig(
                model=detector_settings.model,
                device=detector_settings.device,
                confidence=detector_settings.confidence,
                iou=detector_settings.iou,
                image_size=detector_settings.image_size,
                max_detections=detector_settings.max_detections,
            )
            if detector_settings.backend == "torch":
                from vision_pipeline.perception.yolo.pytorch import TorchYoloDetector

                detector = TorchYoloDetector(yolo_config)
            else:
                detector = UltralyticsYoloDetector(yolo_config)
        except (ValueError, YoloError) as error:
            print(f"detector error: {error}", file=sys.stderr)
            return 2
        print(
            f"detector backend={detector_settings.backend} model={detector_settings.model} "
            f"device={detector_settings.device} "
            f"imgsz={detector_settings.image_size} confidence={detector_settings.confidence:g}"
        )

    detection_pipeline: LatestFrameDetectionPipeline | None = None
    if detector is not None and runtime_settings.detection_mode == "latest":
        detection_pipeline = LatestFrameDetectionPipeline(detector)
        detection_pipeline.start()
    if detector is not None:
        print(f"detection scheduling={runtime_settings.detection_mode}")

    feature_extractor: FeatureExtractor[torch.Tensor] | None = None
    feature_visualizer: DinoPcaVisualizer | None = None
    if feature_settings.enabled:
        try:
            from vision_pipeline.perception.dinov2 import DinoV2Config
            from vision_pipeline.perception.dinov2.pytorch import TorchDinoV2FeatureExtractor
            from vision_pipeline.perception.dinov2.visualization import DinoPcaVisualizer
            from vision_pipeline.perception.regions import pool_detection_regions

            print(
                f"loading feature backend={feature_settings.backend} "
                f"model={feature_settings.model} device={feature_settings.device}"
            )
            feature_extractor = TorchDinoV2FeatureExtractor(
                DinoV2Config(
                    model=feature_settings.model,
                    device=feature_settings.device,
                    image_size=feature_settings.image_size,
                )
            )
            if feature_settings.visualization == "pca":
                feature_visualizer = DinoPcaVisualizer()
        except ModuleNotFoundError:
            print(
                "DINOv2 dependencies are missing. Install with: "
                'python3 -m pip install -e ".[webcam,dino]"',
                file=sys.stderr,
            )
            return 2
        except (ValueError, DinoV2Error) as error:
            print(f"feature extractor error: {error}", file=sys.stderr)
            return 2
        print(
            f"features input-long-edge={feature_settings.image_size} "
            f"visualization={feature_settings.visualization} scheduling=synchronous"
        )

    tracker: GreedyAppearanceTracker | None = None
    if tracking_settings.enabled:
        from vision_pipeline.perception.tracking import (
            AppearanceTrackerConfig,
            GreedyAppearanceTracker,
        )

        tracker = GreedyAppearanceTracker(
            AppearanceTrackerConfig(
                appearance_weight=tracking_settings.appearance_weight,
                minimum_cosine_similarity=tracking_settings.minimum_cosine_similarity,
                minimum_iou=tracking_settings.minimum_iou,
                minimum_score=tracking_settings.minimum_score,
                descriptor_momentum=tracking_settings.descriptor_momentum,
                max_missed_frames=tracking_settings.max_missed_frames,
                class_aware=tracking_settings.class_aware,
            )
        )
        print(
            "tracking association=greedy(dino-cosine+iou) "
            f"appearance-weight={tracking_settings.appearance_weight:g} "
            f"max-missed={tracking_settings.max_missed_frames}"
        )

    config = WebcamConfig(
        device_index=settings.device_index,
        backend=WebcamBackend(settings.backend),
        width=settings.width,
        height=settings.height,
        fps=settings.fps,
        source_id=settings.source_id,
        frame_id=FrameId(settings.frame_id),
    )

    camera = OpenCvWebcam(config)
    frame_count = 0
    first_receipt_ns: int | None = None
    latest_receipt_ns: int | None = None
    detection_frames = 0
    detection_wall_times: list[float] = []
    queue_wait_times: list[float] = []
    end_to_end_times: list[float] = []
    preprocess_ms = 0.0
    inference_ms = 0.0
    postprocess_ms = 0.0
    latest_detections: DetectionBatch | None = None
    latest_detection_sample: SensorSample[ImageFrame] | None = None
    first_detection_reported = False
    feature_frames = 0
    feature_wall_times: list[float] = []
    feature_preprocess_ms = 0.0
    feature_inference_ms = 0.0
    feature_postprocess_ms = 0.0
    latest_features: FeatureBatch[torch.Tensor] | None = None
    latest_feature_sample: SensorSample[ImageFrame] | None = None
    latest_region_descriptors: RegionDescriptorBatch[torch.Tensor] | None = None
    latest_tracking: TrackingBatch[torch.Tensor] | None = None
    first_feature_reported = False
    first_region_reported = False
    first_tracking_reported = False
    pipeline_close_error: PipelineWorkerError | None = None
    window_name = "vision-pipeline camera + tracking"
    feature_window_name = "vision-pipeline DINOv2 PCA"

    try:
        with camera:
            info = camera.info
            requested_mode = (
                "native-default"
                if info.requested.width is None
                and info.requested.height is None
                and info.requested.fps is None
                else (
                    f"{_format_requested(info.requested.width)}x"
                    f"{_format_requested(info.requested.height)}@"
                    f"{_format_requested(info.requested.fps)}"
                )
            )
            print(
                f"webcam backend={info.backend} "
                f"requested={requested_mode} "
                f"reported={info.actual.width}x{info.actual.height}@{info.actual.fps:g}"
            )
            print(f"property requests accepted={info.accepted}")
            mismatches = []
            if settings.width is not None and info.actual.width != settings.width:
                mismatches.append(f"width requested={settings.width} actual={info.actual.width}")
            if settings.height is not None and info.actual.height != settings.height:
                mismatches.append(f"height requested={settings.height} actual={info.actual.height}")
            if settings.fps is not None and (
                info.actual.fps is None or abs(info.actual.fps - settings.fps) > 0.5
            ):
                mismatches.append(f"fps requested={settings.fps:g} actual={info.actual.fps}")
            if mismatches:
                print(
                    "warning: camera did not negotiate all raw parameters: "
                    + "; ".join(mismatches),
                    file=sys.stderr,
                )

            if not settings.headless:
                window_flags = cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO
                cv2.namedWindow(window_name, window_flags)
                cv2.resizeWindow(window_name, 1280, 720)
                if feature_visualizer is not None:
                    cv2.namedWindow(feature_window_name, window_flags)
                    cv2.resizeWindow(feature_window_name, 1280, 720)

            while settings.max_frames == 0 or frame_count < settings.max_frames:
                sample = camera.read()
                receipt_ns = sample.header.received_at.nanoseconds
                first_receipt_ns = receipt_ns if first_receipt_ns is None else first_receipt_ns
                latest_receipt_ns = receipt_ns
                frame_count += 1

                processed: ProcessedFrame | None = None
                if detection_pipeline is not None:
                    detection_pipeline.submit(sample)
                    processed = detection_pipeline.take_latest()
                elif detector is not None:
                    latest_detections = detector.detect(sample.payload, sample.header)
                    latest_detection_sample = sample
                    detection_frames += 1
                    detection_wall_times.append(latest_detections.timing.wall_ms)
                    preprocess_ms += latest_detections.timing.preprocess_ms
                    inference_ms += latest_detections.timing.inference_ms
                    postprocess_ms += latest_detections.timing.postprocess_ms

                if processed is not None:
                    latest_detections = processed.detections
                    latest_detection_sample = processed.sample
                    detection_frames += 1
                    detection_wall_times.append(latest_detections.timing.wall_ms)
                    queue_wait_times.append(processed.queue_wait_ms)
                    end_to_end_times.append(processed.end_to_end_ms)
                    preprocess_ms += latest_detections.timing.preprocess_ms
                    inference_ms += latest_detections.timing.inference_ms
                    postprocess_ms += latest_detections.timing.postprocess_ms

                if feature_extractor is not None:
                    latest_features = feature_extractor.extract(sample.payload, sample.header)
                    latest_feature_sample = sample
                    feature_frames += 1
                    feature_wall_times.append(latest_features.timing.wall_ms)
                    feature_preprocess_ms += latest_features.timing.preprocess_ms
                    feature_inference_ms += latest_features.timing.inference_ms
                    feature_postprocess_ms += latest_features.timing.postprocess_ms
                    if (
                        latest_detections is not None
                        and latest_detections.source == latest_features.source
                    ):
                        latest_region_descriptors = pool_detection_regions(
                            latest_detections,
                            latest_features,
                        )
                        if tracker is not None:
                            latest_tracking = tracker.update(latest_region_descriptors)

                if frame_count == 1:
                    print(
                        f"first frame={sample.payload.width}x{sample.payload.height} "
                        f"format={sample.payload.pixel_format.value} "
                        f"stride={sample.payload.row_stride_bytes} "
                        f"bytes={sample.payload.size_bytes}"
                    )
                    print(
                        f"timing captured=unavailable "
                        f"received_clock={sample.header.received_at.clock.domain_id}"
                    )
                    print(
                        f"placement produced_on={sample.header.produced_on.kind.value} "
                        f"payload_memory={sample.payload_memory.kind.value}"
                    )
                if latest_detections is not None and not first_detection_reported:
                    print(
                        f"first detection batch objects={len(latest_detections.detections)} "
                        f"computed_on={latest_detections.produced_on.kind.value}/"
                        f"{latest_detections.produced_on.device_id} "
                        f"wall={latest_detections.timing.wall_ms:.2f} ms"
                    )
                    first_detection_reported = True
                if latest_features is not None and not first_feature_reported:
                    print(
                        "first feature batch "
                        f"global={tuple(latest_features.global_features.shape)} "
                        f"spatial={tuple(latest_features.spatial_features.shape)} "
                        f"grid={latest_features.geometry.grid_width}x"
                        f"{latest_features.geometry.grid_height} "
                        f"computed_on={latest_features.produced_on.kind.value}/"
                        f"{latest_features.produced_on.device_id} "
                        f"wall={latest_features.timing.wall_ms:.2f} ms"
                    )
                    first_feature_reported = True
                if latest_region_descriptors is not None and not first_region_reported:
                    print(
                        "first YOLO-to-DINO region batch "
                        f"objects={len(latest_region_descriptors.regions)} "
                        f"descriptors={tuple(latest_region_descriptors.descriptors.shape)} "
                        f"device={latest_region_descriptors.descriptors.device}"
                    )
                    first_region_reported = True
                if latest_tracking is not None and not first_tracking_reported:
                    track_ids = [item.track_id for item in latest_tracking.tracked_detections]
                    print(
                        f"first tracking batch ids={track_ids} "
                        f"active={latest_tracking.active_track_count}"
                    )
                    first_tracking_reported = True

                elapsed_ns = latest_receipt_ns - first_receipt_ns
                measured_fps = (frame_count - 1) * 1_000_000_000 / elapsed_ns if elapsed_ns else 0.0

                if not settings.headless:
                    preview_sample = latest_feature_sample or latest_detection_sample or sample
                    matching_detections = (
                        latest_detections
                        if latest_detections is not None
                        and latest_detections.source == preview_sample.header
                        else None
                    )
                    matching_tracking = (
                        latest_tracking
                        if latest_tracking is not None
                        and latest_tracking.regions.detections.source == preview_sample.header
                        else None
                    )
                    if matching_tracking is not None:
                        preview = draw_tracks(preview_sample.payload, matching_tracking)
                    elif matching_detections is not None:
                        preview = draw_detections(preview_sample.payload, matching_detections)
                    else:
                        preview = preview_sample.payload.data.copy()
                    if settings.mirror_preview:
                        # OpenCV's stub widens the dtype even though flip preserves uint8.
                        preview = cv2.flip(preview, 1)  # type: ignore[assignment]
                    overlay = (
                        f"capture_seq={sample.header.sequence_number} "
                        f"display_seq={preview_sample.header.sequence_number} "
                        f"{preview_sample.payload.width}x{preview_sample.payload.height} "
                        f"BGR8 host {measured_fps:.1f} fps"
                    )
                    if matching_detections is not None:
                        overlay += (
                            f" det={len(matching_detections.detections)} "
                            f"infer={matching_detections.timing.wall_ms:.1f}ms"
                        )
                    if latest_features is not None:
                        overlay += f" dino={latest_features.timing.wall_ms:.1f}ms"
                    if matching_tracking is not None:
                        overlay += (
                            f" tracks={len(matching_tracking.tracked_detections)}/"
                            f"{matching_tracking.active_track_count}"
                        )
                    cv2.putText(
                        preview,
                        overlay,
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    if (
                        feature_visualizer is not None
                        and latest_features is not None
                        and latest_feature_sample is not None
                    ):
                        feature_preview = feature_visualizer.render(
                            latest_feature_sample.payload,
                            latest_features,
                        )
                        if matching_tracking is not None:
                            feature_preview = draw_tracks(
                                ImageFrame(feature_preview, PixelFormat.BGR8),
                                matching_tracking,
                            )
                        elif matching_detections is not None:
                            feature_preview = draw_detections(
                                ImageFrame(feature_preview, PixelFormat.BGR8),
                                matching_detections,
                            )
                        cv2.putText(
                            feature_preview,
                            "DINOv2 patch features (PCA)",
                            (16, 32),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )
                        cv2.imshow(feature_window_name, feature_preview)
                    cv2.imshow(window_name, preview)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q")):
                        break
    except (
        DinoV2Error,
        PipelineWorkerError,
        TrackingError,
        WebcamError,
        YoloError,
    ) as error:
        if isinstance(error, WebcamError):
            component = "webcam"
        elif isinstance(error, DinoV2Error):
            component = "feature extractor"
        elif isinstance(error, TrackingError):
            component = "tracker"
        else:
            component = "detector"
        print(f"{component} error: {error}", file=sys.stderr)
        if sys.platform == "darwin":
            print(
                "On macOS, grant Camera access to your terminal or IDE in "
                "System Settings > Privacy & Security > Camera.",
                file=sys.stderr,
            )
        return 2
    except KeyboardInterrupt:
        pass
    finally:
        if detection_pipeline is not None:
            try:
                detection_pipeline.close(drain=True)
            except PipelineWorkerError as error:
                pipeline_close_error = error
        if not settings.headless:
            cv2.destroyAllWindows()

    if pipeline_close_error is not None:
        print(f"detector error: {pipeline_close_error}", file=sys.stderr)
        return 2

    if detection_pipeline is not None:
        final_result = detection_pipeline.take_latest()
        if final_result is not None:
            latest_detections = final_result.detections
            detection_frames += 1
            detection_wall_times.append(latest_detections.timing.wall_ms)
            queue_wait_times.append(final_result.queue_wait_ms)
            end_to_end_times.append(final_result.end_to_end_ms)
            preprocess_ms += latest_detections.timing.preprocess_ms
            inference_ms += latest_detections.timing.inference_ms
            postprocess_ms += latest_detections.timing.postprocess_ms

    if frame_count > 1 and first_receipt_ns is not None and latest_receipt_ns is not None:
        elapsed_s = (latest_receipt_ns - first_receipt_ns) / 1_000_000_000
        effective_fps = (frame_count - 1) / elapsed_s if elapsed_s else 0.0
    else:
        effective_fps = 0.0
    rate_name = (
        "synchronous loop throughput"
        if feature_extractor is not None or (detector is not None and detection_pipeline is None)
        else "host-receipt rate"
    )
    print(f"captured {frame_count} frames; {rate_name}={effective_fps:.2f} fps")
    if detection_frames:
        mean_wall_ms = sum(detection_wall_times) / detection_frames
        print(
            f"detected {detection_frames} frames; "
            f"wall mean={mean_wall_ms:.2f} ms "
            f"p50={_percentile(detection_wall_times, 0.50):.2f} ms "
            f"p95={_percentile(detection_wall_times, 0.95):.2f} ms"
        )
        print(
            "backend mean timings: "
            f"preprocess={preprocess_ms / detection_frames:.2f} ms "
            f"inference={inference_ms / detection_frames:.2f} ms "
            f"postprocess={postprocess_ms / detection_frames:.2f} ms"
        )
        if end_to_end_times:
            print(
                "pipeline latency: "
                f"queue mean={sum(queue_wait_times) / len(queue_wait_times):.2f} ms "
                f"end-to-end mean={sum(end_to_end_times) / len(end_to_end_times):.2f} ms "
                f"p95={_percentile(end_to_end_times, 0.95):.2f} ms"
            )
    if feature_frames:
        mean_feature_wall_ms = sum(feature_wall_times) / feature_frames
        print(
            f"extracted features from {feature_frames} frames; "
            f"wall mean={mean_feature_wall_ms:.2f} ms "
            f"p50={_percentile(feature_wall_times, 0.50):.2f} ms "
            f"p95={_percentile(feature_wall_times, 0.95):.2f} ms"
        )
        print(
            "feature backend mean timings: "
            f"preprocess={feature_preprocess_ms / feature_frames:.2f} ms "
            f"inference={feature_inference_ms / feature_frames:.2f} ms "
            f"postprocess={feature_postprocess_ms / feature_frames:.2f} ms"
        )
    if detection_pipeline is not None:
        stats = detection_pipeline.stats
        print(
            f"scheduler submitted={stats.submitted} processed={stats.processed} "
            f"stale-input-drops={stats.dropped_before_processing} "
            f"unconsumed-result-drops={stats.dropped_results}"
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "kuka-webcam-sim":
        try:
            from vision_pipeline.apps.kuka_webcam_teleop import (
                default_poc_root,
                run_kuka_webcam_teleop,
            )

            return run_kuka_webcam_teleop(
                poc_root=args.poc_root or default_poc_root(),
                camera=args.camera,
                model_name=args.model,
                device=args.device,
                headless=args.headless,
                max_seconds=args.max_seconds,
            )
        except (ValueError, OSError, RuntimeError, ModuleNotFoundError) as error:
            print(f"KUKA webcam simulation error: {error}", file=sys.stderr)
            return 2
    if args.command == "human-pose":
        from vision_pipeline.apps.human_pose_viewer import run_human_pose_viewer
        from vision_pipeline.perception.pose import HumanJoint
        from vision_pipeline.perception.pose.chains import LEFT_ARM, RIGHT_ARM, LimbChain

        try:
            chains = []
            for spec in args.chain:
                name, separator, joints = spec.partition(":")
                if not separator:
                    raise ValueError("chain must have NAME:JOINT,JOINT,... format")
                chain_joints = tuple(HumanJoint(item) for item in joints.split(","))
                chains.append(LimbChain(name, chain_joints))
            return run_human_pose_viewer(
                camera=args.camera,
                model=args.model,
                device=args.device,
                person_index=args.person_index,
                chains=tuple(chains) or (LEFT_ARM, RIGHT_ARM),
                max_frames=args.max_frames,
                headless=args.headless,
                output=args.output,
            )
        except (ValueError, OSError, ModuleNotFoundError) as error:
            print(f"human pose error: {error}", file=sys.stderr)
            return 2
    if args.command == "webcam":
        return _run_webcam(args)
    if args.command == "dino-webcam":
        return _run_dino_webcam(args)
    if args.command == "calibrate-stereo":
        return _run_stereo_calibration(args)
    if args.command == "fusion-sim":
        return _run_fusion_simulation(args)
    if args.command == "iphone-receiver":
        return _run_iphone_receiver(args)
    parser.error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
