"""Validated YAML configuration with explicit CLI override semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """A configuration file is missing, malformed, or semantically invalid."""


_BACKENDS = {"auto", "avfoundation", "v4l2", "msmf"}
_DETECTOR_BACKENDS = {"torch", "ultralytics"}
_DETECTION_MODES = {"latest", "synchronous"}
_FEATURE_BACKENDS = {"dinov2-torch"}
_FEATURE_VISUALIZATIONS = {"none", "pca"}
_DINO_DEMO_MODES = {
    "depth",
    "point-cloud",
    "semantic",
    "retrieval",
    "dense-match",
    "sparse-match",
}


@dataclass(frozen=True, slots=True)
class WebcamAppConfig:
    device_index: int = 0
    backend: str = "auto"
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_id: str = "laptop_webcam_optical"
    source_id: str | None = None
    probe: bool = False
    probe_report: str | None = None
    max_frames: int = 0
    headless: bool = False
    mirror_preview: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.device_index, bool) or not isinstance(self.device_index, int):
            raise ConfigError("webcam.device_index must be an integer")
        if self.device_index < 0:
            raise ConfigError("webcam.device_index must be non-negative")
        if self.backend not in _BACKENDS:
            raise ConfigError(f"webcam.backend must be one of {sorted(_BACKENDS)}")
        for field_name, value in (("width", self.width), ("height", self.height)):
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, int):
                    raise ConfigError(f"webcam.{field_name} must be an integer or null")
                if value <= 0:
                    raise ConfigError(f"webcam.{field_name} must be positive")
        if self.fps is not None:
            if isinstance(self.fps, bool) or not isinstance(self.fps, int | float):
                raise ConfigError("webcam.fps must be a number or null")
            if not math.isfinite(self.fps) or self.fps <= 0:
                raise ConfigError("webcam.fps must be positive and finite")
        if not isinstance(self.frame_id, str) or not self.frame_id.strip():
            raise ConfigError("webcam.frame_id must be a non-empty string")
        if self.source_id is not None and (
            not isinstance(self.source_id, str) or not self.source_id.strip()
        ):
            raise ConfigError("webcam.source_id must be a non-empty string or null")
        if self.probe_report is not None and (
            not isinstance(self.probe_report, str) or not self.probe_report.strip()
        ):
            raise ConfigError("webcam.probe_report must be a non-empty path or null")
        for field_name in ("probe", "headless", "mirror_preview"):
            if not isinstance(getattr(self, field_name), bool):
                raise ConfigError(f"webcam.{field_name} must be true or false")
        if isinstance(self.max_frames, bool) or not isinstance(self.max_frames, int):
            raise ConfigError("webcam.max_frames must be an integer")
        if self.max_frames < 0:
            raise ConfigError("webcam.max_frames must be non-negative")
        if self.headless and self.max_frames == 0 and not (self.probe or self.probe_report):
            raise ConfigError("headless webcam capture requires max_frames > 0")


@dataclass(frozen=True, slots=True)
class YoloAppConfig:
    enabled: bool = False
    backend: str = "ultralytics"
    model: str = "yolo26n.pt"
    device: str = "cpu"
    confidence: float = 0.25
    iou: float = 0.7
    image_size: int = 640
    max_detections: int = 300

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigError("detector.enabled must be true or false")
        if self.backend not in _DETECTOR_BACKENDS:
            raise ConfigError(f"detector.backend must be one of {sorted(_DETECTOR_BACKENDS)}")
        for field_name in ("model", "device"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"detector.{field_name} must be a non-empty string")
        for field_name in ("confidence", "iou"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"detector.{field_name} must be a number")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ConfigError(f"detector.{field_name} must be between zero and one")
        for field_name in ("image_size", "max_detections"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"detector.{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class FeatureExtractorAppConfig:
    enabled: bool = False
    backend: str = "dinov2-torch"
    model: str = "dinov2_vits14_reg"
    device: str = "cpu"
    image_size: int = 518
    visualization: str = "pca"

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ConfigError("features.enabled must be true or false")
        if self.backend not in _FEATURE_BACKENDS:
            raise ConfigError(f"features.backend must be one of {sorted(_FEATURE_BACKENDS)}")
        for field_name in ("model", "device"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"features.{field_name} must be a non-empty string")
        if (
            isinstance(self.image_size, bool)
            or not isinstance(self.image_size, int)
            or self.image_size <= 0
        ):
            raise ConfigError("features.image_size must be a positive integer")
        if self.visualization not in _FEATURE_VISUALIZATIONS:
            raise ConfigError(
                f"features.visualization must be one of {sorted(_FEATURE_VISUALIZATIONS)}"
            )


@dataclass(frozen=True, slots=True)
class TrackingAppConfig:
    enabled: bool = False
    appearance_weight: float = 0.7
    minimum_cosine_similarity: float = 0.65
    minimum_iou: float = 0.01
    minimum_score: float = 0.45
    descriptor_momentum: float = 0.8
    max_missed_frames: int = 8
    class_aware: bool = True

    def __post_init__(self) -> None:
        for field_name in ("enabled", "class_aware"):
            if not isinstance(getattr(self, field_name), bool):
                raise ConfigError(f"tracking.{field_name} must be true or false")
        for field_name in (
            "appearance_weight",
            "minimum_cosine_similarity",
            "minimum_iou",
            "minimum_score",
            "descriptor_momentum",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"tracking.{field_name} must be a number")
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ConfigError(f"tracking.{field_name} must be between zero and one")
        if (
            isinstance(self.max_missed_frames, bool)
            or not isinstance(self.max_missed_frames, int)
            or self.max_missed_frames < 0
        ):
            raise ConfigError("tracking.max_missed_frames must be a non-negative integer")


@dataclass(frozen=True, slots=True)
class RuntimeAppConfig:
    detection_mode: str = "latest"

    def __post_init__(self) -> None:
        if self.detection_mode not in _DETECTION_MODES:
            raise ConfigError(f"runtime.detection_mode must be one of {sorted(_DETECTION_MODES)}")


@dataclass(frozen=True, slots=True)
class DinoDemoAppConfig:
    """Settings for the standalone, detector-free DINOv2 capability host."""

    mode: str = "dense-match"
    depth_model: str = "dinov2_vits14_ld"
    depth_weights: str = "NYU"
    gallery_capacity: int = 64
    window_width: int = 1280
    window_height: int = 720

    def __post_init__(self) -> None:
        if self.mode not in _DINO_DEMO_MODES:
            raise ConfigError(f"dino_demo.mode must be one of {sorted(_DINO_DEMO_MODES)}")
        if not isinstance(self.depth_model, str) or not self.depth_model.strip():
            raise ConfigError("dino_demo.depth_model must be a non-empty string")
        if self.depth_weights not in {"NYU", "KITTI"}:
            raise ConfigError("dino_demo.depth_weights must be NYU or KITTI")
        for field_name in ("gallery_capacity", "window_width", "window_height"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"dino_demo.{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class PointCloudAppConfig:
    """Depth deprojection and diagnostic 3D-view parameters."""

    horizontal_fov_degrees: float = 60.0
    fx: float | None = None
    fy: float | None = None
    cx: float | None = None
    cy: float | None = None
    sampling_stride: int = 4
    min_depth_metres: float = 0.1
    max_depth_metres: float = 10.0
    point_size: int = 2

    def __post_init__(self) -> None:
        if (
            isinstance(self.horizontal_fov_degrees, bool)
            or not isinstance(self.horizontal_fov_degrees, int | float)
            or not math.isfinite(self.horizontal_fov_degrees)
            or not 0 < self.horizontal_fov_degrees < 180
        ):
            raise ConfigError("point_cloud.horizontal_fov_degrees must be between 0 and 180")
        supplied_intrinsics = tuple(getattr(self, name) for name in ("fx", "fy", "cx", "cy"))
        if any(value is not None for value in supplied_intrinsics) and not all(
            value is not None for value in supplied_intrinsics
        ):
            raise ConfigError("point_cloud fx, fy, cx, and cy must be provided together")
        for field_name, value in zip(
            ("fx", "fy", "cx", "cy"),
            supplied_intrinsics,
            strict=True,
        ):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
            ):
                raise ConfigError(f"point_cloud.{field_name} must be finite or null")
        if self.fx is not None and self.fx <= 0:
            raise ConfigError("point_cloud.fx must be positive")
        if self.fy is not None and self.fy <= 0:
            raise ConfigError("point_cloud.fy must be positive")
        for field_name in ("sampling_stride", "point_size"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"point_cloud.{field_name} must be a positive integer")
        for field_name in ("min_depth_metres", "max_depth_metres"):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ConfigError(f"point_cloud.{field_name} must be non-negative and finite")
        if self.max_depth_metres <= self.min_depth_metres:
            raise ConfigError("point_cloud.max_depth_metres must exceed min_depth_metres")

    @property
    def has_calibrated_intrinsics(self) -> bool:
        return self.fx is not None


@dataclass(frozen=True, slots=True)
class FusionExperimentConfig:
    """Deterministic asynchronous IMU plus multi-camera simulation parameters."""

    duration_seconds: float = 6.0
    seed: int = 7
    imu_hz: int = 100
    camera_a_hz: int = 10
    camera_b_hz: int = 20
    imu_noise_std_metres_per_second2: float = 0.08
    acceleration_process_std_metres_per_second2: float = 0.15
    camera_a_position_std_metres: float = 0.03
    camera_b_position_std_metres: float = 0.05
    report_hz: int = 2

    def __post_init__(self) -> None:
        for field_name in ("duration_seconds",):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigError(f"fusion_experiment.{field_name} must be positive and finite")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ConfigError("fusion_experiment.seed must be an integer")
        for field_name in ("imu_hz", "camera_a_hz", "camera_b_hz", "report_hz"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"fusion_experiment.{field_name} must be a positive integer")
        for camera_rate in (self.camera_a_hz, self.camera_b_hz, self.report_hz):
            if self.imu_hz % camera_rate:
                raise ConfigError(
                    "fusion_experiment camera and report rates must divide imu_hz "
                    "for this deterministic exercise"
                )
        for field_name in (
            "imu_noise_std_metres_per_second2",
            "acceleration_process_std_metres_per_second2",
            "camera_a_position_std_metres",
            "camera_b_position_std_metres",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigError(f"fusion_experiment.{field_name} must be positive and finite")


@dataclass(frozen=True, slots=True)
class StereoCalibrationAppConfig:
    """Automatic paired-view ChArUco calibration parameters."""

    camera_a_device_index: int = 0
    camera_b_device_index: int = 1
    backend: str = "auto"
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    camera_a_frame_id: str = "laptop_webcam_optical"
    camera_b_frame_id: str = "iphone_camera_optical"
    squares_x: int = 7
    squares_y: int = 5
    square_length_metres: float = 0.04
    marker_length_metres: float = 0.03
    dictionary: str = "DICT_4X4_50"
    minimum_shared_corners: int = 12
    required_pairs: int = 20
    minimum_view_novelty: float = 0.08
    maximum_pair_skew_ms: float = 50.0
    output_path: str = "calibrations/webcam_iphone.yaml"
    board_image_path: str = "calibrations/charuco_board.png"
    board_image_width_pixels: int = 1400

    def __post_init__(self) -> None:
        for field_name in (
            "camera_a_device_index",
            "camera_b_device_index",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConfigError(f"stereo_calibration.{field_name} must be a non-negative integer")
        if self.camera_a_device_index == self.camera_b_device_index:
            raise ConfigError("stereo_calibration camera indices must differ")
        if self.backend not in _BACKENDS:
            raise ConfigError(f"stereo_calibration.backend must be one of {sorted(_BACKENDS)}")
        for field_name in (
            "width",
            "height",
            "squares_x",
            "squares_y",
            "minimum_shared_corners",
            "required_pairs",
            "board_image_width_pixels",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigError(f"stereo_calibration.{field_name} must be a positive integer")
        if self.squares_x < 3 or self.squares_y < 3:
            raise ConfigError("stereo_calibration board must have at least 3x3 squares")
        for field_name in (
            "fps",
            "square_length_metres",
            "marker_length_metres",
            "minimum_view_novelty",
            "maximum_pair_skew_ms",
        ):
            value = getattr(self, field_name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ConfigError(f"stereo_calibration.{field_name} must be positive and finite")
        if self.marker_length_metres >= self.square_length_metres:
            raise ConfigError("stereo_calibration marker length must be below square length")
        for field_name in (
            "camera_a_frame_id",
            "camera_b_frame_id",
            "dictionary",
            "output_path",
            "board_image_path",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConfigError(f"stereo_calibration.{field_name} must be non-empty")


@dataclass(frozen=True, slots=True)
class PipelineAppConfig:
    webcam: WebcamAppConfig = WebcamAppConfig()
    detector: YoloAppConfig = YoloAppConfig()
    features: FeatureExtractorAppConfig = FeatureExtractorAppConfig()
    tracking: TrackingAppConfig = TrackingAppConfig()
    runtime: RuntimeAppConfig = RuntimeAppConfig()
    dino_demo: DinoDemoAppConfig = DinoDemoAppConfig()
    point_cloud: PointCloudAppConfig = PointCloudAppConfig()
    fusion_experiment: FusionExperimentConfig = FusionExperimentConfig()
    stereo_calibration: StereoCalibrationAppConfig = StereoCalibrationAppConfig()


def _known_webcam_keys() -> set[str]:
    return {field.name for field in fields(WebcamAppConfig)}


def _known_detector_keys() -> set[str]:
    return {field.name for field in fields(YoloAppConfig)}


def _known_feature_keys() -> set[str]:
    return {field.name for field in fields(FeatureExtractorAppConfig)}


def _known_tracking_keys() -> set[str]:
    return {field.name for field in fields(TrackingAppConfig)}


def _known_runtime_keys() -> set[str]:
    return {field.name for field in fields(RuntimeAppConfig)}


def _known_dino_demo_keys() -> set[str]:
    return {field.name for field in fields(DinoDemoAppConfig)}


def _known_point_cloud_keys() -> set[str]:
    return {field.name for field in fields(PointCloudAppConfig)}


def _known_fusion_experiment_keys() -> set[str]:
    return {field.name for field in fields(FusionExperimentConfig)}


def _known_stereo_calibration_keys() -> set[str]:
    return {field.name for field in fields(StereoCalibrationAppConfig)}


def load_pipeline_config(path: str | Path) -> PipelineAppConfig:
    config_path = Path(path)
    try:
        with config_path.open(encoding="utf-8") as stream:
            document: Any = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigError(f"cannot read configuration {config_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {config_path}: {error}") from error

    if document is None:
        document = {}
    if not isinstance(document, Mapping):
        raise ConfigError("configuration root must be a mapping")
    unknown_sections = set(document) - {
        "webcam",
        "detector",
        "features",
        "tracking",
        "runtime",
        "dino_demo",
        "point_cloud",
        "fusion_experiment",
        "stereo_calibration",
    }
    if unknown_sections:
        raise ConfigError(f"unknown configuration sections: {sorted(unknown_sections)}")

    webcam = document.get("webcam", {})
    if not isinstance(webcam, Mapping):
        raise ConfigError("webcam configuration must be a mapping")
    unknown_keys = set(webcam) - _known_webcam_keys()
    if unknown_keys:
        raise ConfigError(f"unknown webcam configuration keys: {sorted(unknown_keys)}")
    detector = document.get("detector", {})
    if not isinstance(detector, Mapping):
        raise ConfigError("detector configuration must be a mapping")
    unknown_detector_keys = set(detector) - _known_detector_keys()
    if unknown_detector_keys:
        raise ConfigError(f"unknown detector configuration keys: {sorted(unknown_detector_keys)}")
    feature_extractor = document.get("features", {})
    if not isinstance(feature_extractor, Mapping):
        raise ConfigError("features configuration must be a mapping")
    unknown_feature_keys = set(feature_extractor) - _known_feature_keys()
    if unknown_feature_keys:
        raise ConfigError(f"unknown features configuration keys: {sorted(unknown_feature_keys)}")
    tracking = document.get("tracking", {})
    if not isinstance(tracking, Mapping):
        raise ConfigError("tracking configuration must be a mapping")
    unknown_tracking_keys = set(tracking) - _known_tracking_keys()
    if unknown_tracking_keys:
        raise ConfigError(f"unknown tracking configuration keys: {sorted(unknown_tracking_keys)}")
    runtime = document.get("runtime", {})
    if not isinstance(runtime, Mapping):
        raise ConfigError("runtime configuration must be a mapping")
    unknown_runtime_keys = set(runtime) - _known_runtime_keys()
    if unknown_runtime_keys:
        raise ConfigError(f"unknown runtime configuration keys: {sorted(unknown_runtime_keys)}")
    dino_demo = document.get("dino_demo", {})
    if not isinstance(dino_demo, Mapping):
        raise ConfigError("dino_demo configuration must be a mapping")
    unknown_dino_demo_keys = set(dino_demo) - _known_dino_demo_keys()
    if unknown_dino_demo_keys:
        raise ConfigError(f"unknown dino_demo configuration keys: {sorted(unknown_dino_demo_keys)}")
    point_cloud = document.get("point_cloud", {})
    if not isinstance(point_cloud, Mapping):
        raise ConfigError("point_cloud configuration must be a mapping")
    unknown_point_cloud_keys = set(point_cloud) - _known_point_cloud_keys()
    if unknown_point_cloud_keys:
        raise ConfigError(
            f"unknown point_cloud configuration keys: {sorted(unknown_point_cloud_keys)}"
        )
    fusion_experiment = document.get("fusion_experiment", {})
    if not isinstance(fusion_experiment, Mapping):
        raise ConfigError("fusion_experiment configuration must be a mapping")
    unknown_fusion_experiment_keys = set(fusion_experiment) - _known_fusion_experiment_keys()
    if unknown_fusion_experiment_keys:
        raise ConfigError(
            "unknown fusion_experiment configuration keys: "
            f"{sorted(unknown_fusion_experiment_keys)}"
        )
    stereo_calibration = document.get("stereo_calibration", {})
    if not isinstance(stereo_calibration, Mapping):
        raise ConfigError("stereo_calibration configuration must be a mapping")
    unknown_stereo_calibration_keys = set(stereo_calibration) - _known_stereo_calibration_keys()
    if unknown_stereo_calibration_keys:
        raise ConfigError(
            "unknown stereo_calibration configuration keys: "
            f"{sorted(unknown_stereo_calibration_keys)}"
        )
    try:
        return PipelineAppConfig(
            webcam=WebcamAppConfig(**dict(webcam)),
            detector=YoloAppConfig(**dict(detector)),
            features=FeatureExtractorAppConfig(**dict(feature_extractor)),
            tracking=TrackingAppConfig(**dict(tracking)),
            runtime=RuntimeAppConfig(**dict(runtime)),
            dino_demo=DinoDemoAppConfig(**dict(dino_demo)),
            point_cloud=PointCloudAppConfig(**dict(point_cloud)),
            fusion_experiment=FusionExperimentConfig(**dict(fusion_experiment)),
            stereo_calibration=StereoCalibrationAppConfig(**dict(stereo_calibration)),
        )
    except TypeError as error:
        raise ConfigError(f"invalid configuration fields: {error}") from error


def load_webcam_config(path: str | Path) -> WebcamAppConfig:
    """Load only the webcam section for callers that do not run perception."""

    return load_pipeline_config(path).webcam


def merge_webcam_overrides(
    base: WebcamAppConfig,
    overrides: Mapping[str, object],
) -> WebcamAppConfig:
    """Apply explicit CLI values on top of a validated base configuration."""

    values = {key: value for key, value in overrides.items() if value is not None}
    unknown_keys = set(values) - _known_webcam_keys()
    if unknown_keys:
        raise ConfigError(f"unknown webcam overrides: {sorted(unknown_keys)}")

    try:
        # Values come from a dynamic CLI mapping and are validated by __post_init__.
        return replace(base, **values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ConfigError(f"invalid webcam command-line override: {error}") from error


def merge_yolo_overrides(
    base: YoloAppConfig,
    overrides: Mapping[str, object],
) -> YoloAppConfig:
    """Apply explicit detector CLI values on top of the YAML configuration."""

    values = {key: value for key, value in overrides.items() if value is not None}
    unknown_keys = set(values) - _known_detector_keys()
    if unknown_keys:
        raise ConfigError(f"unknown detector overrides: {sorted(unknown_keys)}")
    try:
        return replace(base, **values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ConfigError(f"invalid detector command-line override: {error}") from error


def merge_feature_overrides(
    base: FeatureExtractorAppConfig,
    overrides: Mapping[str, object],
) -> FeatureExtractorAppConfig:
    """Apply explicit feature-extractor CLI values on top of YAML configuration."""

    values = {key: value for key, value in overrides.items() if value is not None}
    unknown_keys = set(values) - _known_feature_keys()
    if unknown_keys:
        raise ConfigError(f"unknown feature extractor overrides: {sorted(unknown_keys)}")
    try:
        return replace(base, **values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ConfigError(f"invalid feature command-line override: {error}") from error


def merge_tracking_overrides(
    base: TrackingAppConfig,
    overrides: Mapping[str, object],
) -> TrackingAppConfig:
    """Apply explicit tracker CLI values on top of YAML configuration."""

    values = {key: value for key, value in overrides.items() if value is not None}
    unknown_keys = set(values) - _known_tracking_keys()
    if unknown_keys:
        raise ConfigError(f"unknown tracking overrides: {sorted(unknown_keys)}")
    try:
        return replace(base, **values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ConfigError(f"invalid tracking command-line override: {error}") from error


def merge_runtime_overrides(
    base: RuntimeAppConfig,
    overrides: Mapping[str, object],
) -> RuntimeAppConfig:
    """Apply explicit runtime CLI values on top of the YAML configuration."""

    values = {key: value for key, value in overrides.items() if value is not None}
    unknown_keys = set(values) - _known_runtime_keys()
    if unknown_keys:
        raise ConfigError(f"unknown runtime overrides: {sorted(unknown_keys)}")
    try:
        return replace(base, **values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ConfigError(f"invalid runtime command-line override: {error}") from error


def merge_dino_demo_overrides(
    base: DinoDemoAppConfig,
    overrides: Mapping[str, object],
) -> DinoDemoAppConfig:
    """Apply explicit DINO demo CLI values on top of YAML configuration."""

    values = {key: value for key, value in overrides.items() if value is not None}
    unknown_keys = set(values) - _known_dino_demo_keys()
    if unknown_keys:
        raise ConfigError(f"unknown DINO demo overrides: {sorted(unknown_keys)}")
    try:
        return replace(base, **values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ConfigError(f"invalid DINO demo command-line override: {error}") from error


def merge_point_cloud_overrides(
    base: PointCloudAppConfig,
    overrides: Mapping[str, object],
) -> PointCloudAppConfig:
    """Apply explicit point-cloud CLI values on top of YAML configuration."""

    values = {key: value for key, value in overrides.items() if value is not None}
    unknown_keys = set(values) - _known_point_cloud_keys()
    if unknown_keys:
        raise ConfigError(f"unknown point-cloud overrides: {sorted(unknown_keys)}")
    try:
        return replace(base, **values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ConfigError(f"invalid point-cloud command-line override: {error}") from error


def merge_fusion_experiment_overrides(
    base: FusionExperimentConfig,
    overrides: Mapping[str, object],
) -> FusionExperimentConfig:
    """Apply explicit fusion-simulation CLI values over YAML parameters."""

    values = {key: value for key, value in overrides.items() if value is not None}
    unknown_keys = set(values) - _known_fusion_experiment_keys()
    if unknown_keys:
        raise ConfigError(f"unknown fusion experiment overrides: {sorted(unknown_keys)}")
    try:
        return replace(base, **values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ConfigError(f"invalid fusion experiment command-line override: {error}") from error


def merge_stereo_calibration_overrides(
    base: StereoCalibrationAppConfig,
    overrides: Mapping[str, object],
) -> StereoCalibrationAppConfig:
    """Apply explicit stereo-calibration CLI values over YAML parameters."""

    values = {key: value for key, value in overrides.items() if value is not None}
    unknown_keys = set(values) - _known_stereo_calibration_keys()
    if unknown_keys:
        raise ConfigError(f"unknown stereo calibration overrides: {sorted(unknown_keys)}")
    try:
        return replace(base, **values)  # type: ignore[arg-type]
    except TypeError as error:
        raise ConfigError(f"invalid stereo calibration command-line override: {error}") from error
