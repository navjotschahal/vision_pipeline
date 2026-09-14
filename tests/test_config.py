from pathlib import Path

import pytest

from vision_pipeline.config import (
    ConfigError,
    DinoDemoAppConfig,
    FeatureExtractorAppConfig,
    FusionExperimentConfig,
    PointCloudAppConfig,
    StereoCalibrationAppConfig,
    TrackingAppConfig,
    WebcamAppConfig,
    load_pipeline_config,
    load_webcam_config,
    merge_dino_demo_overrides,
    merge_feature_overrides,
    merge_fusion_experiment_overrides,
    merge_point_cloud_overrides,
    merge_tracking_overrides,
    merge_webcam_overrides,
)


def write_config(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_yaml_raw_camera_parameters_are_loaded(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "webcam.yaml",
        """
webcam:
  device_index: 1
  backend: avfoundation
  width: 1920
  height: 1080
  fps: 60
  source_id: sensors/continuity/rgb
  mirror_preview: true
""",
    )

    config = load_webcam_config(path)

    assert config.device_index == 1
    assert (config.width, config.height, config.fps) == (1920, 1080, 60)
    assert config.source_id == "sensors/continuity/rgb"
    assert config.mirror_preview


def test_cli_raw_parameters_override_yaml_values_independently() -> None:
    base = WebcamAppConfig(width=1280, height=720, fps=30.0)

    result = merge_webcam_overrides(
        base,
        {"width": 1920, "height": None, "fps": 60.0},
    )

    assert (result.width, result.height, result.fps) == (1920, 720, 60.0)


def test_probe_report_path_is_validated() -> None:
    with pytest.raises(ConfigError, match="probe_report"):
        WebcamAppConfig(probe_report="  ")


def test_detector_section_is_loaded_separately(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "pipeline.yaml",
        """
webcam:
  width: 1280
detector:
  enabled: true
  backend: torch
  model: custom.pt
  device: mps
  confidence: 0.4
  image_size: 512
runtime:
  detection_mode: synchronous
features:
  enabled: true
  backend: dinov2-torch
  model: dinov2_vits14_reg
  device: mps
  image_size: 518
  visualization: pca
tracking:
  enabled: true
  appearance_weight: 0.8
  minimum_cosine_similarity: 0.7
  max_missed_frames: 4
""",
    )

    config = load_pipeline_config(path)

    assert config.webcam.width == 1280
    assert config.detector.enabled
    assert config.detector.backend == "torch"
    assert config.detector.model == "custom.pt"
    assert config.detector.device == "mps"
    assert config.detector.confidence == 0.4
    assert config.detector.image_size == 512
    assert config.features.enabled
    assert config.features.model == "dinov2_vits14_reg"
    assert config.features.device == "mps"
    assert config.features.image_size == 518
    assert config.features.visualization == "pca"
    assert config.tracking.enabled
    assert config.tracking.appearance_weight == 0.8
    assert config.tracking.minimum_cosine_similarity == 0.7
    assert config.tracking.max_missed_frames == 4
    assert config.runtime.detection_mode == "synchronous"


def test_feature_cli_overrides_are_independent() -> None:
    base = FeatureExtractorAppConfig(device="cpu", image_size=518)

    result = merge_feature_overrides(
        base,
        {"enabled": True, "device": "mps", "image_size": 392, "model": None},
    )

    assert result.enabled
    assert result.device == "mps"
    assert result.image_size == 392
    assert result.model == base.model


def test_tracking_cli_overrides_are_independent() -> None:
    base = TrackingAppConfig(appearance_weight=0.7, max_missed_frames=8)

    result = merge_tracking_overrides(
        base,
        {"enabled": True, "appearance_weight": 0.9, "max_missed_frames": None},
    )

    assert result.enabled
    assert result.appearance_weight == 0.9
    assert result.max_missed_frames == 8


def test_dino_demo_config_and_cli_override_are_independent(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "dino.yaml",
        """
dino_demo:
  mode: semantic
  depth_weights: KITTI
  gallery_capacity: 12
  window_width: 1600
  window_height: 900
""",
    )

    loaded = load_pipeline_config(path).dino_demo
    result = merge_dino_demo_overrides(loaded, {"mode": "sparse-match", "depth_weights": None})

    assert loaded == DinoDemoAppConfig(
        mode="semantic",
        depth_weights="KITTI",
        gallery_capacity=12,
        window_width=1600,
        window_height=900,
    )
    assert result.mode == "sparse-match"
    assert result.depth_weights == "KITTI"


def test_point_cloud_config_requires_complete_calibrated_intrinsics() -> None:
    with pytest.raises(ConfigError, match="provided together"):
        PointCloudAppConfig(fx=800, fy=800)

    base = PointCloudAppConfig(horizontal_fov_degrees=60, sampling_stride=4)
    result = merge_point_cloud_overrides(
        base,
        {"horizontal_fov_degrees": 72.0, "sampling_stride": 6, "point_size": None},
    )

    assert result.horizontal_fov_degrees == 72
    assert result.sampling_stride == 6
    assert result.point_size == base.point_size


def test_fusion_and_stereo_calibration_sections_are_validated(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "sensors.yaml",
        """
fusion_experiment:
  duration_seconds: 4
  imu_hz: 100
  camera_a_hz: 10
  camera_b_hz: 20
stereo_calibration:
  camera_a_device_index: 0
  camera_b_device_index: 1
  width: 1280
  height: 720
  required_pairs: 24
""",
    )

    config = load_pipeline_config(path)

    assert config.fusion_experiment.duration_seconds == 4
    assert config.stereo_calibration == StereoCalibrationAppConfig(required_pairs=24)
    assert merge_fusion_experiment_overrides(
        config.fusion_experiment, {"seed": 99, "imu_hz": None}
    ) == FusionExperimentConfig(duration_seconds=4, seed=99)


def test_fusion_camera_rates_must_align_to_imu_ticks() -> None:
    with pytest.raises(ConfigError, match="must divide"):
        FusionExperimentConfig(imu_hz=100, camera_a_hz=30)


def test_unknown_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = write_config(
        tmp_path / "webcam.yaml",
        """
webcam:
  device_index: 0
  guessed_magic_fps: 240
""",
    )

    with pytest.raises(ConfigError, match="unknown webcam configuration keys"):
        load_webcam_config(path)


def test_unbounded_headless_capture_is_rejected() -> None:
    with pytest.raises(ConfigError, match="requires max_frames"):
        WebcamAppConfig(headless=True, max_frames=0)
