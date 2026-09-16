"""YAML configuration for the click-selection app and its benchmark.

Loading this file never imports a research backend; only the segmenter factory does.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vision_pipeline.apps.tabletop_config import COLOR_FRAME_ID
from vision_pipeline.perception.objects import WorkspaceBounds
from vision_pipeline.perception.objects.selected_geometry import SelectedGeometryConfig
from vision_pipeline.perception.objects.selection import SelectionPolicy
from vision_pipeline.sources.realsense import RealSenseConfig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPOSITORY_ROOT / "configs" / "object_selection.yaml"


class ObjectSelectionConfigError(ValueError):
    """The YAML file is missing a required section or has an invalid value."""


@dataclass(frozen=True, slots=True)
class SegmenterSettings:
    backend: str
    models_root: Path
    device: str
    memory_window_frames: int
    max_conditioning_frames: int
    recent_memory_frames: int | None


@dataclass(frozen=True, slots=True)
class ObjectSelectionConfig:
    realsense: RealSenseConfig
    segmenter: SegmenterSettings
    policy: SelectionPolicy
    geometry: SelectedGeometryConfig


def _section(raw: dict[str, Any], name: str, path: str | Path) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise ObjectSelectionConfigError(f"{path}: missing a {name!r} mapping section")
    return value


def _vector3(values: object, name: str) -> tuple[float, float, float]:
    if not isinstance(values, list) or len(values) != 3:
        raise ObjectSelectionConfigError(f"{name} must be a 3-element list")
    return (float(values[0]), float(values[1]), float(values[2]))


def load_object_selection_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ObjectSelectionConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ObjectSelectionConfigError(f"{path}: top level must be a mapping")
    try:
        rs = _section(raw, "realsense", path)
        realsense = RealSenseConfig(
            serial_number=str(rs["serial_number"]) if rs.get("serial_number") else None,
            color_width=int(rs["color_width"]),
            color_height=int(rs["color_height"]),
            color_fps=int(rs["color_fps"]),
            depth_width=int(rs["depth_width"]),
            depth_height=int(rs["depth_height"]),
            depth_fps=int(rs["depth_fps"]),
            color_frame_id=COLOR_FRAME_ID,
        )
        selection = _section(raw, "selection", path)
        models_root = Path(str(selection["models_root"])).expanduser()
        if not models_root.is_absolute():
            models_root = REPOSITORY_ROOT / models_root
        segmenter = SegmenterSettings(
            backend=str(selection["backend"]),
            models_root=models_root,
            device=str(selection.get("device", "cuda:0")),
            memory_window_frames=int(selection.get("memory_window_frames", 16)),
            max_conditioning_frames=int(selection.get("max_conditioning_frames", 4)),
            recent_memory_frames=(
                None
                if selection.get("recent_memory_frames") is None
                else int(selection["recent_memory_frames"])
            ),
        )
        policy_raw = selection.get("policy", {})
        ratio = policy_raw.get("max_area_ratio", 4.0)
        policy = SelectionPolicy(
            minimum_object_score=float(policy_raw.get("minimum_object_score", 0.5)),
            minimum_mask_pixels=int(policy_raw.get("minimum_mask_pixels", 50)),
            max_area_ratio=None if ratio is None else float(ratio),
        )
        geometry_raw = dict(_section(raw, "geometry", path))
        workspace_raw = geometry_raw.pop("workspace", None)
        workspace = (
            None
            if workspace_raw is None
            else WorkspaceBounds(
                reference_frame=COLOR_FRAME_ID,
                minimum_metres=_vector3(workspace_raw.get("minimum_metres"), "minimum_metres"),
                maximum_metres=_vector3(workspace_raw.get("maximum_metres"), "maximum_metres"),
            )
        )
        geometry = SelectedGeometryConfig(workspace=workspace, **geometry_raw)
    except (KeyError, TypeError, ValueError) as error:
        if isinstance(error, ObjectSelectionConfigError):
            raise
        raise ObjectSelectionConfigError(f"{path}: {error}") from error
    return ObjectSelectionConfig(realsense, segmenter, policy, geometry)


def build_segmenter(settings: SegmenterSettings, *, compile_image_encoder: bool = False) -> Any:
    """Construct the configured research backend, importing it only now."""

    from vision_pipeline.perception.objects.selection_backend import build_research_segmenter

    return build_research_segmenter(
        settings.backend,
        models_root=settings.models_root,
        device=settings.device,
        memory_window_frames=settings.memory_window_frames,
        max_conditioning_frames=settings.max_conditioning_frames,
        recent_memory_frames=settings.recent_memory_frames,
        compile_image_encoder=compile_image_encoder,
    )


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "REPOSITORY_ROOT",
    "ObjectSelectionConfig",
    "ObjectSelectionConfigError",
    "SegmenterSettings",
    "build_segmenter",
    "load_object_selection_config",
]
