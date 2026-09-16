"""Shared YAML configuration for the phase-1 tabletop estimator and viewer.

Both `tabletop_estimator.py` (the producer) and `tabletop_viewer.py` (the consumer)
load this same file, so the RealSense mode, the workspace bounds, and the point budget
that sizes the shared-memory channel never drift apart between the two processes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from vision_pipeline.contracts import FrameId
from vision_pipeline.perception.objects import TabletopBoxEstimatorConfig, WorkspaceBounds
from vision_pipeline.runtime.box_view_channel import BoxViewGeometry
from vision_pipeline.sources.realsense import RealSenseConfig

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "tabletop_box.yaml"
COLOR_FRAME_ID = FrameId("realsense_color_optical")


class TabletopConfigError(ValueError):
    """The YAML file is missing a required section or has an invalid value."""


@dataclass(frozen=True, slots=True)
class TabletopBoxConfig:
    realsense: RealSenseConfig
    estimator: TabletopBoxEstimatorConfig

    @property
    def channel_geometry(self) -> BoxViewGeometry:
        stride = self.estimator.sampling_stride
        return BoxViewGeometry(
            width=self.realsense.color_width,
            height=self.realsense.color_height,
            max_points=math.ceil(self.realsense.color_width / stride)
            * math.ceil(self.realsense.color_height / stride),
        )


def _section(raw: dict[str, Any], name: str) -> dict[str, Any]:
    value = raw.get(name)
    if not isinstance(value, dict):
        raise TabletopConfigError(f"config is missing a {name!r} mapping section")
    return value


def _vector3(section: dict[str, Any], key: str) -> tuple[float, float, float]:
    value = section.get(key)
    if not isinstance(value, list) or len(value) != 3:
        raise TabletopConfigError(f"workspace.{key} must be a 3-element list")
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError) as error:
        raise TabletopConfigError(f"workspace.{key} elements must be numbers") from error


def load_tabletop_box_config(path: str | Path = DEFAULT_CONFIG_PATH) -> TabletopBoxConfig:
    text = Path(path).read_text()
    raw = yaml.safe_load(text)
    if not isinstance(raw, dict):
        raise TabletopConfigError(f"{path}: top level must be a mapping")

    rs = _section(raw, "realsense")
    try:
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
    except (KeyError, TypeError, ValueError) as error:
        raise TabletopConfigError(f"{path}: invalid [realsense] section: {error}") from error

    ws = _section(raw, "workspace")
    workspace = WorkspaceBounds(
        reference_frame=COLOR_FRAME_ID,
        minimum_metres=_vector3(ws, "minimum_metres"),
        maximum_metres=_vector3(ws, "maximum_metres"),
    )

    est = _section(raw, "estimator")
    try:
        estimator = TabletopBoxEstimatorConfig(
            workspace=workspace,
            label=str(est.get("label", "box")),
            confidence=float(est.get("confidence", 1.0)),
            min_depth_metres=float(est["min_depth_metres"]),
            max_depth_metres=float(est["max_depth_metres"]),
            sampling_stride=int(est.get("sampling_stride", 2)),
            plane_distance_threshold_metres=float(est["plane_distance_threshold_metres"]),
            clearance_metres=float(est["clearance_metres"]),
            cell_size_metres=float(est["cell_size_metres"]),
            minimum_cluster_points=int(est["minimum_cluster_points"]),
            minimum_plane_inliers=int(est["minimum_plane_inliers"]),
            seed=int(est.get("seed", 0)),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise TabletopConfigError(f"{path}: invalid [estimator] section: {error}") from error

    return TabletopBoxConfig(realsense=realsense, estimator=estimator)


__all__ = [
    "COLOR_FRAME_ID",
    "DEFAULT_CONFIG_PATH",
    "TabletopBoxConfig",
    "TabletopConfigError",
    "load_tabletop_box_config",
]
