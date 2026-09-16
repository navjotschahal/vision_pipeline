"""Phase 1A+1B producer: RealSense -> BoxEstimator seam -> print -> shared memory.

Runs the box-pose estimator against the live RealSense stream, prints one status line
per second (centre, yaw, extents, point count, capture timestamp, measured rate), and
publishes a shared-memory frame for `tabletop_viewer.py` on every iteration. The publish
is an unconditional, fixed-cost seqlock write (`runtime/shm_seqlock.py`): it never waits
on a reader, so a viewer attaching, falling behind, or not running at all cannot change
this loop's measured rate (design rule 4, and phase 1's gate).
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import torch

from vision_pipeline.apps.tabletop_config import (
    DEFAULT_CONFIG_PATH,
    TabletopBoxConfig,
    load_tabletop_box_config,
)
from vision_pipeline.perception.objects import ObjectObservation3D, TabletopBoxEstimator
from vision_pipeline.perception.objects.box_estimator import TabletopDiagnostics
from vision_pipeline.rgbd import AlignedRgbdFrame
from vision_pipeline.runtime.box_view_channel import (
    CATEGORY_ABOVE_PLANE,
    CATEGORY_CLUSTER,
    CATEGORY_OUTSIDE_WORKSPACE,
    CATEGORY_PLANE,
    BoxViewWriter,
    box_yaw_radians,
)
from vision_pipeline.sources.realsense import RealSenseError, RealSenseSource

_STATUS_INTERVAL_S = 1.0


@dataclass(frozen=True, slots=True)
class TabletopEstimatorAppConfig:
    config: TabletopBoxConfig
    publish: bool = True
    max_frames: int = 0


def _categories_from_diagnostics(diagnostics: TabletopDiagnostics) -> np.ndarray:
    point_count = int(diagnostics.full_cloud.xyz.shape[0])
    categories = np.full(point_count, CATEGORY_OUTSIDE_WORKSPACE, dtype=np.uint8)
    categories[diagnostics.workspace_mask.cpu().numpy()] = CATEGORY_PLANE
    if diagnostics.above_plane_mask is not None:
        categories[diagnostics.above_plane_mask.cpu().numpy()] = CATEGORY_ABOVE_PLANE
    if diagnostics.cluster_mask is not None:
        categories[diagnostics.cluster_mask.cpu().numpy()] = CATEGORY_CLUSTER
    return categories


def run_tabletop_estimator(app_config: TabletopEstimatorAppConfig) -> int:
    config = app_config.config
    source = RealSenseSource(config.realsense)
    estimator = TabletopBoxEstimator(config.estimator)
    writer = BoxViewWriter(config.channel_geometry) if app_config.publish else None

    frames_submitted = 0
    frames_failed = 0
    window_count = 0
    window_start = time.monotonic()
    measured_rate = 0.0

    try:
        source.open()
        info = source.info
        print(
            f"RealSense {info.device_name} serial={info.serial_number} "
            f"usb={info.usb_type_descriptor} fw={info.firmware_version}"
        )
        print(
            f"depth={info.actual_depth.mode.width}x{info.actual_depth.mode.height}"
            f"@{info.actual_depth.mode.fps} "
            f"color={info.actual_color.mode.width}x{info.actual_color.mode.height}"
            f"@{info.actual_color.mode.fps}"
        )
        if writer is not None:
            print(f"publishing box view on shared-memory channel {writer.name!r}")

        while app_config.max_frames <= 0 or frames_submitted < app_config.max_frames:
            frame = source.read()
            frames_submitted += 1
            observation = estimator.estimate(frame)
            if observation is None:
                frames_failed += 1

            diagnostics = estimator.last_diagnostics
            if writer is not None and diagnostics is not None:
                _publish(writer, frame, diagnostics, observation, frames_submitted, frames_failed)

            window_count += 1
            now = time.monotonic()
            elapsed = now - window_start
            if elapsed >= _STATUS_INTERVAL_S:
                measured_rate = window_count / elapsed
                window_count = 0
                window_start = now
                _print_status(observation, frames_submitted, frames_failed, measured_rate)
    except KeyboardInterrupt:
        print("\ninterrupted by operator")
    except RealSenseError as error:
        print(f"RealSense error: {error}", file=sys.stderr)
        return 2
    finally:
        if writer is not None:
            writer.close()
        if source.is_open:
            source.close()
        print(
            f"final: submitted={frames_submitted} failed={frames_failed} "
            f"success_rate={_safe_ratio(frames_submitted - frames_failed, frames_submitted):.1%}"
        )
    return 0


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _print_status(
    observation: ObjectObservation3D | None,
    frames_submitted: int,
    frames_failed: int,
    measured_rate: float,
) -> None:
    if observation is None:
        print(f"rate={measured_rate:5.1f} Hz  no box in view  submitted={frames_submitted}")
        return
    captured_at = observation.source.captured_at
    stamp = (
        datetime.fromtimestamp(captured_at.nanoseconds / 1e9, tz=UTC).isoformat()
        if captured_at is not None
        else "unknown"
    )
    x, y, z = observation.bounds.pose.position_metres
    sx, sy, sz = observation.bounds.size_metres
    yaw_degrees = np.degrees(box_yaw_radians(observation.bounds.pose.orientation_xyzw))
    print(
        f"rate={measured_rate:5.1f} Hz  "
        f"centre=({x:+.3f},{y:+.3f},{z:+.3f}) m  yaw={yaw_degrees:+6.1f} deg  "
        f"extents=({sx:.3f},{sy:.3f},{sz:.3f}) m  points={observation.point_count:5d}  "
        f"captured={stamp}  submitted={frames_submitted} failed={frames_failed}"
    )


def _publish(
    writer: BoxViewWriter,
    frame: AlignedRgbdFrame,
    diagnostics: TabletopDiagnostics,
    observation: ObjectObservation3D | None,
    frames_submitted: int,
    frames_failed: int,
) -> None:
    categories = _categories_from_diagnostics(diagnostics)
    xyz = diagnostics.full_cloud.xyz.to(dtype=torch.float32).cpu().numpy()
    colors = diagnostics.full_cloud.colors_rgb
    rgb = (
        colors.to(dtype=torch.uint8).cpu().numpy()
        if colors is not None
        else np.zeros((xyz.shape[0], 3), dtype=np.uint8)
    )
    captured_at = frame.depth.header.captured_at
    captured_at_ns = captured_at.nanoseconds if captured_at is not None else 0

    has_observation = observation is not None
    if observation is not None:
        box_position = observation.bounds.pose.position_metres
        box_orientation = observation.bounds.pose.orientation_xyzw
        box_size = observation.bounds.size_metres
        cluster_point_count = observation.point_count
    else:
        box_position = (0.0, 0.0, 0.0)
        box_orientation = (0.0, 0.0, 0.0, 1.0)
        box_size = (0.0, 0.0, 0.0)
        cluster_point_count = 0

    plane = diagnostics.segmentation.plane if diagnostics.segmentation is not None else None
    writer.publish(
        frame_index=frames_submitted,
        captured_at_ns=captured_at_ns,
        host_written_at_ns=time.time_ns(),
        color_bgr=frame.color.payload.data,
        points_xyz=xyz,
        points_rgb=rgb,
        points_category=categories,
        has_observation=has_observation,
        box_position_metres=box_position,
        box_orientation_xyzw=box_orientation,
        box_size_metres=box_size,
        cluster_point_count=cluster_point_count,
        plane_normal=plane.normal if plane is not None else (0.0, 0.0, 0.0),
        plane_offset_metres=plane.offset_metres if plane is not None else 0.0,
        estimator_frames_submitted=frames_submitted,
        estimator_frames_failed=frames_failed,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="tabletop_box.yaml path")
    parser.add_argument(
        "--no-publish",
        dest="publish",
        action="store_false",
        default=True,
        help="run the estimator without writing to the shared-memory channel",
    )
    parser.add_argument("--max-frames", type=int, default=0, help="zero means unlimited")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_tabletop_box_config(args.config)
    return run_tabletop_estimator(
        TabletopEstimatorAppConfig(config=config, publish=args.publish, max_frames=args.max_frames)
    )


if __name__ == "__main__":
    raise SystemExit(main())
