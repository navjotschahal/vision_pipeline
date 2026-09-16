"""Phase 1B consumer: reads the shared-memory box view and renders it with cv2.

Runs as a separate process from `tabletop_estimator.py` so the estimator's measured
rate never depends on how fast this window can render (design rule 4, and phase 1's
gate). Every tick reads whatever the seqlock currently holds -- always the producer's
latest frame, never a queue -- and only redraws when the frame actually changed, so a
slow renderer just re-shows a frame's worth of staleness rather than falling behind.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from vision_pipeline.apps.tabletop_config import DEFAULT_CONFIG_PATH, load_tabletop_box_config
from vision_pipeline.contracts import (
    ClockDomain,
    ClockKind,
    ComputeKind,
    ComputePlacement,
    FrameId,
    MeasurementKind,
    MemoryKind,
    MemoryPlacement,
    SampleHeader,
    TimePoint,
)
from vision_pipeline.geometry.camera import PinholeIntrinsics
from vision_pipeline.geometry.pointcloud import GeometryKind, PointCloud
from vision_pipeline.geometry.spatial import RigidTransform3D
from vision_pipeline.geometry.visualization import PointCloudRenderer, PointCloudViewConfig
from vision_pipeline.runtime.box_view_channel import (
    CATEGORY_ABOVE_PLANE,
    CATEGORY_CLUSTER,
    CATEGORY_OUTSIDE_WORKSPACE,
    CATEGORY_PLANE,
    CHANNEL_NAME,
    BoxViewFrame,
    BoxViewGeometry,
    BoxViewReader,
    box_yaw_radians,
)
from vision_pipeline.runtime.shm_seqlock import SeqlockTornReadError

_CATEGORY_COLORS_RGB = {
    CATEGORY_OUTSIDE_WORKSPACE: (55, 55, 55),
    CATEGORY_PLANE: (70, 130, 180),
    CATEGORY_ABOVE_PLANE: (225, 195, 40),
    CATEGORY_CLUSTER: (60, 220, 90),
}
_BOX_EDGE_COLOR_RGB = (255, 110, 20)
_MARKER_COLOR_RGB = (0, 230, 230)
_EDGE_SAMPLES = 24
_MARKER_ARM_METRES = 0.03
_MARKER_SAMPLES = 9
_STATUS_INTERVAL_S = 1.0
_ROTATE_KEYS = {ord("j"), ord("l"), ord("i"), ord("k"), ord("u"), ord("o")}
_ZOOM_KEYS = {ord("+"), ord("="), ord("-"), ord("_")}

_SCRATCH_FRAME = FrameId("tabletop-viewer-scratch")
_SCRATCH_CLOCK = ClockDomain("tabletop-viewer/local", ClockKind.HOST_MONOTONIC)

_BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


@dataclass(frozen=True, slots=True)
class TabletopViewerAppConfig:
    geometry: BoxViewGeometry
    channel_name: str = CHANNEL_NAME
    window_width: int = 960
    window_height: int = 720
    point_size: int = 2


def _box_edge_points(
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
    size: tuple[float, float, float],
) -> np.ndarray:
    half = tuple(value / 2 for value in size)
    corners_local = [
        (sx, sy, sz)
        for sx in (-half[0], half[0])
        for sy in (-half[1], half[1])
        for sz in (-half[2], half[2])
    ]
    transform = RigidTransform3D(
        source_frame=_SCRATCH_FRAME,
        target_frame=_SCRATCH_FRAME,
        translation_metres=position,
        rotation_xyzw=orientation,
    )
    corners_world = np.array([transform.apply_point(corner) for corner in corners_local])
    segments = []
    steps = np.linspace(0.0, 1.0, _EDGE_SAMPLES)[:, None]
    for a, b in _BOX_EDGES:
        segments.append(corners_world[a] * (1 - steps) + corners_world[b] * steps)
    return np.concatenate(segments, axis=0)


def _marker_points(position: tuple[float, float, float]) -> np.ndarray:
    offsets = np.linspace(-_MARKER_ARM_METRES, _MARKER_ARM_METRES, _MARKER_SAMPLES)
    center = np.array(position)
    arms = []
    for axis in range(3):
        arm = np.tile(center, (_MARKER_SAMPLES, 1))
        arm[:, axis] += offsets
        arms.append(arm)
    return np.concatenate(arms, axis=0)


def _scratch_header() -> SampleHeader:
    now = TimePoint(time.monotonic_ns(), _SCRATCH_CLOCK)
    return SampleHeader(
        sample_id="tabletop-viewer/scratch",
        source_id="tabletop-viewer/scratch",
        sequence_number=0,
        measurement_kind=MeasurementKind.DEPTH_IMAGE,
        captured_at=now,
        received_at=now,
        frame_id=_SCRATCH_FRAME,
        calibration=None,
        producer="tabletop_viewer",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "tabletop-viewer"),
    )


def _build_view_cloud(frame: BoxViewFrame) -> PointCloud[torch.Tensor]:
    category_colors = np.zeros((len(_CATEGORY_COLORS_RGB), 3), dtype=np.uint8)
    for category, color in _CATEGORY_COLORS_RGB.items():
        category_colors[category] = color
    scene_colors = category_colors[frame.points_category]

    xyz_parts = [frame.points_xyz.astype(np.float32)]
    color_parts = [scene_colors]
    if frame.has_observation:
        edges = _box_edge_points(
            frame.box_position_metres, frame.box_orientation_xyzw, frame.box_size_metres
        ).astype(np.float32)
        marker = _marker_points(frame.box_position_metres).astype(np.float32)
        edge_color = np.array(_BOX_EDGE_COLOR_RGB, dtype=np.uint8)
        marker_color = np.array(_MARKER_COLOR_RGB, dtype=np.uint8)
        xyz_parts.append(edges)
        color_parts.append(np.tile(edge_color, (edges.shape[0], 1)))
        xyz_parts.append(marker)
        color_parts.append(np.tile(marker_color, (marker.shape[0], 1)))

    xyz = np.concatenate(xyz_parts, axis=0)
    colors = np.concatenate(color_parts, axis=0)
    return PointCloud(
        source=_scratch_header(),
        xyz=torch.from_numpy(xyz),
        colors_rgb=torch.from_numpy(colors),
        intrinsics=PinholeIntrinsics(1, 1, fx=1.0, fy=1.0, cx=0.0, cy=0.0),
        geometry_kind=GeometryKind.MEASURED_METRIC,
        sampling_stride=1,
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "tabletop-viewer"),
        memory=MemoryPlacement(MemoryKind.HOST),
    )


def _wait_for_channel(
    geometry: BoxViewGeometry, channel_name: str, *, timeout_s: float = 30.0
) -> BoxViewReader:
    """Attach once the estimator has created the channel, whichever process started first."""

    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return BoxViewReader(geometry, name=channel_name)
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise
            print(f"waiting for {channel_name!r} -- start tabletop_estimator.py first")
            time.sleep(1.0)


def run_tabletop_viewer(config: TabletopViewerAppConfig) -> int:
    reader = _wait_for_channel(config.geometry, config.channel_name)
    renderer = PointCloudRenderer(
        PointCloudViewConfig(
            width=config.window_width, height=config.window_height, point_size=config.point_size
        )
    )

    last_frame_index: int | None = None
    producer_dropped_total = 0
    torn_read_count = 0
    render_count = 0
    window_start = time.monotonic()
    render_rate = 0.0

    try:
        print(f"attached to shared-memory channel {config.channel_name!r}; waiting for frames...")
        while True:
            key = cv2.waitKey(15) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in _ROTATE_KEYS or key == ord("r"):
                renderer.rotate(
                    yaw_degrees=5.0 if key == ord("l") else -5.0 if key == ord("j") else 0.0,
                    pitch_degrees=5.0 if key == ord("i") else -5.0 if key == ord("k") else 0.0,
                    roll_degrees=5.0 if key == ord("o") else -5.0 if key == ord("u") else 0.0,
                )
                if key == ord("r"):
                    renderer.reset()
            elif key in _ZOOM_KEYS:
                renderer.change_zoom(1.15 if key in (ord("+"), ord("=")) else 1 / 1.15)

            try:
                frame = reader.read()
            except SeqlockTornReadError:
                # A producer publishing every ~33 ms occasionally wins the race against
                # this tick's retry budget. There is always a next tick, so this is a
                # skip, not a failure -- crashing the viewer over a single missed frame
                # would violate the same "never stall on the other side" rule this
                # channel exists to satisfy.
                torn_read_count += 1
                continue
            if frame is None:
                continue
            if last_frame_index is not None and frame.frame_index > last_frame_index + 1:
                producer_dropped_total += frame.frame_index - last_frame_index - 1
            if frame.frame_index == last_frame_index:
                continue
            last_frame_index = frame.frame_index

            cloud = _build_view_cloud(frame)
            age_ms = (time.time_ns() - frame.captured_at_ns) / 1e6
            yaw_degrees = np.degrees(box_yaw_radians(frame.box_orientation_xyzw))
            status = (
                f"box: has_object={frame.has_observation} "
                f"cluster_pts={frame.cluster_point_count} yaw={yaw_degrees:+.1f}deg "
                f"age={age_ms:.0f}ms",
                f"producer: submitted={frame.estimator_frames_submitted} "
                f"failed={frame.estimator_frames_failed} "
                f"never_shown={producer_dropped_total}",
                f"viewer: render_rate={render_rate:.1f} Hz",
            )
            canvas = renderer.render(
                cloud, title="REALSENSE TABLETOP BOX", extra_status_lines=status
            )
            cv2.imshow("tabletop point cloud", canvas)
            cv2.imshow("tabletop color", frame.color_bgr)

            render_count += 1
            now = time.monotonic()
            elapsed = now - window_start
            if elapsed >= _STATUS_INTERVAL_S:
                render_rate = render_count / elapsed
                render_count = 0
                window_start = now
                print(
                    f"viewer render_rate={render_rate:5.1f} Hz  "
                    f"producer_frames_never_shown={producer_dropped_total}  "
                    f"torn_reads={torn_read_count}  "
                    f"has_object={frame.has_observation}"
                )
    except KeyboardInterrupt:
        print("\ninterrupted by operator")
    finally:
        reader.close()
        cv2.destroyAllWindows()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="tabletop_box.yaml path")
    parser.add_argument("--channel", default=CHANNEL_NAME, help="shared-memory channel name")
    parser.add_argument("--window-width", type=int, default=960)
    parser.add_argument("--window-height", type=int, default=720)
    parser.add_argument("--point-size", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    box_config = load_tabletop_box_config(args.config)
    return run_tabletop_viewer(
        TabletopViewerAppConfig(
            geometry=box_config.channel_geometry,
            channel_name=args.channel,
            window_width=args.window_width,
            window_height=args.window_height,
            point_size=args.point_size,
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
