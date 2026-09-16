"""Click-select one object on the live RealSense stream and report its metric 3D geometry.

One command starts capture, the configured promptable segmenter, geometry, and the UI:

    python -m vision_pipeline.apps.select_object

Controls (color panel):

- left click: select an object (UNSELECTED or LOST), or add a positive prompt (TRACKING)
- right click, or shift + left click: add a negative prompt while TRACKING
- ``c``: clear the selection; the next left click reselects
- ``j``/``l``, ``i``/``k``, ``u``/``o``, ``+``/``-``, ``r``: rotate/zoom/reset the cloud view
- ``q`` or Esc: quit

The color panel always shows the exact frame whose aligned depth produced the displayed
mask and geometry. Positions are in the camera's color optical frame; nothing here is
robot-frame and nothing commands motion. ``--ndjson`` writes one dry-run observation
record per geometry output. ``--no-display`` with ``--click`` runs headless.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np
import torch

from vision_pipeline.apps.object_selection_config import (
    DEFAULT_CONFIG_PATH,
    build_segmenter,
    load_object_selection_config,
)
from vision_pipeline.apps.tabletop_viewer import _box_edge_points, _marker_points
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
from vision_pipeline.perception.objects.selected_geometry import SelectedGeometryEstimator
from vision_pipeline.perception.objects.selection import (
    Click,
    SelectionSession,
    SelectionState,
    map_click,
)
from vision_pipeline.runtime.latest_rgbd import LatestRgbdCapture
from vision_pipeline.runtime.selection_pipeline import (
    CommandKind,
    GeometryOutput,
    MaskOutput,
    OperatorCommand,
    SelectionPipeline,
)
from vision_pipeline.sources.realsense import RealSenseError, RealSenseSource
from vision_pipeline.sources.rgbd_recording import ReplayRgbdSource

WINDOW = "object selection"
_STATUS_INTERVAL_S = 1.0
_CLOUD_RENDER_INTERVAL_S = 1 / 15
_STATE_COLORS_BGR = {
    SelectionState.UNSELECTED: (160, 160, 160),
    SelectionState.TRACKING: (60, 200, 60),
    SelectionState.LOST: (40, 40, 230),
}
_STAGE_COLORS_RGB = np.array(
    [(150, 60, 60), (170, 90, 50), (150, 150, 60), (200, 120, 200), (60, 220, 90)],
    dtype=np.uint8,
)
_PLANE_INLIER_RGB = (70, 130, 180)
_PLANE_OUTLIER_RGB = (60, 60, 60)
_BOX_EDGE_RGB = (255, 110, 20)
_MARKER_RGB = (0, 230, 230)
_SCRATCH_FRAME = FrameId("select-object-view")
_SCRATCH_CLOCK = ClockDomain("select-object/view", ClockKind.HOST_MONOTONIC)
_BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)  # fmt: skip


def _time_record(point: TimePoint | None) -> dict[str, Any] | None:
    if point is None:
        return None
    return {"nanoseconds": point.nanoseconds, "clock": point.clock.domain_id}


def geometry_record(output: GeometryOutput) -> dict[str, Any]:
    """A JSON-serializable dry-run observation, with full timing and frame provenance."""

    mask = output.mask
    update = mask.update
    frame = mask.frame
    record: dict[str, Any] = {
        "schema": "selected-object-observation-v1",
        "frame_id": frame.depth.header.frame_id.value,
        "frameset_id": frame.frameset_id,
        "color_sample_id": frame.color.header.sample_id,
        "depth_sample_id": frame.depth.header.sample_id,
        "color_captured_at": _time_record(frame.color.header.captured_at),
        "depth_captured_at": _time_record(frame.depth.header.captured_at),
        "received_at": _time_record(frame.color.header.received_at),
        "state": update.state.value,
        "selection_id": update.selection_id,
        "loss_reason": update.loss_reason.value if update.loss_reason else None,
        "mask_backend": update.segmentation.backend if update.segmentation else None,
        "object_score": update.segmentation.object_score if update.segmentation else None,
        "capture_to_mask_ms": mask.capture_to_mask_ms,
        "capture_to_geometry_ms": output.capture_to_geometry_ms,
        "geometry_rejection": output.rejection.value if output.rejection else None,
    }
    estimate = output.geometry
    if estimate is not None:
        bounds = estimate.observation.bounds
        plane = estimate.support_plane
        record.update(
            {
                "centroid_metres": list(estimate.centroid_metres),
                "covariance_metres2": list(estimate.covariance_metres2),
                "bounds": {
                    "position_metres": list(bounds.pose.position_metres),
                    "orientation_xyzw": list(bounds.pose.orientation_xyzw),
                    "size_metres": list(bounds.size_metres),
                    "kind": "pca-observed-surface-bounds; not an object pose",
                },
                "point_count": estimate.observation.point_count,
                "support_plane": None
                if plane is None
                else {"normal": list(plane.normal), "offset_metres": plane.offset_metres},
                "quality": {
                    key: getattr(estimate.quality, key)
                    for key in estimate.quality.__dataclass_fields__
                },
                "method": estimate.method,
            }
        )
    return record


def _project(points: np.ndarray, intrinsics: PinholeIntrinsics) -> np.ndarray | None:
    if np.any(points[:, 2] <= 1e-3):
        return None
    u = intrinsics.fx * points[:, 0] / points[:, 2] + intrinsics.cx
    v = intrinsics.fy * points[:, 1] / points[:, 2] + intrinsics.cy
    return np.stack((u, v), axis=1)


def _box_corners(output: GeometryOutput) -> np.ndarray | None:
    if output.geometry is None:
        return None
    bounds = output.geometry.observation.bounds
    half = [size / 2 for size in bounds.size_metres]
    transform = RigidTransform3D(
        source_frame=_SCRATCH_FRAME,
        target_frame=_SCRATCH_FRAME,
        translation_metres=bounds.pose.position_metres,
        rotation_xyzw=bounds.pose.orientation_xyzw,
    )
    return np.array(
        [
            transform.apply_point((sx, sy, sz))
            for sx in (-half[0], half[0])
            for sy in (-half[1], half[1])
            for sz in (-half[2], half[2])
        ]
    )


class SelectionView:
    """Draws pipeline outputs and turns mouse events into operator commands."""

    def __init__(self, pipeline: SelectionPipeline, *, scale: float, cloud_size: int) -> None:
        self._pipeline = pipeline
        self._scale = scale
        self._renderer = PointCloudRenderer(
            PointCloudViewConfig(width=cloud_size, height=cloud_size, point_size=2)
        )
        self._cloud_size = cloud_size
        self._displayed: MaskOutput | None = None
        self._clicks: list[Click] = []
        self._message = "left click an object to select it"
        self._cloud_canvas = np.zeros((cloud_size, cloud_size, 3), np.uint8)
        self._cloud_rendered_at = 0.0
        self._cloud_source_index = -1

    def on_mouse(self, event: int, x: int, y: int, flags: int, _: object) -> None:
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        displayed = self._displayed
        if displayed is None:
            return
        payload = displayed.frame.color.payload
        width, height = round(payload.width * self._scale), round(payload.height * self._scale)
        negative = event == cv2.EVENT_RBUTTONDOWN or bool(flags & cv2.EVENT_FLAG_SHIFTKEY)
        try:
            click = map_click(
                x,
                y,
                viewport=(0, 0, width, height),
                image_size=(payload.width, payload.height),
                positive=not negative,
            )
        except ValueError:
            return
        state = displayed.update.state
        frameset_id = displayed.frame.frameset_id
        if state is SelectionState.TRACKING:
            self._pipeline.submit(OperatorCommand(CommandKind.REFINE, frameset_id, (click,)))
            self._clicks.append(click)
            self._message = f"{'negative' if negative else 'positive'} prompt added"
        elif negative:
            self._message = "negative clicks refine a TRACKING selection; left click to select"
        else:
            self._pipeline.submit(OperatorCommand(CommandKind.SELECT, frameset_id, (click,)))
            self._clicks = [click]
            self._message = "selecting"

    def clear(self) -> None:
        self._pipeline.submit(OperatorCommand(CommandKind.CLEAR))
        self._clicks = []
        self._message = "cleared; left click an object to select it"

    def handle_view_key(self, key: int) -> None:
        steps = {"j": (-5, 0, 0), "l": (5, 0, 0), "k": (0, -5, 0), "i": (0, 5, 0)}
        steps |= {"u": (0, 0, -5), "o": (0, 0, 5)}
        char = chr(key) if 0 <= key < 256 else ""
        if char in steps:
            yaw, pitch, roll = steps[char]
            self._renderer.rotate(yaw_degrees=yaw, pitch_degrees=pitch, roll_degrees=roll)
        elif char in "+=":
            self._renderer.change_zoom(1.15)
        elif char in "-_":
            self._renderer.change_zoom(1 / 1.15)
        elif char == "r":
            self._renderer.reset()
        else:
            return
        self._cloud_rendered_at = 0.0

    def render(self) -> np.ndarray | None:
        latest = self._pipeline.latest_mask
        if latest is None:
            return None
        geometry = self._pipeline.latest_geometry
        # Geometry trails the mask by about one output. Showing the frame the geometry was
        # computed on keeps the image, mask, bounds, and clicks on one RGB-D pair.
        if geometry is not None and geometry.mask.index >= latest.index - 2:
            mask = geometry.mask
        else:
            mask, geometry = latest, None
        self._displayed = mask
        if mask.update.state is SelectionState.TRACKING and self._message == "selecting":
            self._message = "left click: add positive  right click: add negative  c: clear"
        elif mask.update.state is SelectionState.LOST and self._message == "selecting":
            self._message = "selection failed; left click to select again"
        color = self._color_panel(mask, geometry)
        now = time.monotonic()
        geometry = self._pipeline.latest_geometry
        if geometry is not None and (
            now - self._cloud_rendered_at >= _CLOUD_RENDER_INTERVAL_S
            and geometry.mask.index != self._cloud_source_index
        ):
            self._cloud_canvas = self._cloud_panel(geometry)
            self._cloud_rendered_at = now
            self._cloud_source_index = geometry.mask.index
        height = max(color.shape[0], self._cloud_size)
        canvas = np.zeros((height, color.shape[1] + self._cloud_size, 3), np.uint8)
        canvas[: color.shape[0], : color.shape[1]] = color
        canvas[: self._cloud_size, color.shape[1] :] = self._cloud_canvas
        return canvas

    def _color_panel(self, mask: MaskOutput, geometry: GeometryOutput | None) -> np.ndarray:
        frame = mask.frame
        image = frame.color.payload.data.copy()
        update = mask.update
        if update.segmentation is not None:
            selected = update.segmentation.mask
            tint = (0, 170, 0) if update.is_tracking else (0, 0, 200)
            image[selected] = (0.5 * image[selected] + 0.5 * np.array(tint)).astype(np.uint8)
        if geometry is not None and geometry.geometry is not None:
            corners = _box_corners(geometry)
            projected = None if corners is None else _project(corners, frame.color_intrinsics)
            if projected is not None:
                points = np.rint(projected).astype(np.int32)
                for a, b in _BOX_EDGES:
                    cv2.line(image, tuple(points[a]), tuple(points[b]), (20, 110, 255), 2)
            centre = _project(np.array([geometry.geometry.centroid_metres]), frame.color_intrinsics)
            if centre is not None:
                cv2.drawMarker(
                    image,
                    tuple(np.rint(centre[0]).astype(np.int32)),
                    (230, 230, 0),
                    cv2.MARKER_CROSS,
                    18,
                    2,
                )
        if self._scale != 1.0:
            image = np.asarray(
                cv2.resize(
                    image, None, fx=self._scale, fy=self._scale, interpolation=cv2.INTER_LINEAR
                ),
                dtype=np.uint8,
            )
        if update.state is not SelectionState.UNSELECTED:
            for click in self._clicks:
                point = (round((click.x + 0.5) * self._scale), round((click.y + 0.5) * self._scale))
                marker = cv2.MARKER_CROSS if click.positive else cv2.MARKER_TILTED_CROSS
                tone = (60, 255, 60) if click.positive else (60, 60, 255)
                cv2.drawMarker(image, point, tone, marker, 16, 2)
        self._draw_status(image, mask, geometry)
        return image

    def _draw_status(
        self, image: np.ndarray, mask: MaskOutput, geometry: GeometryOutput | None
    ) -> None:
        update = mask.update
        state = update.state
        banner = state.value
        if update.loss_reason is not None:
            banner += f" ({update.loss_reason.value})"
        elif state is SelectionState.LOST:
            banner += " - left click to reselect"
        lines = [banner]
        if update.segmentation is not None:
            lines.append(
                f"{update.segmentation.backend} score={update.segmentation.object_score:.2f} "
                f"mask={update.segmentation.mask_pixels}px"
            )
        if mask.capture_to_mask_ms is not None:
            lines.append(f"capture->mask {mask.capture_to_mask_ms:.0f} ms")
        if geometry is not None and geometry.geometry is not None:
            estimate = geometry.geometry
            x, y, z = estimate.centroid_metres
            sx, sy, sz = estimate.observation.bounds.size_metres
            lines.append(f"centroid ({x:+.3f},{y:+.3f},{z:+.3f}) m  camera frame")
            lines.append(
                f"bounds {sx:.3f}x{sy:.3f}x{sz:.3f} m  points={estimate.quality.final_points} "
                f"valid={estimate.quality.valid_depth_fraction:.0%}"
            )
        elif geometry is not None and geometry.rejection is not None:
            lines.append(f"no 3D: {geometry.rejection.value}")
        for error in mask.command_errors:
            self._message = error
        lines.append(self._message)
        cv2.rectangle(image, (0, 0), (image.shape[1], 22 * len(lines) + 8), (0, 0, 0), -1)
        for row, text in enumerate(lines):
            tone = _STATE_COLORS_BGR[state] if row == 0 else (235, 235, 235)
            cv2.putText(
                image, text, (8, 22 * (row + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, tone, 1,
                cv2.LINE_AA,
            )  # fmt: skip

    def _cloud_panel(self, output: GeometryOutput) -> np.ndarray:
        diagnostics = output.diagnostics
        xyz_parts: list[np.ndarray] = []
        color_parts: list[np.ndarray] = []
        if diagnostics is not None and diagnostics.frameset_id == output.mask.frame.frameset_id:
            if diagnostics.background_xyz is not None and diagnostics.plane_inliers is not None:
                xyz_parts.append(diagnostics.background_xyz)
                tones = np.where(
                    diagnostics.plane_inliers[:, None],
                    np.array(_PLANE_INLIER_RGB, np.uint8),
                    np.array(_PLANE_OUTLIER_RGB, np.uint8),
                )
                color_parts.append(tones.astype(np.uint8))
            xyz_parts.append(diagnostics.stage_cloud)
            color_parts.append(_STAGE_COLORS_RGB[diagnostics.stage])
        estimate = output.geometry
        if estimate is not None:
            bounds = estimate.observation.bounds
            edges = _box_edge_points(
                bounds.pose.position_metres, bounds.pose.orientation_xyzw, bounds.size_metres
            ).astype(np.float32)
            marker = _marker_points(estimate.centroid_metres).astype(np.float32)
            xyz_parts += [edges, marker]
            color_parts += [
                np.tile(np.array(_BOX_EDGE_RGB, np.uint8), (len(edges), 1)),
                np.tile(np.array(_MARKER_RGB, np.uint8), (len(marker), 1)),
            ]
        if not xyz_parts:
            canvas = np.full((self._cloud_size, self._cloud_size, 3), 18, np.uint8)
            cv2.putText(
                canvas, "no selected geometry", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (200, 200, 200), 1, cv2.LINE_AA,
            )  # fmt: skip
            return canvas
        now = TimePoint(time.monotonic_ns(), _SCRATCH_CLOCK)
        header = SampleHeader(
            sample_id="select-object/view",
            source_id="select-object/view",
            sequence_number=0,
            measurement_kind=MeasurementKind.DEPTH_IMAGE,
            captured_at=now,
            received_at=now,
            frame_id=_SCRATCH_FRAME,
            calibration=None,
            producer="select_object",
            produced_on=ComputePlacement(ComputeKind.HOST_CPU, "select-object-view"),
        )
        cloud = PointCloud(
            source=header,
            xyz=torch.from_numpy(np.concatenate(xyz_parts).astype(np.float32)),
            colors_rgb=torch.from_numpy(np.concatenate(color_parts).astype(np.uint8)),
            # The points were deprojected with SDK-reported intrinsics; this view cloud
            # needs none, and the label keeps the renderer from calling it approximate.
            intrinsics=PinholeIntrinsics(
                1, 1, fx=1.0, fy=1.0, cx=0.0, cy=0.0, calibration_id="realsense-sdk"
            ),
            geometry_kind=GeometryKind.MEASURED_METRIC,
            sampling_stride=1,
            produced_on=ComputePlacement(ComputeKind.HOST_CPU, "select-object-view"),
            memory=MemoryPlacement(MemoryKind.HOST),
        )
        return self._renderer.render(
            cloud,
            title="SELECTED OBJECT",
            extra_status_lines=(
                "green final, pink isolated, yellow table",
                "clearance, brown workspace, red other",
                "surface, blue plane, orange bounds",
            ),
        )


class _StatusPrinter:
    """Once-per-second console line plus optional NDJSON records."""

    def __init__(self, ndjson: TextIO | None) -> None:
        self._ndjson = ndjson
        self._window: deque[float] = deque(maxlen=512)
        self._last_print = time.monotonic()
        self._geometry_count = 0

    def __call__(self, output: GeometryOutput) -> None:
        now = time.monotonic()
        if output.geometry is not None:
            self._geometry_count += 1
        if output.mask.capture_to_mask_ms is not None and output.mask.update.is_tracking:
            self._window.append(output.mask.capture_to_mask_ms)
        if self._ndjson is not None and output.mask.update.state is not SelectionState.UNSELECTED:
            self._ndjson.write(json.dumps(geometry_record(output)) + "\n")
        elapsed = now - self._last_print
        if elapsed < _STATUS_INTERVAL_S:
            return
        rate = self._geometry_count / elapsed
        self._geometry_count = 0
        self._last_print = now
        update = output.mask.update
        latency = (
            f"capture->mask p50={np.percentile(self._window, 50):.0f} "
            f"p95={np.percentile(self._window, 95):.0f} ms"
            if self._window
            else "capture->mask n/a"
        )
        self._window.clear()
        estimate = output.geometry
        if estimate is None:
            detail = f"no 3D ({output.rejection.value})" if output.rejection else "no 3D"
        else:
            x, y, z = estimate.centroid_metres
            sx, sy, sz = estimate.observation.bounds.size_metres
            detail = (
                f"centroid=({x:+.3f},{y:+.3f},{z:+.3f}) m size=({sx:.3f},{sy:.3f},{sz:.3f}) m "
                f"points={estimate.quality.final_points}"
            )
        print(
            f"{update.state.value:10s} geometry_rate={rate:5.1f} Hz  {latency}  {detail}",
            flush=True,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--backend", help="override selection.backend from the config")
    parser.add_argument("--replay", help="paced replay of a recording instead of the camera")
    parser.add_argument("--click", type=float, nargs=2, help="select this pixel on start")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--duration", type=float, default=0.0, help="seconds; 0 runs until quit")
    parser.add_argument("--ndjson", type=Path, help="append dry-run observation records here")
    parser.add_argument("--display-scale", type=float, default=1.0)
    parser.add_argument("--cloud-size", type=int, default=560)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.no_display and args.click is None:
        print("--no-display requires --click", file=sys.stderr)
        return 2
    if not math.isfinite(args.display_scale) or args.display_scale <= 0:
        print("--display-scale must be positive", file=sys.stderr)
        return 2
    config = load_object_selection_config(args.config)
    settings = config.segmenter
    if args.backend:
        settings = replace(settings, backend=args.backend)
    print(f"loading {settings.backend} on {settings.device} ...", flush=True)
    segmenter = build_segmenter(settings)
    source: RealSenseSource | ReplayRgbdSource
    live = args.replay is None
    source = (
        RealSenseSource(config.realsense) if live else ReplayRgbdSource(args.replay, pacing=True)
    )
    try:
        source.open()
    except RealSenseError as error:
        print(f"RealSense error: {error}", file=sys.stderr)
        return 2
    if isinstance(source, RealSenseSource):
        info = source.info
        print(
            f"RealSense {info.device_name} serial={info.serial_number} "
            f"usb={info.usb_type_descriptor} fw={info.firmware_version} "
            f"color={info.actual_color.mode.width}x{info.actual_color.mode.height}"
            f"@{info.actual_color.mode.fps}"
        )
    ndjson = args.ndjson.open("a") if args.ndjson else None
    printer = _StatusPrinter(ndjson)
    auto_click = Click(*args.click) if args.click else None

    def on_mask(output: MaskOutput) -> None:
        nonlocal auto_click
        if auto_click is not None and output.update.state is SelectionState.UNSELECTED:
            pipeline.submit(
                OperatorCommand(CommandKind.SELECT, output.frame.frameset_id, (auto_click,))
            )
            auto_click = None

    pipeline = SelectionPipeline(
        LatestRgbdCapture(source),
        SelectionSession(segmenter, config.policy),
        SelectedGeometryEstimator(config.geometry),
        live_timing=live,
        on_mask=on_mask,
        on_geometry=printer,
    )
    view = (
        None
        if args.no_display
        else SelectionView(pipeline, scale=args.display_scale, cloud_size=args.cloud_size)
    )
    deadline = time.monotonic() + args.duration if args.duration > 0 else math.inf
    try:
        pipeline.start()
        if view is not None:
            cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(WINDOW, view.on_mouse)
        last_index = -1
        while time.monotonic() < deadline and not pipeline.finished:
            if view is None:
                time.sleep(0.2)
                continue
            latest = pipeline.latest_mask
            if latest is not None and latest.index != last_index:
                canvas = view.render()
                if canvas is not None:
                    cv2.imshow(WINDOW, canvas)
                last_index = latest.index
            key = cv2.waitKey(5) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("c"):
                view.clear()
            elif key != 255:
                view.handle_view_key(key)
    except KeyboardInterrupt:
        print("\ninterrupted by operator")
    finally:
        pipeline.stop()
        source.close()
        if ndjson is not None:
            ndjson.close()
        if view is not None:
            cv2.destroyAllWindows()
    stats = pipeline.stats()
    print(
        f"final: frames_read={stats.capture_frames_read} processed={stats.frames_processed} "
        f"tracking={stats.tracking_outputs} lost_events={stats.lost_events} "
        f"capture_overwritten={stats.capture_frames_overwritten} "
        f"sequence_gaps={stats.capture_sequence_gaps} geometry={stats.geometry_computed} "
        f"rejected={stats.geometry_rejected} capture_to_mask_p50/p95="
        f"{stats.capture_to_mask_ms.p50}/{stats.capture_to_mask_ms.p95} ms "
        f"peak_cuda_allocated={torch.cuda.max_memory_allocated() / 2**20:.0f} MiB"
    )
    if pipeline.error is not None:
        print(f"pipeline error: {pipeline.error!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
