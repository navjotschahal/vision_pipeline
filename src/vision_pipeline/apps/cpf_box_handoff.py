"""Select a box, estimate it in CPF's world frame, and hand it to the bimanual grasp planner.

    python -m vision_pipeline.apps.cpf_box_handoff --measure        # crosshair for taping
    python -m vision_pipeline.apps.cpf_box_handoff                  # select + estimate + save

Pipeline: click-selected mask (EfficientTAM) -> metric cloud from the same frame's aligned
depth -> tape + IMU extrinsic into ``openarm_body_link0`` -> gravity-aligned box -> median
over a window -> ``object_pos`` and ``half_width`` for
``cpf/src/openarm_hw/tools/grasp_planner.py``.

The colour view draws the world box (magenta), the two hand attractors before any squeeze
(left green, right orange, at ``object_pos +/- (half_width + palm_offset)`` along world y),
and whether the estimate is ready. Press ``s`` to save the handoff JSON and print the
planner arguments; ``c`` clears the selection; ``q`` quits. Nothing here moves the robot.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from vision_pipeline.apps.object_selection_config import (
    DEFAULT_CONFIG_PATH,
    REPOSITORY_ROOT,
    build_segmenter,
    load_object_selection_config,
)
from vision_pipeline.apps.select_object import WINDOW, SelectionView, run_view_loop
from vision_pipeline.calibration.hand_eye import invert_transform
from vision_pipeline.calibration.manual_extrinsic import (
    GravitySample,
    ManualExtrinsic,
    build_manual_extrinsic,
    load_tape_measurement,
    sample_gravity,
)
from vision_pipeline.geometry.camera import PinholeIntrinsics
from vision_pipeline.perception.objects.box_grasp import (
    BoxEstimateWindow,
    BoxFitError,
    CpfGraspLimits,
    StableBox,
    WorldBoxEstimate,
    box_corners_world,
    cpf_handoff,
    fit_world_box,
)
from vision_pipeline.perception.objects.selected_geometry import SelectedGeometryEstimator
from vision_pipeline.perception.objects.selection import SelectionSession
from vision_pipeline.runtime.latest_rgbd import LatestRgbdCapture
from vision_pipeline.runtime.selection_pipeline import GeometryOutput, MaskOutput, SelectionPipeline
from vision_pipeline.sources.realsense import RealSenseError, RealSenseSource
from vision_pipeline.sources.rgbd_recording import ReplayRgbdSource

TAPE_CONFIG = REPOSITORY_ROOT / "configs" / "calibration" / "camera_world_tape.yaml"
HANDOFF_DIR = REPOSITORY_ROOT / "recordings" / "cpf_handoffs"
_EDGES = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _project(
    points_world: np.ndarray, camera_from_world: np.ndarray, intrinsics: PinholeIntrinsics
) -> np.ndarray | None:
    camera = points_world @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
    if np.any(camera[:, 2] <= 0.05):
        return None
    u = intrinsics.fx * camera[:, 0] / camera[:, 2] + intrinsics.cx
    v = intrinsics.fy * camera[:, 1] / camera[:, 2] + intrinsics.cy
    return np.asarray(np.rint(np.stack((u, v), axis=1)), dtype=np.int32)


class BoxHandoff:
    """Consumes geometry outputs, keeps the estimate window, and draws the world overlay."""

    def __init__(self, extrinsic: ManualExtrinsic, limits: CpfGraspLimits, window: int) -> None:
        self.extrinsic = extrinsic
        self.limits = limits
        self.window = BoxEstimateWindow(size=window)
        self.camera_from_world = invert_transform(extrinsic.world_from_camera)
        self.last_error: str | None = None
        self.record: dict[str, Any] | None = None

    def on_geometry(self, output: GeometryOutput) -> None:
        if not output.mask.update.is_tracking:
            self.window.clear()
            self.record = None
            return
        if output.geometry is None:
            self.last_error = output.rejection.value if output.rejection else "no geometry"
            return
        try:
            estimate = fit_world_box(output.geometry, self.extrinsic.world_from_camera)
        except BoxFitError as error:
            self.last_error = str(error)
            return
        self.last_error = None
        self.window.add(estimate)
        stable = self.window.summary()
        if stable is not None:
            self.record = cpf_handoff(stable, self.limits, extrinsic=self.extrinsic.summary())

    def draw(self, image: np.ndarray, mask: MaskOutput, geometry: GeometryOutput | None) -> None:
        record = self.record
        intrinsics = mask.frame.color_intrinsics
        lines: list[tuple[str, tuple[int, int, int]]] = []
        if record is not None and mask.update.is_tracking:
            stable = self.window.summary()
            assert stable is not None
            corners = box_corners_world(_as_estimate(stable))
            pixels = _project(corners, self.camera_from_world, intrinsics)
            if pixels is not None:
                for a, b in _EDGES:
                    cv2.line(
                        image, tuple(pixels[a]), tuple(pixels[b]), (255, 0, 255), 2, cv2.LINE_AA
                    )
            attractors = record["attractors_before_squeeze"]
            points = np.array([record["object_pos"], attractors["left"], attractors["right"]])
            marks = _project(points, self.camera_from_world, intrinsics)
            if marks is not None:
                cv2.drawMarker(image, tuple(marks[0]), (255, 255, 255), cv2.MARKER_CROSS, 16, 2)
                cv2.circle(image, tuple(marks[1]), 8, (60, 220, 60), 2)
                cv2.circle(image, tuple(marks[2]), 8, (0, 140, 255), 2)
            x, y, z = record["object_pos"]
            lines.append(
                (
                    f"world object_pos ({x:+.3f},{y:+.3f},{z:+.3f}) m  "
                    f"half_width {record['half_width']:.3f} m",
                    (255, 255, 255),
                )
            )
            lines.append(
                (
                    f"yaw {record['yaw_deg']:+.1f} deg  n={record['samples']}  "
                    f"std {max(record['centre_std_mm']):.1f} mm  table tilt "
                    + (
                        "n/a"
                        if record["support_tilt_deg"] is None
                        else f"{record['support_tilt_deg']:.1f} deg"
                    ),
                    (255, 255, 255),
                )
            )
            if record["ready"]:
                lines.append(("READY - press s to save for grasp_planner", (60, 220, 60)))
            else:
                lines.extend((problem[:90], (60, 60, 255)) for problem in record["problems"])
        elif self.last_error is not None:
            lines.append((f"no box: {self.last_error}"[:90], (60, 60, 255)))
        top = image.shape[0] - 22 * len(lines) - 6
        if lines:
            cv2.rectangle(image, (0, top - 4), (image.shape[1], image.shape[0]), (0, 0, 0), -1)
        for row, (text, colour) in enumerate(lines):
            cv2.putText(
                image,
                text,
                (8, top + 16 + 22 * row),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                colour,
                1,
                cv2.LINE_AA,
            )

    def save(self, directory: Path) -> Path | None:
        record = self.record
        if record is None:
            print("nothing to save: select the box and wait for the estimate", flush=True)
            return None
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / time.strftime("cpf_box_%Y%m%d_%H%M%S.json")
        path.write_text(
            json.dumps({**record, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2)
        )
        arguments = " ".join(record["grasp_planner_args"])
        status = "READY" if record["ready"] else "NOT READY: " + "; ".join(record["problems"])
        print(f"saved {path}\n  {status}\n  grasp_planner.py {arguments}", flush=True)
        return path


def _as_estimate(stable: StableBox) -> WorldBoxEstimate:
    """The window median shaped as a per-frame estimate, for drawing its corners."""

    return WorldBoxEstimate(
        frameset_id=stable.last_frameset_id,
        selection_id=stable.selection_id,
        depth_captured_at=None,
        centre_world=stable.centre_world,
        half_extents=stable.half_extents,
        yaw_deg=stable.yaw_deg,
        top_z=stable.centre_world[2] + stable.half_extents[2],
        bottom_z=stable.centre_world[2] - stable.half_extents[2],
        bottom_from_support_plane=stable.bottom_from_support_plane_fraction > 0.5,
        support_tilt_deg=stable.support_tilt_deg,
        point_count=0,
        top_points=0,
    )


def run_measure_mode(serial: str | None) -> int:
    """Show a crosshair on the optical axis for taping the heading aim point."""

    gravity = sample_gravity(serial)
    up = gravity.up_in_color()
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(-up[2])))))
    print(f"camera optical axis points {pitch:.1f} deg below horizontal (IMU)", flush=True)
    source = RealSenseSource(load_object_selection_config().realsense)
    source.open()
    try:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        while True:
            frame = source.read()
            image = frame.color.payload.data.copy()
            k = frame.color_intrinsics
            centre = (round(k.cx), round(k.cy))
            cv2.drawMarker(image, centre, (0, 255, 255), cv2.MARKER_CROSS, 60, 1)
            cv2.circle(image, centre, 6, (0, 255, 255), 1)
            cv2.putText(
                image,
                "tape the table point under the crosshair -> aim_point_world_metres; q quits",
                (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW, image)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                return 0
    finally:
        source.close()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--tape", default=str(TAPE_CONFIG), help="tape measurement YAML")
    parser.add_argument(
        "--measure", action="store_true", help="crosshair mode for taping the heading"
    )
    parser.add_argument(
        "--replay", help="paced replay instead of the camera (needs --gravity-json)"
    )
    parser.add_argument(
        "--gravity-json", help="use a saved accelerometer sample instead of sampling now"
    )
    parser.add_argument("--output-dir", default=str(HANDOFF_DIR))
    parser.add_argument("--window", type=int, default=30, help="frames in the median window")
    defaults = CpfGraspLimits()
    parser.add_argument("--palm-offset", type=float, default=defaults.palm_offset)
    parser.add_argument("--max-yaw-deg", type=float, default=defaults.max_yaw_deg)
    args = parser.parse_args(argv)
    config = load_object_selection_config(args.config)
    serial = config.realsense.serial_number
    if args.measure:
        return run_measure_mode(serial)

    tape = load_tape_measurement(args.tape)
    if args.gravity_json:
        saved = json.loads(Path(args.gravity_json).read_text())
        saved = saved.get("gravity", saved)
        mean, spread = saved["mean_acceleration_imu"], saved["standard_deviation_imu"]
        gravity = GravitySample(
            mean_acceleration_imu=(float(mean[0]), float(mean[1]), float(mean[2])),
            standard_deviation_imu=(float(spread[0]), float(spread[1]), float(spread[2])),
            samples=int(saved["samples"]),
            duration_s=float(saved["duration_s"]),
            color_from_imu_rotation=tuple(float(v) for v in saved["color_from_imu_rotation"]),
        )
    elif args.replay:
        parser.error("--replay needs --gravity-json (the recording has no IMU data)")
    else:
        print("sampling the accelerometer; keep the camera still...", flush=True)
        gravity = sample_gravity(serial)
    extrinsic = build_manual_extrinsic(tape, gravity)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "last_extrinsic.json").write_text(
        json.dumps({**extrinsic.summary(), "gravity": asdict(gravity)}, indent=2)
    )
    summary = extrinsic.summary()
    print(
        "camera in world: position "
        f"{np.round(summary['camera_position_world_metres'], 3).tolist()} m, heading "
        f"{summary['heading_deg']:.1f} deg, pitch down {summary['camera_pitch_down_deg']:.1f} deg, "
        f"image tilt {summary['camera_roll_deg']:.1f} deg",
        flush=True,
    )

    print(f"loading {config.segmenter.backend} ...", flush=True)
    segmenter = build_segmenter(config.segmenter)
    source: RealSenseSource | ReplayRgbdSource
    source = (
        RealSenseSource(config.realsense)
        if args.replay is None
        else ReplayRgbdSource(args.replay, pacing=True)
    )
    try:
        source.open()
    except RealSenseError as error:
        print(f"RealSense error: {error}", file=sys.stderr)
        return 2
    limits = replace(CpfGraspLimits(), palm_offset=args.palm_offset, max_yaw_deg=args.max_yaw_deg)
    handoff = BoxHandoff(extrinsic, limits, args.window)
    pipeline = SelectionPipeline(
        LatestRgbdCapture(source),
        SelectionSession(segmenter, config.policy),
        SelectedGeometryEstimator(config.geometry),
        live_timing=args.replay is None,
        on_geometry=handoff.on_geometry,
    )
    view = SelectionView(pipeline, scale=1.0, cloud_size=480, overlay=handoff.draw)

    def on_key(key: int) -> None:
        if key == ord("s"):
            handoff.save(output_dir)

    print("left click the box; press s to save when READY, q to quit", flush=True)
    try:
        pipeline.start()
        run_view_loop(pipeline, view, on_key=on_key)
    except KeyboardInterrupt:
        print("\ninterrupted by operator")
    finally:
        pipeline.stop()
        source.close()
        cv2.destroyAllWindows()
    if pipeline.error is not None:
        print(f"pipeline error: {pipeline.error!r}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
