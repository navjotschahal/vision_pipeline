"""Camera-to-robot calibration without a marker on the robot: depth registered to the model.

    python -m vision_pipeline.apps.openarm_model_calibration capture --arms right
    python -m vision_pipeline.apps.openarm_model_calibration capture --static --auto 3
    python -m vision_pipeline.apps.openarm_model_calibration solve --session <dir>
    python -m vision_pipeline.apps.openarm_model_calibration residuals --session <dir>
    python -m vision_pipeline.apps.openarm_hand_eye check --arm right          # same overlay

The marker plate lies flat on the table, which is the robot's own reference plane
(``openarm_body_link0`` has z = 0 at the table top), so its detected corners pin roll,
pitch and height. Depth points of the torso, the shoulder mounts and every arm whose
driver publishes joint state are registered to the MuJoCo visual meshes posed with that
joint state (robust point-to-plane ICP), starting from the tape estimate or a previous
calibration. Several still arm poses, hand-guided in ``--mode damp``, add the surface
variety that fixes x, y and yaw. ``--static`` captures with no driver at all (torso and
plate only), which is enough for a first number and for a drift check. Nothing here
commands motion; the shared-memory command block is never written.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.apps.object_selection_config import load_object_selection_config
from vision_pipeline.apps.openarm_hand_eye import (
    TAPE_EXTRINSIC,
    ColorCamera,
    Detection,
    PatternDetector,
    Stillness,
    _add_board_arguments,
    draw_panel,
    pattern_from_args,
    pattern_from_record,
)
from vision_pipeline.calibration.cloud_registration import (
    RegistrationError,
    RegistrationResult,
    RegistrationView,
    evaluate_view,
    register_views,
)
from vision_pipeline.calibration.hand_eye import (
    Matrix,
    invert_transform,
    make_transform,
    rotation_angle_deg,
)
from vision_pipeline.calibration.model_cloud import RobotSurfaceModel
from vision_pipeline.calibration.openarm_cpf import (
    CALIBRATION_DIR,
    CURRENT_CALIBRATION,
    DEFAULT_RIG_PATH,
    RESULT_SCHEMA,
    ArmState,
    SharedMemoryArms,
    camera_orientation_summary,
    load_cpf_rig,
    load_extrinsic_file,
    save_hand_eye_result,
    transform_difference,
)
from vision_pipeline.sources.realsense import RealSenseError, RealSenseSource

WINDOW = "openarm model calibration"
SESSIONS_DIR = CALIBRATION_DIR / "registration_sessions"
_GREEN = (60, 220, 60)
_YELLOW = (0, 220, 255)
_RED = (60, 60, 255)
_WHITE = (235, 235, 235)
_PRESETS = {"custom": 0, "default": 1, "hand": 2, "high_accuracy": 3, "high_density": 4}
_ZERO = np.zeros(5)


# ---------------------------------------------------------------------------------------
# Camera helpers


def apply_depth_options(
    serial: str | None, preset: str | None, laser_power: float | None
) -> dict[str, Any]:
    """Set the D400 visual preset and laser power before the stream starts; return what stuck."""

    rs: Any = importlib.import_module("pyrealsense2")
    device = None
    for candidate in rs.context().query_devices():
        if serial is None or candidate.get_info(rs.camera_info.serial_number) == serial:
            device = candidate
            break
    if device is None:
        raise RuntimeError(f"no RealSense device with serial {serial!r}")
    sensor = device.first_depth_sensor()
    if preset is not None:
        sensor.set_option(rs.option.visual_preset, float(_PRESETS[preset]))
        time.sleep(0.3)
    if laser_power is not None:
        sensor.set_option(rs.option.laser_power, float(laser_power))
    return {
        "visual_preset": float(sensor.get_option(rs.option.visual_preset)),
        "laser_power": float(sensor.get_option(rs.option.laser_power)),
    }


def median_depth(frames: list[NDArray[np.float32]], minimum_valid: int) -> NDArray[np.float32]:
    stack = np.stack(frames)
    valid = stack > 0
    counts = valid.sum(axis=0)
    with np.errstate(all="ignore"):
        median = np.nanmedian(np.where(valid, stack, np.nan), axis=0)
    median = np.where(counts >= minimum_valid, median, 0.0)
    return cast(NDArray[np.float32], np.nan_to_num(median).astype(np.float32))


def depth_to_points(
    depth: NDArray[np.float32],
    camera_matrix: NDArray[np.float64],
    *,
    stride: int,
    min_depth: float,
    max_depth: float,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    """Camera-frame points of the valid sampled pixels and their flat pixel indices."""

    height, width = depth.shape
    rows = np.arange(0, height, stride)
    columns = np.arange(0, width, stride)
    v, u = np.meshgrid(rows, columns, indexing="ij")
    z = depth[::stride, ::stride].astype(np.float64)
    keep = np.isfinite(z) & (z >= min_depth) & (z <= max_depth)
    z, u, v = z[keep], u[keep], v[keep]
    x = (u - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
    y = (v - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
    return np.column_stack((x, y, z)), (v * width + u).astype(np.int64)


# ---------------------------------------------------------------------------------------
# Session files


@dataclass(frozen=True, slots=True)
class CapturedView:
    name: str
    depth: NDArray[np.float32]
    color: NDArray[np.uint8]
    joints: dict[str, dict[str, float]]
    hands: dict[str, list[float]]
    board: dict[str, Any] | None
    frames_averaged: int
    depth_valid_fraction: float


class RegistrationSession:
    def __init__(self, directory: Path, header: dict[str, Any]) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "session.json").write_text(json.dumps(header, indent=2))
        self.count = 0

    def add(self, view: CapturedView) -> Path:
        stem = self.directory / view.name
        np.save(f"{stem}.depth.npy", view.depth)
        cv2.imwrite(f"{stem}.color.png", view.color)
        record = {
            "name": view.name,
            "captured_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "joints": view.joints,
            "hands": view.hands,
            "board": view.board,
            "frames_averaged": view.frames_averaged,
            "depth_valid_fraction": view.depth_valid_fraction,
        }
        Path(f"{stem}.json").write_text(json.dumps(record, indent=2))
        self.count += 1
        return Path(f"{stem}.json")

    def next_name(self) -> str:
        return f"view-{self.count:02d}"


def load_registration_session(directory: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    header = json.loads((directory / "session.json").read_text())
    records = []
    for path in sorted(directory.glob("view-*.json")):
        record = json.loads(path.read_text())
        record["depth"] = np.load(directory / f"{record['name']}.depth.npy")
        records.append(record)
    return header, records


# ---------------------------------------------------------------------------------------
# Capture


def _board_record(detection: Detection) -> dict[str, Any]:
    return {
        "ids": detection.ids.tolist(),
        "corners_px": detection.corners_px.tolist(),
        "camera_from_target": detection.observation.camera_from_target.tolist(),
        "pnp_rms_px": detection.observation.reprojection_rms_pixels,
        "view_angle_deg": detection.view_angle_deg,
    }


def run_capture(args: argparse.Namespace) -> int:
    rig = load_cpf_rig(args.rig)
    arms = [] if args.static else [rig.arm(name) for name in args.arms]
    if not args.static and not arms:
        raise ValueError("give --arms or --static")
    config = load_object_selection_config(args.config)
    serial = config.realsense.serial_number
    options: dict[str, Any] = {}
    if args.depth_preset is not None or args.laser_power is not None:
        options = apply_depth_options(serial, args.depth_preset, args.laser_power)
        print(f"depth options: {options}")
    pattern = pattern_from_args(rig, args)
    print(f"plate: {pattern.describe()} lying on the table at z = {args.board_height_mm:.1f} mm")
    shm = SharedMemoryArms(rig.hw_dir, rig.shm_name) if arms else None
    if shm is not None and shm.created:
        print(
            f"created shared memory {rig.shm_name}; start the driver(s) with --shm {rig.shm_name}"
        )
    source = RealSenseSource(config.realsense)
    try:
        source.open()
    except RealSenseError as error:
        print(f"RealSense error: {error}", file=sys.stderr)
        return 2
    info = source.info
    first = source.read()
    k = first.color_intrinsics
    camera_matrix = np.array([[k.fx, 0.0, k.cx], [0.0, k.fy, k.cy], [0.0, 0.0, 1.0]])
    detector = PatternDetector(pattern, camera_matrix, 8)
    header = {
        "schema": "openarm-model-registration-session-v1",
        "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "static": bool(args.static),
        "arms": [arm.name for arm in arms],
        "joint_names": {arm.name: list(arm.profile.joint_names) for arm in arms},
        "world_frame": rig.rig.reference_frame,
        "camera_frame": rig.rig.camera_frame,
        "model_xml": str(rig.model_xml),
        "rig": str(args.rig),
        "camera": {
            "serial": info.serial_number,
            "name": info.device_name,
            "firmware": info.firmware_version,
            "usb_type_descriptor": info.usb_type_descriptor,
            "width": k.width,
            "height": k.height,
            "camera_matrix": camera_matrix.tolist(),
            "depth_scale_metres": info.depth_scale_metres,
            "depth_options": options,
        },
        "board": {**pattern.record(), "height_mm": args.board_height_mm},
        "capture": {
            "frames_per_view": args.frames_per_view,
            "settle_s": args.settle_s,
            "dq_still_rad_s": args.dq_still,
            "min_hand_move_m": args.min_hand_move,
            "min_rotation_deg": args.min_rotation_deg,
        },
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "static" if args.static else "_".join(arm.name for arm in arms)
    session = RegistrationSession(SESSIONS_DIR / f"{stamp}_{label}", header)
    print(f"session {session.directory}")
    stillness = {arm.name: Stillness(args.dq_still, args.settle_s) for arm in arms}
    captured_hands: dict[str, list[tuple[NDArray[np.float64], NDArray[np.float64]]]] = {
        arm.name: [] for arm in arms
    }
    fk_cache: dict[str, Any] = {}
    frames: list[NDArray[np.float32]] = []
    collecting = False
    force = False
    auto_left = args.auto if args.static else 0
    next_auto = time.monotonic() + 2.0
    last_print = 0.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        while True:
            frame = source.read()
            now = time.monotonic()
            image = frame.color.payload.data.copy()
            depth = frame.depth.payload.data
            states: dict[str, ArmState | None] = {}
            reasons: list[str] = []
            ready = True
            if shm is not None:
                for arm in arms:
                    state = shm.read(arm.shm_index)
                    states[arm.name] = state
                    if state is None or not state.publishing:
                        reasons.append(shm.why_not_publishing(state, arm.name))
                        ready = False
                    elif state.fault != 0:
                        reasons.append(f"{arm.name}: driver fault {state.fault}")
                        ready = False
                    else:
                        still, held = stillness[arm.name].update(state.dq, now)
                        if not still:
                            reasons.append(
                                f"{arm.name}: {'moving' if held == 0 else f'settling {held:.1f} s'}"
                            )
                            ready = False
            detection, why = detector.detect(image, lambda p: p)
            if detection is None:
                reasons.append(f"plate: {why}")
            # novelty against captured hand poses (position or rotation of the hand)
            novel = True
            if ready and arms and not force:
                for arm in arms:
                    state = states[arm.name]
                    assert state is not None
                    fk = fk_cache.get(arm.name)
                    if fk is None:
                        from vision_pipeline.calibration.openarm_cpf import MujocoForwardKinematics

                        fk = MujocoForwardKinematics(
                            rig.model_xml, arm.profile.joint_names, arm.mujoco_body
                        )
                        fk_cache[arm.name] = fk
                    pose = fk.world_from_body(state.q)
                    for position, rotation in captured_hands[arm.name]:
                        if (
                            np.linalg.norm(pose[:3, 3] - position) < args.min_hand_move
                            and rotation_angle_deg(rotation.T @ pose[:3, :3])
                            < args.min_rotation_deg
                        ):
                            novel = False
                if not novel:
                    reasons.append("hand has not moved enough since the last view")
            trigger = (ready and novel and (arms or force)) or force
            if args.static and auto_left > 0 and now >= next_auto:
                trigger = True
            if trigger:
                collecting = True
            if collecting:
                frames.append(depth.copy())
                if len(frames) >= args.frames_per_view:
                    median = median_depth(frames, max(3, args.frames_per_view // 2))
                    joints = {
                        arm.name: dict(
                            zip(arm.profile.joint_names, states[arm.name].q.tolist(), strict=True)  # type: ignore[union-attr]
                        )
                        for arm in arms
                        if states.get(arm.name) is not None
                    }
                    hands = {
                        arm.name: states[arm.name].hand.tolist()  # type: ignore[union-attr]
                        for arm in arms
                        if states.get(arm.name) is not None
                    }
                    view = CapturedView(
                        name=session.next_name(),
                        depth=median,
                        color=image,
                        joints=joints,
                        hands=hands,
                        board=None if detection is None else _board_record(detection),
                        frames_averaged=len(frames),
                        depth_valid_fraction=float(np.mean(median > 0)),
                    )
                    session.add(view)
                    for arm in arms:
                        pose = fk_cache[arm.name].world_from_body(states[arm.name].q)  # type: ignore[union-attr]
                        captured_hands[arm.name].append((pose[:3, 3].copy(), pose[:3, :3].copy()))
                    print(
                        f"captured {view.name}: depth valid {view.depth_valid_fraction:.0%}, "
                        f"plate {'yes' if view.board else 'no'}, arms {list(joints)}",
                        flush=True,
                    )
                    print("\a", end="", flush=True)
                    frames = []
                    collecting = False
                    force = False
                    if args.static and auto_left > 0:
                        auto_left -= 1
                        next_auto = now + 2.0
            # overlay
            invalid = depth <= 0
            image[invalid] = (0.6 * image[invalid]).astype(np.uint8)
            if detection is not None:
                for u, v in np.rint(detection.corners_px).astype(np.int32):
                    cv2.circle(image, (int(u), int(v)), 3, _GREEN, -1, cv2.LINE_AA)
            status = (
                f"CAPTURING {len(frames)}/{args.frames_per_view}"
                if collecting
                else ("READY - hold still" if ready and novel and arms else "WAITING")
            )
            lines = [(f"views {session.count}   {status}", _GREEN if collecting else _YELLOW)]
            for reason in reasons[:4]:
                lines.append(
                    (reason[:110], _RED if "driver" in reason or "shm" in reason else _WHITE)
                )
            if detection is not None:
                lines.append(
                    (
                        f"plate: {detection.ids.size} corners, "
                        f"PnP {detection.observation.reprojection_rms_pixels:.2f} px, "
                        f"{detection.distance_m:.2f} m",
                        _WHITE,
                    )
                )
            lines.append(
                (
                    f"depth valid {float(np.mean(~invalid)):.0%} (dark = no depth)   "
                    "space: capture now   q: quit",
                    (180, 180, 180),
                )
            )
            draw_panel(image, lines)
            cv2.imshow(WINDOW, image)
            if now - last_print >= 5.0:
                last_print = now
                print(f"{status} views={session.count} " + "; ".join(reasons[:3]), flush=True)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                force = True
            if args.static and args.auto and auto_left == 0 and not collecting:
                break
    except KeyboardInterrupt:
        print("\ninterrupted by operator")
    finally:
        source.close()
        if shm is not None:
            shm.close()
        cv2.destroyAllWindows()
    print(f"{session.count} views saved in {session.directory}")
    if session.count == 0:
        return 1
    if not args.no_solve:
        return 0 if solve_session(session.directory, args) is not None else 1
    return 0


# ---------------------------------------------------------------------------------------
# Solve


def _initial_estimate(choice: str) -> tuple[Matrix, str]:
    if choice == "auto":
        choice = "current" if CURRENT_CALIBRATION.is_file() else "tape"
    if choice == "tape":
        path = TAPE_EXTRINSIC
    elif choice == "current":
        path = CURRENT_CALIBRATION
    else:
        path = Path(choice)
    if not path.is_file():
        raise ValueError(
            f"no initial estimate at {path}; run cpf_box_handoff once for the tape estimate"
        )
    return load_extrinsic_file(path).world_from_camera, str(path)


def build_views(
    header: dict[str, Any],
    records: list[dict[str, Any]],
    model: RobotSurfaceModel,
    world_from_camera: Matrix,
    *,
    stride: int,
    crop_min: NDArray[np.float64],
    crop_max: NDArray[np.float64],
    use_plane: bool,
) -> list[RegistrationView]:
    camera_matrix = np.asarray(header["camera"]["camera_matrix"], dtype=np.float64)
    pattern = pattern_from_record(header["board"])
    plane_height = float(header["board"].get("height_mm", 0.0)) / 1000.0
    views: list[RegistrationView] = []
    for record in records:
        points, _ = depth_to_points(
            record["depth"], camera_matrix, stride=stride, min_depth=0.25, max_depth=2.5
        )
        world = points @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
        keep = np.all((world >= crop_min) & (world <= crop_max), axis=1)
        points = points[keep]
        joints: dict[str, float] = {}
        groups = ["body"]
        for arm, values in record.get("joints", {}).items():
            joints.update(values)
            groups.append(arm)
        surface = model.surface(joints, groups)
        plane = None
        board = record.get("board")
        if use_plane and board:
            camera_from_target = np.asarray(board["camera_from_target"], dtype=np.float64)
            ids = np.asarray(board["ids"], dtype=np.int64)
            corners_target = pattern.object_points[ids]
            plane = corners_target @ camera_from_target[:3, :3].T + camera_from_target[:3, 3]
        views.append(
            RegistrationView(
                str(record["name"]), points, surface.points, surface.normals, plane, plane_height
            )
        )
    return views


def solve_session(session_dir: Path, args: argparse.Namespace) -> dict[str, Any] | None:
    header, records = load_registration_session(session_dir)
    if not records:
        print("no views in the session")
        return None
    init, init_source = _initial_estimate(args.init)
    print(f"initial estimate from {init_source}")
    model = RobotSurfaceModel(Path(header["model_xml"]))
    crop_min = np.asarray(args.crop_min, dtype=np.float64)
    crop_max = np.asarray(args.crop_max, dtype=np.float64)
    x = init
    result: RegistrationResult | None = None
    for round_index in range(2):
        views = build_views(
            header, records, model, x, stride=args.stride, crop_min=crop_min, crop_max=crop_max,
            use_plane=not args.no_plane,
        )  # fmt: skip
        try:
            result = register_views(
                views,
                x,
                max_distance_m=args.max_distance if round_index == 0 else args.max_distance / 2,
                final_distance_m=args.final_distance,
                iterations=args.iterations,
                prior_translation_m=args.prior_mm / 1000.0 if args.prior_mm > 0 else None,
                prior_rotation_deg=args.prior_deg if args.prior_deg > 0 else None,
            )
        except RegistrationError as error:
            print(f"registration failed: {error}")
            return None
        x = result.world_from_camera
        print(
            f"round {round_index + 1}: {result.inliers} inliers, RMS {result.rms_mm:.1f} mm, "
            f"plane RMS {result.plane_rms_mm}, iterations {result.iterations}"
        )
    assert result is not None
    views = build_views(
        header, records, model, x, stride=args.stride, crop_min=crop_min, crop_max=crop_max,
        use_plane=not args.no_plane,
    )  # fmt: skip
    stability: dict[str, Any] = {}
    if len(views) >= 3:
        worst_mm = worst_deg = 0.0
        held_rms = []
        for index, held in enumerate(views):
            others = views[:index] + views[index + 1 :]
            try:
                partial = register_views(
                    others, x, max_distance_m=args.max_distance / 2,
                    final_distance_m=args.final_distance, iterations=args.iterations,
                    prior_translation_m=args.prior_mm / 1000.0 if args.prior_mm > 0 else None,
                    prior_rotation_deg=args.prior_deg if args.prior_deg > 0 else None,
                )  # fmt: skip
            except RegistrationError:
                continue
            mm, deg = transform_difference(x, partial.world_from_camera)
            worst_mm, worst_deg = max(worst_mm, mm), max(worst_deg, deg)
            held_rms.append(
                evaluate_view(
                    held, partial.world_from_camera, distance_m=args.final_distance
                ).rms_mm
            )
        stability = {
            "leave_one_out_max_translation_mm": worst_mm,
            "leave_one_out_max_rotation_deg": worst_deg,
            "held_out_rms_mm": held_rms,
        }
    record = {
        "schema": RESULT_SCHEMA,
        "method": (
            "depth-to-model registration: robust point-to-plane ICP of RealSense depth against "
            "the MuJoCo visual meshes posed with shared-memory joint state, plus the marker "
            "plate on the table as the z = 0 plane"
        ),
        "world_frame": header["world_frame"],
        "camera_frame": header["camera_frame"],
        "camera_serial": header["camera"]["serial"],
        "arm": "+".join(header["arms"]) or "static",
        "measured_on": datetime.now(UTC).date().isoformat(),
        "session_dir": str(session_dir),
        "color_resolution": [header["camera"]["width"], header["camera"]["height"]],
        "camera_matrix": header["camera"]["camera_matrix"],
        "board": header["board"],
        "initial_estimate": init_source,
        "world_from_camera": x.tolist(),
        "camera_from_world": invert_transform(x).tolist(),
        "camera_position_world_metres": x[:3, 3].tolist(),
        **camera_orientation_summary(x),
        "views": [dataclasses.asdict(fit) for fit in result.views],
        "samples": len(result.views),
        "inliers": result.inliers,
        "rms_mm": result.rms_mm,
        "plane_points": result.plane_points,
        "plane_rms_mm": result.plane_rms_mm,
        "observability": list(result.observability),
        "weakest_direction": result.weakest,
        "converged": result.converged,
        "stability": stability,
        "comparisons": {},
    }
    comparisons: dict[str, Any] = {}
    mm, deg = transform_difference(x, init)
    comparisons["initial_estimate"] = {"translation_mm": mm, "rotation_deg": deg}
    if TAPE_EXTRINSIC.is_file():
        tape = load_extrinsic_file(TAPE_EXTRINSIC).world_from_camera
        mm, deg = transform_difference(x, tape)
        comparisons["tape_imu_extrinsic"] = {"translation_mm": mm, "rotation_deg": deg}
    for name in ("current_right", "current_left"):
        other = CALIBRATION_DIR / f"{name}.json"
        if other.is_file():
            mm, deg = transform_difference(x, load_extrinsic_file(other).world_from_camera)
            comparisons[f"hand_eye_{name}"] = {"translation_mm": mm, "rotation_deg": deg}
    record["comparisons"] = comparisons
    print_result(record)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_hand_eye_result(
        CALIBRATION_DIR / f"model_registration_{stamp}.json",
        record,
        make_current=not args.no_current,
    )
    (CALIBRATION_DIR / "current_registration.json").write_text(json.dumps(record, indent=2))
    (session_dir / "result.json").write_text(json.dumps(record, indent=2))
    print(
        f"saved {path}"
        + ("" if args.no_current else f"\nsaved {CURRENT_CALIBRATION} (used by cpf_box_handoff)")
    )
    return record


def print_result(record: dict[str, Any]) -> None:
    print("\n=== depth-to-model registration ===")
    position = np.round(record["camera_position_world_metres"], 4).tolist()
    print(f"camera position in {record['world_frame']}: {position} m")
    print(
        f"heading {record['heading_deg']:.2f} deg, pitch down "
        f"{record['camera_pitch_down_deg']:.2f} deg, image roll {record['camera_roll_deg']:.2f} deg"
    )
    print(
        f"{record['inliers']} depth points within tolerance over {record['samples']} view(s), "
        f"point-to-plane RMS {record['rms_mm']:.1f} mm; plate corners {record['plane_points']}, "
        f"plane RMS {record['plane_rms_mm']}"
    )
    for fit in record["views"]:
        print(
            f"  {fit['name']}: {fit['inliers']}/{fit['candidates']} points, "
            f"RMS {fit['rms_mm']:.1f} mm, median {fit['median_mm']:.1f} mm, "
            f"model coverage {fit['model_coverage']:.0%}"
        )
    obs = record["observability"]
    weak = "   <-- poorly constrained, add arm poses" if obs[0] < 0.01 else ""
    print(
        f"observability (weakest/strongest) {obs[0]:.4f}; "
        f"weakest direction: {record['weakest_direction']}{weak}"
    )
    if record["stability"]:
        s = record["stability"]
        print(
            f"leave-one-out: camera moves at most {s['leave_one_out_max_translation_mm']:.1f} mm / "
            f"{s['leave_one_out_max_rotation_deg']:.2f} deg; "
            f"held-out RMS {np.round(s['held_out_rms_mm'], 1).tolist()} mm"
        )
    for name, comparison in record["comparisons"].items():
        print(
            f"vs {name}: {comparison['translation_mm']:.1f} mm, "
            f"{comparison['rotation_deg']:.2f} deg"
        )


def run_solve(args: argparse.Namespace) -> int:
    return 0 if solve_session(Path(args.session), args) is not None else 1


def run_residuals(args: argparse.Namespace) -> int:
    """Colour each view's depth points by distance to the posed model and save PNGs."""

    from scipy.spatial import cKDTree

    session_dir = Path(args.session)
    header, records = load_registration_session(session_dir)
    calibration = load_extrinsic_file(args.calibration)
    x = calibration.world_from_camera
    camera_from_world = invert_transform(x)
    model = RobotSurfaceModel(Path(header["model_xml"]))
    camera_matrix = np.asarray(header["camera"]["camera_matrix"], dtype=np.float64)
    fx, fy, cx, cy = (
        camera_matrix[0, 0],
        camera_matrix[1, 1],
        camera_matrix[0, 2],
        camera_matrix[1, 2],
    )
    for record in records:
        depth = record["depth"]
        height, width = depth.shape
        points, flat = depth_to_points(
            depth, camera_matrix, stride=1, min_depth=0.25, max_depth=2.5
        )
        world = points @ x[:3, :3].T + x[:3, 3]
        joints: dict[str, float] = {}
        groups = ["body"]
        for arm, values in record.get("joints", {}).items():
            joints.update(values)
            groups.append(arm)
        surface = model.surface(joints, groups)
        distance, _ = cKDTree(surface.points).query(world, distance_upper_bound=args.max_distance)
        loaded = cv2.imread(str(session_dir / f"{record['name']}.color.png"))
        overlay = np.zeros((height, width, 3), np.uint8) if loaded is None else np.asarray(loaded)
        overlay = overlay.copy()
        near = np.isfinite(distance)
        scaled = np.clip(distance[near] / args.max_distance * 255, 0, 255).astype(np.uint8)
        colours = cv2.applyColorMap(scaled, cv2.COLORMAP_JET).reshape(-1, 3)
        rows, cols = np.divmod(flat[near], width)
        overlay[rows, cols] = colours
        model_camera = surface.points @ camera_from_world[:3, :3].T + camera_from_world[:3, 3]
        front = model_camera[model_camera[:, 2] > 0.05]
        u = np.rint(fx * front[:, 0] / front[:, 2] + cx).astype(int)
        v = np.rint(fy * front[:, 1] / front[:, 2] + cy).astype(int)
        inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        overlay[v[inside], u[inside]] = (255, 255, 255)
        within = int(near.sum())
        text = (
            f"{record['name']}: {within} depth points within {args.max_distance * 1e3:.0f} mm "
            "of the model (blue near, red far); white = model"
        )
        cv2.putText(overlay, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, _WHITE, 1, cv2.LINE_AA)
        out = session_dir / f"{record['name']}.residual.png"
        cv2.imwrite(str(out), overlay)
        print(
            f"{record['name']}: {within} points within {args.max_distance * 1e3:.0f} mm; "
            f"wrote {out}"
        )
    return 0


# ---------------------------------------------------------------------------------------
# Plate placed against the base


def plate_world_from_camera(
    camera_from_target: Matrix,
    object_points: NDArray[np.float64],
    initial_world_from_camera: Matrix,
    *,
    contact_x_m: float,
    margin_m: float,
    lateral_m: float,
    thickness_m: float,
    long_side_along: str,
) -> tuple[Matrix, Matrix, float]:
    """Camera pose from a plate lying flat with one edge pushed against the base.

    The plate's orientation is axis-aligned by contact: its long side runs along world
    ``long_side_along`` (``"y"`` when it lies along the base's front edge), its printed
    face is up. The initial estimate only picks which of the two possible directions the
    pattern's long axis points. Its nearest black edge is ``margin_m`` in front of the
    contact edge at ``contact_x_m``, its centre is ``lateral_m`` from the column centre,
    and its face is ``thickness_m`` above the table. Returns ``world_from_camera``,
    ``world_from_target`` and the angle (deg) by which the initial estimate disagreed
    with the snapped orientation.
    """

    extent = object_points.max(axis=0) - object_points.min(axis=0)
    long_axis = int(np.argmax(extent[:2]))
    short_axis = 1 - long_axis
    axes_world = initial_world_from_camera[:3, :3] @ camera_from_target[:3, :3]
    world_long = np.array([1.0, 0.0, 0.0]) if long_side_along == "x" else np.array([0.0, 1.0, 0.0])
    sign = 1.0 if float(axes_world[:, long_axis] @ world_long) >= 0.0 else -1.0
    columns: list[NDArray[np.float64] | None] = [None, None, np.array([0.0, 0.0, -1.0])]
    columns[long_axis] = sign * world_long
    if short_axis == 0:
        columns[0] = np.cross(columns[1], columns[2])  # type: ignore[arg-type]
    else:
        columns[1] = np.cross(columns[2], columns[0])  # type: ignore[arg-type]
    rotation = np.column_stack(columns)  # type: ignore[arg-type]
    cosine = float(axes_world[:, long_axis] @ rotation[:, long_axis])
    deviation_deg = math.degrees(math.acos(max(-1.0, min(1.0, cosine))))
    relative = object_points @ rotation.T
    translation = np.array(
        [
            contact_x_m + margin_m - float(relative[:, 0].min()),
            lateral_m - float(relative[:, 1].mean()),
            thickness_m - float(relative[:, 2].mean()),
        ]
    )
    world_from_target = make_transform(rotation, translation)
    return (
        world_from_target @ invert_transform(camera_from_target),
        world_from_target,
        deviation_deg,
    )


def run_plate(args: argparse.Namespace) -> int:
    rig = load_cpf_rig(args.rig)
    pattern = pattern_from_args(rig, args)
    init, init_source = _initial_estimate(args.init)
    model = RobotSurfaceModel(rig.model_xml)
    body = model.surface({}, ("body",)).points
    footprint = body[body[:, 2] < 0.02]
    column = body[(body[:, 2] > 0.1) & (body[:, 2] < 0.6) & (np.abs(body[:, 1]) < 0.02)]
    print(
        f"model: base plate front edge at x = {footprint[:, 0].max():.3f} m, column front face at "
        f"x = {column[:, 0].max():.3f} m; you gave --contact-x {args.contact_x:.3f}"
    )
    print(f"plate: {pattern.describe()}; orientation disambiguated with {init_source}")
    width, height = int(args.color[0]), int(args.color[1])
    fps = int(args.color[2]) if len(args.color) > 2 else None
    camera = ColorCamera(rig.rig.camera_serial, width, height, fps)
    detector = PatternDetector(pattern, camera.camera_matrix, 12)
    per_id: dict[int, list[NDArray[np.float64]]] = {}
    frames = 0
    print(f"observing the plate for {args.seconds:.0f} s; do not touch it", flush=True)
    try:
        deadline = time.monotonic() + args.seconds
        last_image = camera.read()
        while time.monotonic() < deadline:
            last_image = camera.read()
            detection, _ = detector.detect(last_image, camera.undistort_points)
            if detection is None:
                continue
            frames += 1
            for marker_id, corner in zip(
                detection.ids.tolist(), detection.corners_undistorted, strict=True
            ):
                per_id.setdefault(int(marker_id), []).append(corner)
    finally:
        camera.close()
    if frames < 5:
        raise ValueError(f"plate detected in only {frames} frames; move it into view")
    ids = np.asarray(sorted(k for k, v in per_id.items() if len(v) >= frames // 2), dtype=np.int32)
    mean_corners = np.stack([np.mean(np.stack(per_id[int(k)]), axis=0) for k in ids])
    observation = detector.observation_from_points(ids, mean_corners)
    if observation is None:
        raise ValueError("PnP failed on the averaged plate corners")
    x, world_from_target, deviation = plate_world_from_camera(
        observation.camera_from_target,
        pattern.object_points[ids],
        init,
        contact_x_m=args.contact_x,
        margin_m=args.margin_mm / 1000.0,
        lateral_m=args.lateral_mm / 1000.0,
        thickness_m=args.thickness_mm / 1000.0,
        long_side_along=args.long_side_along,
    )
    corners_world = (
        pattern.object_points[ids] @ world_from_target[:3, :3].T + world_from_target[:3, 3]
    )
    orientation = camera_orientation_summary(x)
    mm, deg = transform_difference(x, init)
    print("\n=== plate against the base ===")
    distance_m = float(np.linalg.norm(observation.camera_from_target[:3, 3]))
    print(
        f"PnP: {ids.size} corners averaged over {frames} frames, "
        f"RMS {observation.reprojection_rms_pixels:.2f} px, "
        f"plate {distance_m:.3f} m from the camera"
    )
    print(
        f"plate in world: x {corners_world[:, 0].min():.3f}..{corners_world[:, 0].max():.3f}, "
        f"y {corners_world[:, 1].min():.3f}..{corners_world[:, 1].max():.3f}, "
        f"z {corners_world[:, 2].mean():.4f} m"
    )
    print(
        f"initial estimate disagreed with the snapped plate orientation by {deviation:.1f} deg"
        + (
            "   <-- large: check --long-side-along and the plate placement"
            if deviation > 25
            else ""
        )
    )
    print(f"camera position in {rig.rig.reference_frame}: {np.round(x[:3, 3], 4).tolist()} m")
    print(
        f"heading {orientation['heading_deg']:.2f} deg, pitch down "
        f"{orientation['camera_pitch_down_deg']:.2f} deg, "
        f"image roll {orientation['camera_roll_deg']:.2f} deg"
    )
    print(f"vs {init_source}: {mm:.1f} mm, {deg:.2f} deg")
    record = {
        "schema": RESULT_SCHEMA,
        "method": (
            "plate against the base: PnP of the marker plate lying flat with its edge pushed "
            "against the base, orientation axis-aligned by contact, offsets measured by ruler"
        ),
        "world_frame": rig.rig.reference_frame,
        "camera_frame": rig.rig.camera_frame,
        "camera_serial": camera.serial,
        "arm": "static",
        "measured_on": datetime.now(UTC).date().isoformat(),
        "color_resolution": [camera.width, camera.height],
        "camera_matrix": camera.camera_matrix.tolist(),
        "board": pattern.record(),
        "placement": {
            "contact_x_m": args.contact_x,
            "margin_mm": args.margin_mm,
            "lateral_mm": args.lateral_mm,
            "thickness_mm": args.thickness_mm,
            "long_side_along": args.long_side_along,
            "orientation_deviation_from_initial_deg": deviation,
        },
        "initial_estimate": init_source,
        "world_from_camera": x.tolist(),
        "camera_from_world": invert_transform(x).tolist(),
        "camera_position_world_metres": x[:3, 3].tolist(),
        **orientation,
        "world_from_target": world_from_target.tolist(),
        "camera_from_target": observation.camera_from_target.tolist(),
        "pnp_rms_px": observation.reprojection_rms_pixels,
        "frames": frames,
        "samples": 1,
        "reprojection_rms_pixels": observation.reprojection_rms_pixels,
        "comparisons": {"initial_estimate": {"translation_mm": mm, "rotation_deg": deg}},
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = save_hand_eye_result(
        CALIBRATION_DIR / f"plate_{stamp}.json", record, make_current=not args.no_current
    )
    (CALIBRATION_DIR / "current_plate.json").write_text(json.dumps(record, indent=2))
    print(
        f"saved {path}"
        + ("" if args.no_current else f"\nsaved {CURRENT_CALIBRATION} (used by cpf_box_handoff)")
    )
    return 0


# ---------------------------------------------------------------------------------------
# CLI


def _add_solve_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--init", default="auto", help="auto | tape | current | path to an extrinsic JSON"
    )
    parser.add_argument("--stride", type=int, default=2, help="depth pixel stride")
    parser.add_argument(
        "--max-distance", type=float, default=0.04, help="initial correspondence distance, m"
    )
    parser.add_argument(
        "--final-distance", type=float, default=0.01, help="final correspondence distance, m"
    )
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument(
        "--crop-min", type=float, nargs=3, default=[-0.35, -0.9, 0.03], metavar="M",
        help="world-frame box that must contain the robot (under the initial estimate)",
    )  # fmt: skip
    parser.add_argument("--crop-max", type=float, nargs=3, default=[1.0, 0.9, 1.3], metavar="M")
    parser.add_argument("--no-plane", action="store_true", help="ignore the plate plane constraint")
    parser.add_argument(
        "--prior-mm", type=float, default=30.0, help="prior sigma on camera position; 0 disables"
    )
    parser.add_argument(
        "--prior-deg", type=float, default=3.0, help="prior sigma on camera rotation; 0 disables"
    )
    parser.add_argument("--no-current", action="store_true", help="do not overwrite current.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--rig", default=str(DEFAULT_RIG_PATH))
    commands = parser.add_subparsers(dest="command", required=True)

    capture = commands.add_parser("capture", help="record depth + joint state views, then solve")
    capture.add_argument("--arms", nargs="+", default=["right"], choices=("right", "left"))
    capture.add_argument("--static", action="store_true", help="no driver: torso and plate only")
    capture.add_argument(
        "--auto", type=int, default=0, help="static: capture this many views automatically"
    )
    capture.add_argument("--config", default=None, help="object_selection.yaml (camera settings)")
    _add_board_arguments(capture)
    capture.add_argument(
        "--board-height-mm", type=float, default=0.0, help="plate top above the table"
    )
    capture.add_argument(
        "--depth-preset",
        choices=tuple(_PRESETS),
        default=None,
        help="D400 visual preset to apply before streaming; default keeps the camera's",
    )
    capture.add_argument(
        "--laser-power",
        type=float,
        default=None,
        help="D400 laser power to apply before streaming; default leaves the camera alone "
        "(changing it on a just-released D435i has hung the stream)",
    )
    capture.add_argument("--frames-per-view", type=int, default=12)
    capture.add_argument("--settle-s", type=float, default=0.8)
    capture.add_argument("--dq-still", type=float, default=0.03)
    capture.add_argument("--min-hand-move", type=float, default=0.08, help="m between views")
    capture.add_argument("--min-rotation-deg", type=float, default=15.0)
    capture.add_argument("--no-solve", action="store_true")
    _add_solve_arguments(capture)
    capture.set_defaults(run=run_capture)

    solve = commands.add_parser("solve", help="register a recorded session")
    solve.add_argument("--session", required=True)
    _add_solve_arguments(solve)
    solve.set_defaults(run=run_solve)

    plate = commands.add_parser(
        "plate", help="extrinsic from the plate lying flat against the base (no driver)"
    )
    _add_board_arguments(plate)
    plate.add_argument(
        "--contact-x", type=float, default=0.095,
        help="world x of the contact edge: base plate front 0.095, column face 0.030",
    )  # fmt: skip
    plate.add_argument(
        "--margin-mm", type=float, required=True, help="contact edge to the nearest black tag edge"
    )
    plate.add_argument(
        "--lateral-mm", type=float, default=0.0, help="pattern centre offset to the robot's left"
    )
    plate.add_argument(
        "--thickness-mm", type=float, default=0.0, help="height of the printed face above the table"
    )
    plate.add_argument(
        "--long-side-along",
        choices=("y", "x"),
        default="y",
        help="world axis the plate's long side runs along",
    )
    plate.add_argument(
        "--init",
        default="auto",
        help="auto | tape | current | path; only picks the orientation sign",
    )
    plate.add_argument("--seconds", type=float, default=4.0)
    plate.add_argument("--no-current", action="store_true")
    plate.add_argument(
        "--color", type=int, nargs="+", default=[1280, 720], metavar="N",
        help="colour WIDTH HEIGHT [FPS]",
    )  # fmt: skip
    plate.set_defaults(run=run_plate)

    residuals = commands.add_parser("residuals", help="per-view images of depth-to-model distance")
    residuals.add_argument("--session", required=True)
    residuals.add_argument("--calibration", default=str(CURRENT_CALIBRATION))
    residuals.add_argument("--max-distance", type=float, default=0.03)
    residuals.set_defaults(run=run_residuals)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "config", None) is None and hasattr(args, "config"):
        from vision_pipeline.apps.object_selection_config import DEFAULT_CONFIG_PATH

        args.config = str(DEFAULT_CONFIG_PATH)
    try:
        return int(args.run(args))
    except (RuntimeError, ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
