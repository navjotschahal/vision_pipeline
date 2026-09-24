"""Webcam-only planar teleoperation of CPF PoC's KUKA iiwa14 in MuJoCo."""

from __future__ import annotations

import math
import multiprocessing as mp
import sys
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import yaml

from vision_pipeline.apps.kuka_camera_preview import run_preview
from vision_pipeline.contracts import TimePoint
from vision_pipeline.geometry.spatial import Pose3D
from vision_pipeline.perception.pose import HumanPose2D
from vision_pipeline.perception.pose.yolo import YoloPoseConfig, YoloPoseEstimator
from vision_pipeline.sources.opencv_webcam import OpenCvWebcam, WebcamConfig
from vision_pipeline.teleoperation import TeleopState
from vision_pipeline.teleoperation.webcam_planar import WebcamPlanarRetargeter


def default_poc_root() -> Path:
    return Path(__file__).resolve().parents[4] / "Navjot_IS" / "poc"


@dataclass(slots=True)
class CameraObservation:
    pose: HumanPose2D | None = None
    preview: np.ndarray | None = None
    sample_id: str | None = None
    width: int = 0
    height: int = 0
    people: int = 0
    error: Exception | None = None


class CameraWorker:
    def __init__(self, camera: int, model: str, device: str | None) -> None:
        self.camera = camera
        self.model = model
        self.device = device
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = CameraObservation()
        self._thread = threading.Thread(target=self._run, name="kuka-webcam-pose", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def latest(self) -> CameraObservation:
        with self._lock:
            return self._latest

    def _run(self) -> None:
        try:
            estimator = YoloPoseEstimator(YoloPoseConfig(model=self.model, device=self.device))
            with OpenCvWebcam(WebcamConfig(device_index=self.camera)) as webcam:
                while not self._stop.is_set():
                    sample = webcam.read()
                    pose = estimator.estimate(sample.payload, sample.header)
                    scale = min(1.0, 960 / sample.payload.width)
                    preview = cv2.resize(
                        sample.payload.data,
                        (round(sample.payload.width * scale), round(sample.payload.height * scale)),
                    )
                    with self._lock:
                        self._latest = CameraObservation(
                            pose=pose if estimator.last_person_count == 1 else None,
                            preview=preview,
                            sample_id=sample.header.sample_id,
                            width=sample.payload.width,
                            height=sample.payload.height,
                            people=estimator.last_person_count,
                        )
        except Exception as error:
            with self._lock:
                self._latest = CameraObservation(error=error)


class PositionIK:
    """Damped positional IK with joint-range clipping; orientation stays unconstrained."""

    def __init__(
        self, model: mujoco.MjModel, site_id: int, qpos_indices: np.ndarray, dof_indices: np.ndarray
    ) -> None:
        self.model = model
        self.site_id = site_id
        self.qpos_indices = qpos_indices
        self.dof_indices = dof_indices
        self._data = mujoco.MjData(model)
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._lower = model.jnt_range[: len(qpos_indices), 0] + 0.03
        self._upper = model.jnt_range[: len(qpos_indices), 1] - 0.03

    def solve(self, q_now: np.ndarray, target_xyz: np.ndarray) -> tuple[np.ndarray, float]:
        q = q_now.copy()
        for _ in range(12):
            self._data.qpos[self.qpos_indices] = q
            mujoco.mj_forward(self.model, self._data)
            error = target_xyz - self._data.site_xpos[self.site_id]
            if np.linalg.norm(error) < 0.008:
                break
            mujoco.mj_jacSite(self.model, self._data, self._jacp, self._jacr, self.site_id)
            jac = self._jacp[:, self.dof_indices]
            step = jac.T @ np.linalg.solve(jac @ jac.T + 0.03**2 * np.eye(3), error)
            q = np.clip(q + np.clip(step, -0.08, 0.08), self._lower, self._upper)
        self._data.qpos[self.qpos_indices] = q
        mujoco.mj_forward(self.model, self._data)
        residual = float(np.linalg.norm(target_xyz - self._data.site_xpos[self.site_id]))
        return q, residual


def run_kuka_webcam_teleop(
    *,
    poc_root: Path,
    camera: int = 0,
    model_name: str = "yolo11n-pose.pt",
    device: str | None = None,
    headless: bool = False,
    max_seconds: float = 0,
) -> int:
    """Run a simulated KUKA; the CPF PoC is used read-only for model, gains, controller."""
    if not math.isfinite(max_seconds) or max_seconds < 0:
        raise ValueError("max_seconds must be finite and non-negative")
    if headless and max_seconds <= 0:
        raise ValueError("headless mode requires --max-seconds")
    poc_root = poc_root.resolve()
    config_path = poc_root / "src/sim/configs/robots/kuka_iiwa_14.yml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    sim_root = poc_root / "src/sim"
    scene_path = sim_root / config["paths"]["scene_xml"]
    mujoco_model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(mujoco_model)
    poc_src = str(poc_root / "src")
    if poc_src not in sys.path:
        sys.path.insert(0, poc_src)
    from controllers import (  # type: ignore[import-not-found]
        JointImpedanceController,
        MujocoRobotView,
    )

    robot = MujocoRobotView.from_config(mujoco_model, data, config)
    controller = JointImpedanceController(robot, config["controller"])
    data.qpos[robot.qpos_idx] = np.asarray(config["simulation"]["q_home"], dtype=float)
    mujoco.mj_forward(mujoco_model, data)
    site_id = mujoco.mj_name2id(mujoco_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        raise ValueError("KUKA scene has no attachment_site")
    ik = PositionIK(mujoco_model, site_id, robot.qpos_idx, robot.dof_indices)
    retargeter = WebcamPlanarRetargeter()
    q_target = robot.q.copy()
    worker = CameraWorker(camera, model_name, device)
    worker.start()
    preview_process = None
    preview_frames = None
    preview_keys = None
    if not headless:
        context = mp.get_context("spawn")
        preview_frames = context.Queue(maxsize=1)
        preview_keys = context.Queue(maxsize=8)
        preview_process = context.Process(
            target=run_preview, args=(preview_frames, preview_keys), daemon=True
        )
        preview_process.start()
    clutch_key = threading.Event()
    hold_key = threading.Event()
    quit_key = threading.Event()

    def key_callback(keycode: int) -> None:
        key = chr(keycode).lower() if 0 <= keycode < 128 else ""
        if key == "c":
            clutch_key.set()
        elif key == "h":
            hold_key.set()
        elif key == "q" or keycode == 256:
            quit_key.set()

    last_frame_id: str | None = None
    last_preview_id: str | None = None
    status = "HOLD"
    reported_status = ""
    start = time.monotonic()
    next_tick = start
    next_ui = start

    def cycle(viewer: Any = None) -> None:
        nonlocal q_target, status, reported_status, last_frame_id, last_preview_id
        nonlocal next_tick, next_ui
        observation = worker.latest()
        if observation.error is not None:
            raise RuntimeError(f"camera or pose inference failed: {observation.error}")
        if preview_keys is not None:
            from queue import Empty

            while True:
                try:
                    action = preview_keys.get_nowait()
                except Empty:
                    break
                if action == "c":
                    clutch_key.set()
                elif action == "h":
                    hold_key.set()
                elif action == "q":
                    quit_key.set()
        if quit_key.is_set():
            raise KeyboardInterrupt
        now_wall = time.monotonic()
        if hold_key.is_set():
            hold_key.clear()
            retargeter.disarm()
            q_target = robot.q.copy()
            status = "HOLD: disarmed"
        if clutch_key.is_set():
            clutch_key.clear()
            if observation.pose is not None:
                now = TimePoint(
                    time.monotonic_ns(), observation.pose.source_header.received_at.clock
                )
                site_pose = Pose3D(
                    retargeter.config.robot_base_frame,
                    tuple(float(x) for x in data.site_xpos[site_id]),
                    (0.0, 0.0, 0.0, 1.0),
                )
                try:
                    retargeter.engage(
                        observation.pose,
                        image_size=(observation.width, observation.height),
                        robot_pose=site_pose,
                        now=now,
                    )
                    status = "ACTIVE: clutch captured"
                except ValueError as error:
                    status = f"HOLD: {error}"
            else:
                status = (
                    f"HOLD: clutch needs exactly one visible person (detected {observation.people})"
                )
        if observation.pose is not None:
            now = TimePoint(time.monotonic_ns(), observation.pose.source_header.received_at.clock)
            frame_id = observation.pose.source_header.sample_id
            if frame_id != last_frame_id:
                last_frame_id = frame_id
                if retargeter.is_armed:
                    command = retargeter.update(observation.pose, now=now, deadman_pressed=True)
                    if command.state is TeleopState.ACTIVE and command.arm is not None:
                        xyz = np.asarray(command.arm.pose.position_metres)
                        candidate, residual = ik.solve(robot.q, xyz)
                        if residual < 0.05 and command.expires_at.nanoseconds > now.nanoseconds:
                            q_target = candidate
                            status = "ACTIVE"
                        else:
                            q_target = robot.q.copy()
                            status = "HOLD: IK unreachable"
                    else:
                        q_target = robot.q.copy()
                        status = f"HOLD: {command.reason}"
        if observation.pose is None or (
            observation.pose is not None
            and (time.monotonic_ns() - observation.pose.source_header.received_at.nanoseconds)
            > 250_000_000
        ):
            q_target = robot.q.copy()
            status = "HOLD: no fresh single-person pose"
        if status != reported_status:
            print(f"KUKA sim: {status}", flush=True)
            reported_status = status
        tau = controller.compute_torque(q_desired=q_target)
        controller.apply(tau)
        mujoco.mj_step(mujoco_model, data)
        if now_wall >= next_ui:
            if viewer is not None:
                viewer.sync()
            if (
                preview_frames is not None
                and observation.preview is not None
                and observation.sample_id != last_preview_id
            ):
                from queue import Full

                try:
                    preview_frames.put_nowait(
                        (
                            observation.preview,
                            observation.pose,
                            status,
                            observation.people,
                            (observation.width, observation.height),
                        )
                    )
                    last_preview_id = observation.sample_id
                except Full:
                    pass
            next_ui = now_wall + 1 / 30
        next_tick += mujoco_model.opt.timestep
        delay = next_tick - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        elif delay < -0.1:
            next_tick = time.monotonic()

    try:
        if headless:
            while time.monotonic() - start < max_seconds:
                cycle()
        else:
            print("MuJoCo viewer keys: c=clutch, h=hold/disarm, q=quit", flush=True)
            with mujoco.viewer.launch_passive(
                mujoco_model, data, key_callback=key_callback
            ) as viewer:
                while viewer.is_running() and (
                    max_seconds <= 0 or time.monotonic() - start < max_seconds
                ):
                    cycle(viewer)
    except KeyboardInterrupt:
        pass
    finally:
        worker.close()
        if preview_process is not None:
            assert preview_frames is not None
            with suppress(Exception):
                preview_frames.put_nowait(None)
            preview_process.join(timeout=2.0)
            if preview_process.is_alive():
                preview_process.terminate()
                preview_process.join(timeout=1.0)
    return 0


__all__ = ["PositionIK", "run_kuka_webcam_teleop", "default_poc_root"]
