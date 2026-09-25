"""OpenArm/CPF side of hand-eye calibration: rig config, forward kinematics, joint state, results.

CPF drives the OpenArm without ROS: a C++ RT loop per arm publishes ``q``/``dq`` into a
seqlocked shared-memory page (``cpf/src/openarm_hw/tools/shm_layout.py``). This module
turns that into what :mod:`vision_pipeline.calibration.hand_eye` needs:

- :class:`MujocoForwardKinematics` evaluates ``world_from_flange`` from the seven arm
  joint angles with the same MuJoCo model the controller uses. The MuJoCo world body is
  ``openarm_body_link0`` (identity pose), i.e. CPF's world frame, so the hand-eye result
  ``base_from_camera`` is directly ``world_from_camera`` for the grasp planner.
- :class:`SharedMemoryArms` reads one arm's latest joint state without touching commands.
- :func:`save_hand_eye_result` / :func:`load_extrinsic_file` store the calibration in a
  JSON file that ``cpf_box_handoff`` consumes in place of the tape + IMU estimate.

Nothing here commands motion.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from vision_pipeline.contracts import CalibrationRef, FrameId
from vision_pipeline.geometry.spatial import RigidTransform3D

from .hand_eye import Matrix, invert_transform, make_transform, rigid_transform_from_matrix
from .robot_profile import ArmProfile, CalibrationProfileError, CalibrationRig, load_calibration_rig

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RIG_PATH = REPOSITORY_ROOT / "configs" / "calibration" / "openarm_v1_bimanual.yaml"
CALIBRATION_DIR = REPOSITORY_ROOT / "calibrations" / "openarm_v1"
CURRENT_CALIBRATION = CALIBRATION_DIR / "current.json"
RESULT_SCHEMA = "openarm-cpf-hand-eye-v1"


class CpfRigError(ValueError):
    """The rig profile lacks the CPF section or it does not match the machine."""


@dataclass(frozen=True, slots=True)
class CpfArm:
    """One arm as CPF exposes it: shared-memory slot, MuJoCo flange body, joint names."""

    profile: ArmProfile
    shm_index: int
    mujoco_body: str

    @property
    def name(self) -> str:
        return self.profile.name


@dataclass(frozen=True, slots=True)
class CpfRig:
    rig: CalibrationRig
    model_xml: Path
    hw_dir: Path
    shm_name: str
    arms: dict[str, CpfArm]

    def arm(self, name: str) -> CpfArm:
        try:
            return self.arms[name]
        except KeyError as error:
            raise CpfRigError(f"rig has no CPF arm {name!r}; have {sorted(self.arms)}") from error


def load_cpf_rig(path: str | Path = DEFAULT_RIG_PATH) -> CpfRig:
    """Load the calibration rig plus its ``cpf:`` section (model, shared memory, bodies)."""

    rig = load_calibration_rig(path)
    raw = yaml.safe_load(Path(path).read_text())
    cpf = raw.get("cpf") if isinstance(raw, dict) else None
    if not isinstance(cpf, dict):
        raise CpfRigError(f"{path}: missing 'cpf' section (model_xml, hw_dir, shm_name, arms)")
    try:
        model_xml = Path(str(cpf["model_xml"])).expanduser()
        hw_dir = Path(str(cpf["hw_dir"])).expanduser()
        shm_name = str(cpf.get("shm_name", "/openarm_cpf"))
        arms: dict[str, CpfArm] = {}
        for name, entry in dict(cpf["arms"]).items():
            arms[str(name)] = CpfArm(
                profile=rig.arm(str(name)),
                shm_index=int(entry["shm_index"]),
                mujoco_body=str(entry["mujoco_body"]),
            )
    except (KeyError, TypeError, ValueError) as error:
        raise CpfRigError(f"{path}: bad cpf section: {error}") from error
    if not model_xml.is_file():
        raise CpfRigError(f"{path}: cpf.model_xml {model_xml} does not exist")
    if not (hw_dir / "tools" / "shm_layout.py").is_file():
        raise CpfRigError(f"{path}: cpf.hw_dir {hw_dir} has no tools/shm_layout.py")
    if not shm_name.startswith("/"):
        raise CpfRigError("cpf.shm_name must start with '/'")
    return CpfRig(rig=rig, model_xml=model_xml, hw_dir=hw_dir, shm_name=shm_name, arms=arms)


# ---------------------------------------------------------------------------------------
# Forward kinematics


class MujocoForwardKinematics:
    """``world_from_body`` for one arm from its joint angles, using CPF's MuJoCo model.

    The body must be rigidly attached after the arm's last joint (for the OpenArm v1
    model ``openarm_<side>_hand`` sits behind fixed links 7 and 8), which is checked by
    walking the kinematic chain: the joints between the world and the body must be
    exactly ``joint_names``.
    """

    def __init__(self, model_xml: Path, joint_names: tuple[str, ...], body_name: str) -> None:
        mujoco: Any = importlib.import_module("mujoco")
        self._mujoco = mujoco
        self._model = mujoco.MjModel.from_xml_path(str(model_xml))
        self._data = mujoco.MjData(self._model)
        model = self._model
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body < 0:
            raise CpfRigError(f"body {body_name!r} is not in {model_xml}")
        self.body_id = int(body)
        joint_ids = []
        for name in joint_names:
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint < 0:
                raise CpfRigError(f"joint {name!r} is not in {model_xml}")
            joint_ids.append(int(joint))
        chain: list[int] = []
        cursor = self.body_id
        while cursor != 0:
            start = int(model.body_jntadr[cursor])
            chain.extend(range(start, start + int(model.body_jntnum[cursor])))
            cursor = int(model.body_parentid[cursor])
        if set(chain) != set(joint_ids):
            chain_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in chain]
            raise CpfRigError(
                f"joints between world and {body_name!r} are {chain_names}, not {list(joint_names)}"
            )
        self.joint_names = tuple(joint_names)
        self._qpos_index = np.asarray([int(model.jnt_qposadr[j]) for j in joint_ids])
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "openarm_body_link0")
        if root >= 0:
            mujoco.mj_forward(model, self._data)
            if np.linalg.norm(self._data.xpos[root]) > 1e-9 or not np.allclose(
                self._data.xquat[root], (1.0, 0.0, 0.0, 0.0)
            ):
                raise CpfRigError("openarm_body_link0 is not at the MuJoCo world origin")

    def _set(self, q: NDArray[np.float64]) -> None:
        q = np.asarray(q, dtype=np.float64).reshape(-1)
        if q.shape[0] != self._qpos_index.shape[0]:
            raise ValueError(f"expected {self._qpos_index.shape[0]} joint angles, got {q.shape[0]}")
        self._data.qpos[:] = 0.0
        self._data.qpos[self._qpos_index] = q
        self._mujoco.mj_kinematics(self._model, self._data)

    def world_from_body(self, q: NDArray[np.float64]) -> Matrix:
        self._set(q)
        rotation = np.asarray(self._data.xmat[self.body_id], dtype=np.float64).reshape(3, 3)
        return make_transform(rotation, np.asarray(self._data.xpos[self.body_id]))

    def body_positions(self, q: NDArray[np.float64], names: tuple[str, ...]) -> NDArray[np.float64]:
        """World positions of several bodies at ``q`` (for drawing the arm skeleton)."""

        self._set(q)
        mujoco = self._mujoco
        ids = [mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names]
        if any(index < 0 for index in ids):
            missing = [name for name, index in zip(names, ids, strict=True) if index < 0]
            raise CpfRigError(f"bodies not in model: {missing}")
        return np.asarray(self._data.xpos[ids], dtype=np.float64)


def arm_skeleton_bodies(arm: str) -> tuple[str, ...]:
    """Body chain from the shoulder to the fingertip, for the projected-arm overlay."""

    return tuple(
        [f"openarm_{arm}_link{k}" for k in range(0, 8)]
        + [f"openarm_{arm}_hand", f"openarm_{arm}_hand_tcp"]
    )


# ---------------------------------------------------------------------------------------
# Shared-memory joint state


@dataclass(frozen=True, slots=True)
class ArmState:
    q: NDArray[np.float64]
    dq: NDArray[np.float64]
    hand: NDArray[np.float64]
    write_ns: int
    age_ms: float
    cycles: int
    fault: int
    seq: int

    @property
    def publishing(self) -> bool:
        return self.cycles > 0 and self.age_ms < 500.0


class SharedMemoryArms:
    """Read-only view of CPF's status block; never writes a command.

    The driver only opens the segment, so somebody must create it first. If it is
    missing this creates it (like ``cpf_node --create``), which is safe: an all-zero
    command block has ``write_ns == 0`` and is treated as stale by the driver.
    """

    def __init__(self, hw_dir: Path, name: str, *, create_if_missing: bool = True) -> None:
        tools = str(hw_dir / "tools")
        if tools not in sys.path:
            sys.path.insert(0, tools)
        layout: Any = importlib.import_module("shm_layout")  # verifies offsets via shm_probe
        self._layout = layout
        self.created = False
        try:
            self._planner = layout.ShmPlanner(name, create=False)
        except FileNotFoundError:
            if not create_if_missing:
                raise
            self._planner = layout.ShmPlanner(name, create=True)
            self.created = True
        self.name = name

    def read(self, arm_index: int) -> ArmState | None:
        snapshot = self._planner.status(arm_index)
        if snapshot is None:
            return None
        now_ns = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
        write_ns = int(snapshot["write_ns"])
        return ArmState(
            q=np.asarray(snapshot["q"], dtype=np.float64),
            dq=np.asarray(snapshot["dq"], dtype=np.float64),
            hand=np.asarray(snapshot["hand"], dtype=np.float64),
            write_ns=write_ns,
            age_ms=(now_ns - write_ns) / 1e6 if write_ns else math.inf,
            cycles=int(snapshot["cycles"]),
            fault=int(snapshot["fault"]),
            seq=int(snapshot["seq"]),
        )

    def why_not_publishing(self, state: ArmState | None, arm: str) -> str:
        if state is None:
            return f"{arm}: no consistent shm snapshot"
        if state.cycles == 0:
            return (
                f"{arm}: status block empty; start hold_pose_demo --arm {arm} ... --shm {self.name}"
            )
        if state.age_ms >= 500.0:
            return f"{arm}: status {state.age_ms:.0f} ms stale; driver exited?"
        return ""

    def close(self) -> None:
        self._planner.close()


# ---------------------------------------------------------------------------------------
# Result files


def camera_orientation_summary(world_from_camera: Matrix) -> dict[str, float]:
    """Heading, pitch-down and image roll of a camera pose in a z-up world (degrees)."""

    rotation = world_from_camera[:3, :3]
    optical = rotation @ np.array([0.0, 0.0, 1.0])
    right = rotation @ np.array([1.0, 0.0, 0.0])
    return {
        "heading_deg": math.degrees(math.atan2(float(optical[1]), float(optical[0]))),
        "camera_pitch_down_deg": math.degrees(math.asin(max(-1.0, min(1.0, -float(optical[2]))))),
        "camera_roll_deg": math.degrees(math.asin(max(-1.0, min(1.0, -float(right[2]))))),
    }


def save_hand_eye_result(path: Path, record: dict[str, Any], *, make_current: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    if make_current:
        CURRENT_CALIBRATION.parent.mkdir(parents=True, exist_ok=True)
        CURRENT_CALIBRATION.write_text(json.dumps(record, indent=2))
    return path


@dataclass(frozen=True, slots=True)
class FileExtrinsic:
    """A stored ``world_from_camera`` with the evidence it came with.

    Duck-type compatible with ``ManualExtrinsic`` for ``cpf_box_handoff``: it exposes
    ``world_from_camera``, ``summary()``, ``calibration`` and ``as_rigid_transform()``.
    """

    world_from_camera: Matrix
    record: dict[str, Any]
    path: Path

    @property
    def world_frame(self) -> str:
        return str(self.record["world_frame"])

    @property
    def camera_frame(self) -> str:
        return str(self.record["camera_frame"])

    @property
    def calibration(self) -> CalibrationRef:
        return CalibrationRef(
            calibration_id=f"{self.camera_frame}-to-{self.world_frame}/hand-eye",
            revision=str(self.record.get("measured_on", "unknown")),
        )

    def as_rigid_transform(self) -> RigidTransform3D:
        return rigid_transform_from_matrix(
            self.world_from_camera,
            source_frame=FrameId(self.camera_frame),
            target_frame=FrameId(self.world_frame),
        )

    def summary(self) -> dict[str, Any]:
        keys = (
            "schema",
            "method",
            "world_frame",
            "camera_frame",
            "camera_serial",
            "arm",
            "measured_on",
            "samples",
            "reprojection_rms_pixels",
            "validation",
            "world_from_camera",
        )
        summary = {key: self.record[key] for key in keys if key in self.record}
        summary["camera_position_world_metres"] = self.world_from_camera[:3, 3].tolist()
        summary.update(camera_orientation_summary(self.world_from_camera))
        summary["source_file"] = str(self.path)
        return summary


def load_extrinsic_file(path: str | Path) -> FileExtrinsic:
    path = Path(path)
    record = json.loads(path.read_text())
    if not isinstance(record, dict) or "world_from_camera" not in record:
        raise CalibrationProfileError(f"{path}: not an extrinsic record (no world_from_camera)")
    matrix = np.asarray(record["world_from_camera"], dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise CalibrationProfileError(f"{path}: world_from_camera must be a finite 4x4 matrix")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
        raise CalibrationProfileError(f"{path}: world_from_camera rotation is not orthonormal")
    for key in ("world_frame", "camera_frame"):
        if key not in record:
            raise CalibrationProfileError(f"{path}: missing {key}")
    return FileExtrinsic(world_from_camera=matrix, record=record, path=path)


def transform_difference(a: Matrix, b: Matrix) -> tuple[float, float]:
    """Translation (mm) and rotation (deg) between two poses of the same frame."""

    from .hand_eye import rotation_angle_deg

    delta = invert_transform(a) @ b
    return float(np.linalg.norm(delta[:3, 3]) * 1e3), rotation_angle_deg(delta[:3, :3])


__all__ = [
    "CALIBRATION_DIR",
    "CURRENT_CALIBRATION",
    "DEFAULT_RIG_PATH",
    "RESULT_SCHEMA",
    "ArmState",
    "CpfArm",
    "CpfRig",
    "CpfRigError",
    "FileExtrinsic",
    "MujocoForwardKinematics",
    "SharedMemoryArms",
    "arm_skeleton_bodies",
    "camera_orientation_summary",
    "load_cpf_rig",
    "load_extrinsic_file",
    "save_hand_eye_result",
    "transform_difference",
]
