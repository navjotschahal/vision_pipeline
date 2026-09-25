"""Surface samples of the OpenArm MuJoCo model, posed by joint state, for depth registration.

The visual meshes of the model are sampled once, area-weighted, in each geom's own frame.
For a joint configuration ``mj_kinematics`` places every geom, and the samples are carried
into the world frame (``openarm_body_link0``). Registering a depth cloud against these
samples with :mod:`cloud_registration` gives the camera pose without any marker on the
robot. Fingers are excluded because the gripper joints are not published over shared
memory; the static body and the two arms are separate groups so an arm whose driver is
not running can be left out.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .openarm_cpf import CpfRigError

GROUPS = ("body", "right", "left")


@dataclass(frozen=True, slots=True)
class SurfaceSamples:
    """World-frame surface points with unit normals and the geom each came from."""

    points: NDArray[np.float64]
    normals: NDArray[np.float64]
    geom_ids: NDArray[np.int32]

    @property
    def count(self) -> int:
        return int(self.points.shape[0])


def _group_of(body_name: str) -> str | None:
    """``body`` is everything that never moves: the torso and both shoulder mounts (link0)."""

    if body_name.startswith("openarm_body") or body_name.endswith("_link0"):
        return "body"
    if body_name.startswith("openarm_right"):
        return "right"
    if body_name.startswith("openarm_left"):
        return "left"
    return None


class RobotSurfaceModel:
    """Area-weighted samples of the model's visual meshes, posed by ``mj_kinematics``."""

    def __init__(
        self,
        model_xml: Path,
        *,
        density_per_m2: float = 150_000.0,
        min_points_per_geom: int = 30,
        max_points_per_geom: int = 15_000,
        seed: int = 0,
    ) -> None:
        mujoco: Any = importlib.import_module("mujoco")
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.data = mujoco.MjData(self.model)
        model = self.model
        rng = np.random.default_rng(seed)
        self._local: dict[int, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
        self.groups: dict[str, list[int]] = {group: [] for group in GROUPS}
        self.geom_names: dict[int, str] = {}
        for geom in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
            if "_visual" not in name or int(model.geom_type[geom]) != int(
                mujoco.mjtGeom.mjGEOM_MESH
            ):
                continue
            if "finger" in name:
                continue
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom])
            )
            group = _group_of(str(body_name))
            if group is None:
                continue
            samples = self._sample_mesh(
                int(model.geom_dataid[geom]),
                rng,
                density_per_m2,
                min_points_per_geom,
                max_points_per_geom,
            )
            if samples is None:
                continue
            self._local[geom] = samples
            self.groups[group].append(geom)
            self.geom_names[geom] = name
        if not self.groups["body"]:
            raise CpfRigError(f"{model_xml} has no openarm_body visual meshes")
        self._joint_qpos = {
            str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)): int(
                model.jnt_qposadr[joint]
            )
            for joint in range(model.njnt)
        }

    def _sample_mesh(
        self, mesh_id: int, rng: np.random.Generator, density: float, minimum: int, maximum: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | None:
        model = self.model
        vert_start, vert_count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
        face_start, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
        vertices = np.asarray(
            model.mesh_vert[vert_start : vert_start + vert_count], dtype=np.float64
        )
        faces = np.asarray(model.mesh_face[face_start : face_start + face_count], dtype=np.int64)
        triangles = vertices[faces]
        edge_a = triangles[:, 1] - triangles[:, 0]
        edge_b = triangles[:, 2] - triangles[:, 0]
        cross = np.cross(edge_a, edge_b)
        double_area = np.linalg.norm(cross, axis=1)
        keep = double_area > 1e-14
        if not np.any(keep):
            return None
        triangles, edge_a, edge_b, cross, double_area = (
            triangles[keep],
            edge_a[keep],
            edge_b[keep],
            cross[keep],
            double_area[keep],
        )
        area = 0.5 * double_area
        total_area = float(area.sum())
        count = int(min(max(round(total_area * density), minimum), maximum))
        chosen = rng.choice(area.shape[0], size=count, p=area / total_area)
        u = rng.random(count)
        v = rng.random(count)
        flip = u + v > 1.0
        u[flip], v[flip] = 1.0 - u[flip], 1.0 - v[flip]
        points = triangles[chosen, 0] + u[:, None] * edge_a[chosen] + v[:, None] * edge_b[chosen]
        normals = cross[chosen] / double_area[chosen][:, None]
        return points, normals

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self._joint_qpos)

    def set_joints(self, joints: Mapping[str, float]) -> None:
        """Pose the model with the given joint angles; unspecified joints are zero."""

        self.data.qpos[:] = 0.0
        for name, value in joints.items():
            try:
                self.data.qpos[self._joint_qpos[name]] = float(value)
            except KeyError as error:
                raise CpfRigError(f"joint {name!r} is not in the model") from error
        self._mujoco.mj_kinematics(self.model, self.data)

    def surface(self, joints: Mapping[str, float], groups: Sequence[str]) -> SurfaceSamples:
        """World-frame samples of the requested groups at this joint configuration."""

        unknown = [group for group in groups if group not in self.groups]
        if unknown:
            raise ValueError(f"unknown groups {unknown}; choose from {GROUPS}")
        self.set_joints(joints)
        points: list[NDArray[np.float64]] = []
        normals: list[NDArray[np.float64]] = []
        ids: list[NDArray[np.int32]] = []
        for group in groups:
            for geom in self.groups[group]:
                local_points, local_normals = self._local[geom]
                rotation = np.asarray(self.data.geom_xmat[geom], dtype=np.float64).reshape(3, 3)
                translation = np.asarray(self.data.geom_xpos[geom], dtype=np.float64)
                points.append(local_points @ rotation.T + translation)
                normals.append(local_normals @ rotation.T)
                ids.append(np.full(local_points.shape[0], geom, dtype=np.int32))
        if not points:
            raise ValueError("no geoms selected")
        return SurfaceSamples(np.concatenate(points), np.concatenate(normals), np.concatenate(ids))

    def body_position(self, joints: Mapping[str, float], body_name: str) -> NDArray[np.float64]:
        self.set_joints(joints)
        mujoco = self._mujoco
        body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body < 0:
            raise CpfRigError(f"body {body_name!r} is not in the model")
        return np.asarray(self.data.xpos[body], dtype=np.float64).copy()


__all__ = ["GROUPS", "RobotSurfaceModel", "SurfaceSamples"]
