from __future__ import annotations

# ruff: noqa: E402
from dataclasses import replace

import numpy as np
import pytest
import yaml

mujoco = pytest.importorskip("mujoco")

from test_human_pose import CLOCK, CPU, header

from vision_pipeline.apps.kuka_webcam_teleop import PositionIK, default_poc_root
from vision_pipeline.contracts import FrameId, TimePoint
from vision_pipeline.geometry.spatial import Pose3D
from vision_pipeline.perception.pose import HumanJoint, HumanKeypoint2D, HumanPose2D
from vision_pipeline.teleoperation import TeleopState
from vision_pipeline.teleoperation.webcam_planar import WebcamPlanarRetargeter


def pose(sequence: int, received_ns: int, wrist_x: float, wrist_y: float = 240) -> HumanPose2D:
    source = replace(header(sequence), received_at=TimePoint(received_ns, CLOCK))
    return HumanPose2D(
        source_header=source,
        person_id="one-operator",
        landmarks=(
            HumanKeypoint2D(HumanJoint.RIGHT_SHOULDER, (320, 180), 0.9),
            HumanKeypoint2D(HumanJoint.RIGHT_ELBOW, (350, 210), 0.9),
            HumanKeypoint2D(HumanJoint.RIGHT_WRIST, (wrist_x, wrist_y), 0.9),
        ),
        producer="test-pose",
        produced_on=CPU,
    )


def test_planar_mapping_moves_two_axes_and_holds_on_loss() -> None:
    subject = WebcamPlanarRetargeter()
    anchor = Pose3D(FrameId("kuka/base"), (0.66, 0.0, 0.30), (0, 0, 0, 1))
    subject.engage(
        pose(0, 1_000_000_000, 380),
        image_size=(640, 480),
        robot_pose=anchor,
        now=TimePoint(1_000_000_000, CLOCK),
    )
    target = subject.update(
        pose(1, 1_100_000_000, 444), now=TimePoint(1_100_000_000, CLOCK), deadman_pressed=True
    )
    assert target.state is TeleopState.ACTIVE
    assert target.arm is not None
    assert target.arm.pose.position_metres == pytest.approx((0.66, 0.025, 0.30))
    assert target.arm.limited
    held = subject.update(None, now=TimePoint(1_200_000_000, CLOCK), deadman_pressed=True)
    assert held.state is TeleopState.HOLD
    assert held.reason == "operator-not-detected"


def test_poc_kuka_ik_and_impedance_reach_small_cartesian_target() -> None:
    poc = default_poc_root()
    if not (poc / "src/sim/assets/scene_sim_kuka_iiwa_14.xml").exists():
        pytest.skip("CPF PoC checkout is unavailable")
    import sys

    sys.path.insert(0, str(poc / "src"))
    from controllers import JointImpedanceController, MujocoRobotView

    config = yaml.safe_load((poc / "src/sim/configs/robots/kuka_iiwa_14.yml").read_text())
    model = mujoco.MjModel.from_xml_path(str(poc / "src/sim/assets/scene_sim_kuka_iiwa_14.xml"))
    data = mujoco.MjData(model)
    robot = MujocoRobotView.from_config(model, data, config)
    controller = JointImpedanceController(robot, config["controller"])
    data.qpos[robot.qpos_idx] = config["simulation"]["q_home"]
    mujoco.mj_forward(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    start = data.site_xpos[site].copy()
    mapper = WebcamPlanarRetargeter()
    mapper.engage(
        pose(0, 1_000_000_000, 380),
        image_size=(640, 480),
        robot_pose=Pose3D(FrameId("kuka/base"), tuple(start), (0, 0, 0, 1)),
        now=TimePoint(1_000_000_000, CLOCK),
    )
    command = mapper.update(
        pose(1, 1_200_000_000, 444, 192),
        now=TimePoint(1_200_000_000, CLOCK),
        deadman_pressed=True,
    )
    assert command.arm is not None
    desired = np.asarray(command.arm.pose.position_metres)
    assert desired == pytest.approx(start + np.array([0.0, 0.05, 0.05]) / np.sqrt(2))
    assert command.arm.limited
    q_target, residual = PositionIK(model, site, robot.qpos_idx, robot.dof_indices).solve(
        robot.q, desired
    )
    assert residual < 0.01
    for _ in range(1000):
        controller.apply(controller.compute_torque(q_desired=q_target))
        mujoco.mj_step(model, data)
    assert np.linalg.norm(data.site_xpos[site] - desired) < 0.02
