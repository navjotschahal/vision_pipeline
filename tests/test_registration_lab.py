"""Offline capture decoding and posed-model frame checks."""

import json

import numpy as np
import pytest

from vision_pipeline.apps.registration_lab import load_session, model_cloud

o3d = pytest.importorskip("open3d")


def test_session_depth_is_metres_and_rgb_is_not_bgr(tmp_path):
    cv2 = pytest.importorskip("cv2")
    (tmp_path / "session.json").write_text(json.dumps({
        "camera": {"camera_matrix": [[2, 0, 1], [0, 2, 0], [0, 0, 1]]},
    }))
    (tmp_path / "view-00.json").write_text(json.dumps({"joints": {"right": {"j": 0.4}}}))
    np.save(tmp_path / "view-00.depth.npy", np.array([[0, 1, 2]], dtype=np.float32))
    cv2.imwrite(str(tmp_path / "view-00.color.png"), np.array([
        [[0, 0, 0], [10, 20, 30], [40, 50, 60]],
    ], dtype=np.uint8))
    cloud, _, record = load_session(tmp_path, "view-00", o3d)
    np.testing.assert_allclose(np.asarray(cloud.points), [[0, 0, 1], [1, 0, 2]])
    np.testing.assert_allclose(np.asarray(cloud.colors), np.array([[30, 20, 10],
                                                               [60, 50, 40]]) / 255)
    assert record["joints"]["right"]["j"] == 0.4


def test_model_surface_follows_recorded_joint_and_excludes_unknown_arm(tmp_path):
    pytest.importorskip("mujoco")
    xml = tmp_path / "model.xml"
    xml.write_text('''<mujoco><asset><mesh name="tetra"
        vertex="0 0 0  0.1 0 0  0 0.1 0  0 0 0.1"/></asset>
        <worldbody><body name="openarm_body_link0">
        <geom name="body_visual" type="mesh" mesh="tetra"/>
        <body name="openarm_right_link1" pos="1 0 0">
        <joint name="right_slide" type="slide" axis="1 0 0"/>
        <geom name="right_visual" type="mesh" mesh="tetra"/>
        </body></body></worldbody></mujoco>''')
    static = np.asarray(model_cloud(xml, {"joints": {}}, o3d).points)
    posed = np.asarray(model_cloud(xml, {"joints": {"right": {"right_slide": 0.5}}},
                                   o3d).points)
    assert static[:, 0].max() <= 0.101
    assert posed[:, 0].max() > 1.5
    assert posed[:, 0].max() <= 1.601
