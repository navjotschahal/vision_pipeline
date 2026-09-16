from __future__ import annotations

import math

import numpy as np
import pytest

from vision_pipeline.apps.select_object import box_axes
from vision_pipeline.contracts import FrameId
from vision_pipeline.geometry.spatial import OrientedBox3D, Pose3D


def test_box_axes_follow_the_box_rotation_from_its_centre() -> None:
    # 90 degrees about camera z: box x points along camera +y, box y along camera -x.
    half = math.sqrt(0.5)
    bounds = OrientedBox3D(
        pose=Pose3D(FrameId("camera"), (0.1, -0.2, 0.9), (0.0, 0.0, half, half)),
        size_metres=(0.30, 0.10, 0.02),
    )

    origin, endpoints = box_axes(bounds)

    length = 0.6 * 0.30
    assert origin == pytest.approx((0.1, -0.2, 0.9))
    assert endpoints[0] - origin == pytest.approx((0.0, length, 0.0), abs=1e-9)
    assert endpoints[1] - origin == pytest.approx((-length, 0.0, 0.0), abs=1e-9)
    assert endpoints[2] - origin == pytest.approx((0.0, 0.0, length), abs=1e-9)
    directions = (endpoints - origin) / length
    assert directions @ directions.T == pytest.approx(np.eye(3), abs=1e-9)


def test_box_axes_length_is_clamped_for_tiny_and_huge_boxes() -> None:
    identity = (0.0, 0.0, 0.0, 1.0)
    tiny = OrientedBox3D(Pose3D(FrameId("camera"), (0, 0, 1), identity), (0.01, 0.01, 0.01))
    huge = OrientedBox3D(Pose3D(FrameId("camera"), (0, 0, 1), identity), (2.0, 1.0, 1.0))

    assert np.linalg.norm(box_axes(tiny)[1][0] - (0, 0, 1)) == pytest.approx(0.05)
    assert np.linalg.norm(box_axes(huge)[1][0] - (0, 0, 1)) == pytest.approx(0.25)
