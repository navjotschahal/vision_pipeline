from __future__ import annotations

import uuid

import numpy as np
import pytest

from vision_pipeline.runtime.box_view_channel import (
    CATEGORY_CLUSTER,
    CATEGORY_PLANE,
    BoxViewGeometry,
    BoxViewReader,
    BoxViewWriter,
)

WIDTH, HEIGHT, MAX_POINTS = 8, 6, 20


def _channel_name() -> str:
    return f"vision-pipeline-test-boxview-{uuid.uuid4().hex}"


def test_reader_decodes_exactly_what_the_writer_published() -> None:
    name = _channel_name()
    geometry = BoxViewGeometry(width=WIDTH, height=HEIGHT, max_points=MAX_POINTS)
    writer = BoxViewWriter(geometry, name=name)
    try:
        reader = BoxViewReader(geometry, name=name)
        try:
            color = np.full((HEIGHT, WIDTH, 3), 100, dtype=np.uint8)
            xyz = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float32)
            rgb = np.array([[10, 20, 30]] * 3, dtype=np.uint8)
            category = np.array(
                [CATEGORY_PLANE, CATEGORY_CLUSTER, CATEGORY_CLUSTER], dtype=np.uint8
            )

            writer.publish(
                frame_index=7,
                captured_at_ns=123_456_789,
                host_written_at_ns=123_456_999,
                color_bgr=color,
                points_xyz=xyz,
                points_rgb=rgb,
                points_category=category,
                has_observation=True,
                box_position_metres=(0.05, 0.05, 1.0),
                box_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
                box_size_metres=(0.1, 0.1, 0.02),
                cluster_point_count=2,
                plane_normal=(0.0, 0.0, -1.0),
                plane_offset_metres=1.0,
                estimator_frames_submitted=42,
                estimator_frames_failed=1,
            )

            frame = reader.read()
            assert frame is not None
            assert frame.frame_index == 7
            assert frame.captured_at_ns == 123_456_789
            assert frame.has_observation is True
            assert np.array_equal(frame.color_bgr, color)
            assert np.allclose(frame.points_xyz, xyz)
            assert np.array_equal(frame.points_rgb, rgb)
            assert np.array_equal(frame.points_category, category)
            assert frame.box_position_metres == pytest.approx((0.05, 0.05, 1.0))
            assert frame.box_size_metres == pytest.approx((0.1, 0.1, 0.02))
            assert frame.cluster_point_count == 2
            assert frame.plane_normal == pytest.approx((0.0, 0.0, -1.0))
            assert frame.estimator_frames_submitted == 42
            assert frame.estimator_frames_failed == 1
        finally:
            reader.close()
    finally:
        writer.close()


def test_publish_rejects_more_points_than_the_channel_capacity() -> None:
    name = _channel_name()
    geometry = BoxViewGeometry(width=WIDTH, height=HEIGHT, max_points=2)
    writer = BoxViewWriter(geometry, name=name)
    try:
        color = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        xyz = np.zeros((3, 3), dtype=np.float32)
        rgb = np.zeros((3, 3), dtype=np.uint8)
        category = np.zeros(3, dtype=np.uint8)

        with pytest.raises(ValueError, match="capacity"):
            writer.publish(
                frame_index=0,
                captured_at_ns=0,
                host_written_at_ns=0,
                color_bgr=color,
                points_xyz=xyz,
                points_rgb=rgb,
                points_category=category,
                has_observation=False,
            )
    finally:
        writer.close()


def test_a_shrinking_point_count_does_not_leak_the_previous_frames_tail() -> None:
    name = _channel_name()
    geometry = BoxViewGeometry(width=WIDTH, height=HEIGHT, max_points=MAX_POINTS)
    writer = BoxViewWriter(geometry, name=name)
    try:
        reader = BoxViewReader(geometry, name=name)
        try:
            color = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
            big_xyz = np.ones((5, 3), dtype=np.float32)
            writer.publish(
                frame_index=1,
                captured_at_ns=0,
                host_written_at_ns=0,
                color_bgr=color,
                points_xyz=big_xyz,
                points_rgb=np.zeros((5, 3), dtype=np.uint8),
                points_category=np.zeros(5, dtype=np.uint8),
                has_observation=False,
            )
            small_xyz = np.full((2, 3), 2.0, dtype=np.float32)
            writer.publish(
                frame_index=2,
                captured_at_ns=0,
                host_written_at_ns=0,
                color_bgr=color,
                points_xyz=small_xyz,
                points_rgb=np.zeros((2, 3), dtype=np.uint8),
                points_category=np.zeros(2, dtype=np.uint8),
                has_observation=False,
            )

            frame = reader.read()
            assert frame is not None
            assert frame.points_xyz.shape == (2, 3)
            assert np.allclose(frame.points_xyz, 2.0)
        finally:
            reader.close()
    finally:
        writer.close()
