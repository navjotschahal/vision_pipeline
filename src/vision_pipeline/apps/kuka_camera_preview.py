"""OpenCV camera window in its own process beside MuJoCo's macOS viewer."""

from __future__ import annotations

from contextlib import suppress
from queue import Empty, Full
from typing import Any

import cv2
import numpy as np

from vision_pipeline.perception.pose import HumanJoint, HumanPose2D


def run_preview(frame_queue: Any, key_queue: Any) -> None:
    """Draw camera frames; forward c/h/q presses without touching MuJoCo state."""
    title = "KUKA teleop camera (c=clutch, h=hold, q=quit)"
    opened = False
    while True:
        try:
            message = frame_queue.get(timeout=0.05)
        except Empty:
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                with suppress(Full):
                    key_queue.put_nowait("q")
            continue
        if message is None:
            break
        image, pose, status, people, original_size = message
        frame = image.copy()
        if pose is not None:
            _draw_arm(frame, pose, original_size)
        cv2.putText(
            frame,
            f"{status} | people={people}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        cv2.imshow(title, frame)
        if not opened:
            print("KUKA camera preview open", flush=True)
            opened = True
        key = cv2.waitKey(1) & 0xFF
        action = {ord("c"): "c", ord("h"): "h", ord("q"): "q", 27: "q"}.get(key)
        if action is not None:
            with suppress(Full):
                key_queue.put_nowait(action)
    cv2.destroyAllWindows()


def _draw_arm(image: np.ndarray, pose: HumanPose2D, original_size: tuple[int, int]) -> None:
    width, height = original_size
    scale_x = image.shape[1] / width
    scale_y = image.shape[0] / height
    joints = (
        HumanJoint.RIGHT_SHOULDER,
        HumanJoint.RIGHT_ELBOW,
        HumanJoint.RIGHT_WRIST,
    )
    points = [pose.get(joint) for joint in joints]
    for first, second in zip(points[:-1], points[1:], strict=True):
        if first is None or second is None or min(first.confidence, second.confidence) < 0.6:
            continue
        start = (round(first.image_xy[0] * scale_x), round(first.image_xy[1] * scale_y))
        end = (round(second.image_xy[0] * scale_x), round(second.image_xy[1] * scale_y))
        cv2.line(image, start, end, (0, 255, 255), 3)
        cv2.circle(image, end, 5, (0, 255, 0), -1)


__all__ = ["run_preview"]
