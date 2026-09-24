from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from test_human_pose import OPTICAL, header

from vision_pipeline.geometry import PinholeIntrinsics
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.perception.pose.chains import LEFT_ARM
from vision_pipeline.perception.pose.yolo import YoloPoseEstimator
from vision_pipeline.runtime.human_pose_pipeline import HumanPoseLinkPipeline


class FakeTensor:
    def __init__(self, value: np.ndarray) -> None:
        self.value = value

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakePoseModel:
    def predict(self, image: np.ndarray, **options: object) -> list[object]:
        xy = np.zeros((1, 17, 2), dtype=np.float32)
        xy[0, 5] = (1, 1)
        xy[0, 7] = (2, 1)
        xy[0, 9] = (3, 1)
        conf = np.ones((1, 17), dtype=np.float32)
        return [
            SimpleNamespace(keypoints=SimpleNamespace(xy=FakeTensor(xy), conf=FakeTensor(conf)))
        ]


def test_backend_and_pipeline_select_metric_arm_links() -> None:
    image = ImageFrame(np.zeros((5, 5, 3), dtype=np.uint8), PixelFormat.BGR8)
    estimator = YoloPoseEstimator(model=FakePoseModel())
    pipeline = HumanPoseLinkPipeline(estimator, (LEFT_ARM,))
    result = pipeline.process(
        image,
        header(),
        depth_metres=np.ones((5, 5), dtype=np.float32),
        intrinsics=PinholeIntrinsics(5, 5, fx=2, fy=2, cx=2, cy=2),
        optical_frame=OPTICAL,
    )
    assert result.pose_2d is not None
    assert result.pose_3d is not None
    assert result.image_links["left_arm"] == (
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
    )
    assert result.metric_links["left_arm"] == result.image_links["left_arm"]
    assert result.pose_3d.get(LEFT_ARM.joints[-1]).position_metres == (0.5, -0.5, 1.0)
