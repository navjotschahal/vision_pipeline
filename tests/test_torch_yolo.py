from __future__ import annotations

import numpy as np
import pytest
import torch

from vision_pipeline.perception.yolo import YoloError
from vision_pipeline.perception.yolo.decoding import decode_predictions
from vision_pipeline.perception.yolo.preprocessing import letterbox_bgr, preprocess_bgr


def test_letterbox_uses_minimal_stride_aligned_shape() -> None:
    image = np.zeros((360, 640, 3), dtype=np.uint8)

    padded, transform = letterbox_bgr(image, image_size=640, stride=32)

    assert padded.shape == (384, 640, 3)
    assert transform.scale == 1
    assert transform.pad_top == 12
    assert transform.pad_left == 0


def test_preprocess_converts_bgr_hwc_uint8_to_rgb_nchw_float() -> None:
    image = np.array([[[10, 20, 30]]], dtype=np.uint8)

    tensor, transform = preprocess_bgr(image, image_size=1, stride=1, device=torch.device("cpu"))

    assert tensor.shape == (1, 3, 1, 1)
    assert tensor.dtype is torch.float32
    assert tensor[0, :, 0, 0].tolist() == pytest.approx([30 / 255, 20 / 255, 10 / 255])
    assert transform.scale == 1


def test_decoder_suppresses_same_class_overlap_but_keeps_other_class() -> None:
    # Channels are cx, cy, width, height, class-0 score, class-1 score.
    prediction = torch.tensor(
        [
            [
                [10.0, 11.0, 10.0],
                [10.0, 11.0, 10.0],
                [10.0, 10.0, 10.0],
                [10.0, 10.0, 10.0],
                [0.90, 0.80, 0.10],
                [0.10, 0.10, 0.85],
            ]
        ]
    )

    decoded = decode_predictions(
        prediction,
        class_count=2,
        confidence=0.25,
        iou=0.5,
        max_detections=10,
    )

    assert decoded.shape == (2, 6)
    assert decoded[:, 4].tolist() == pytest.approx([0.90, 0.85])
    assert decoded[:, 5].tolist() == [0.0, 1.0]
    assert decoded[0, :4].tolist() == pytest.approx([5.0, 5.0, 15.0, 15.0])


def test_decoder_rejects_unknown_output_layout() -> None:
    with pytest.raises(YoloError, match="prediction channels"):
        decode_predictions(
            torch.zeros((1, 9, 10)),
            class_count=2,
            confidence=0.25,
            iou=0.5,
            max_detections=10,
        )
