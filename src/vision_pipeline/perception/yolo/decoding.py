"""Prediction decoding and non-maximum suppression for YOLO detectors."""

from __future__ import annotations

import torch

from vision_pipeline.perception.yolo import YoloError


def non_maximum_suppression(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Greedy NMS implemented with PyTorch tensor operations on the active device."""

    order = scores.argsort(descending=True)
    kept: list[torch.Tensor] = []
    while order.numel() > 0:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        current_box = boxes[current]
        other_boxes = boxes[remaining]
        intersection_min = torch.maximum(current_box[:2], other_boxes[:, :2])
        intersection_max = torch.minimum(current_box[2:], other_boxes[:, 2:])
        intersection_wh = (intersection_max - intersection_min).clamp(min=0)
        intersection = intersection_wh[:, 0] * intersection_wh[:, 1]
        current_area = (current_box[2] - current_box[0]) * (
            current_box[3] - current_box[1]
        )
        other_area = (other_boxes[:, 2] - other_boxes[:, 0]) * (
            other_boxes[:, 3] - other_boxes[:, 1]
        )
        union = current_area + other_area - intersection
        overlap = intersection / union.clamp(min=torch.finfo(boxes.dtype).eps)
        order = remaining[overlap <= iou_threshold]
    if not kept:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    return torch.stack(kept)


def decode_predictions(
    prediction: torch.Tensor,
    *,
    class_count: int,
    confidence: float,
    iou: float,
    max_detections: int,
) -> torch.Tensor:
    """Decode one YOLO BCN prediction into ``[x1,y1,x2,y2,score,class]`` rows."""

    if prediction.ndim != 3 or prediction.shape[0] != 1:
        raise YoloError(f"expected one BCN prediction tensor, got {tuple(prediction.shape)}")
    expected_channels = 4 + class_count
    if prediction.shape[1] != expected_channels:
        raise YoloError(
            f"expected {expected_channels} prediction channels (4 boxes + {class_count} classes), "
            f"got {prediction.shape[1]}"
        )

    rows = prediction[0].transpose(0, 1)
    scores, class_ids = rows[:, 4:].max(dim=1)
    selected = scores > confidence
    rows = rows[selected]
    scores = scores[selected]
    class_ids = class_ids[selected]
    if rows.shape[0] == 0:
        return torch.empty((0, 6), dtype=prediction.dtype, device=prediction.device)

    centers = rows[:, :2]
    half_sizes = rows[:, 2:4] / 2
    boxes = torch.cat((centers - half_sizes, centers + half_sizes), dim=1)

    # Class offsets prevent boxes from different classes suppressing one another.
    class_offset = 7680.0
    nms_boxes = boxes + class_ids.to(boxes.dtype).unsqueeze(1) * class_offset
    keep = non_maximum_suppression(nms_boxes, scores, iou)[:max_detections]
    return torch.cat(
        (boxes[keep], scores[keep, None], class_ids[keep, None].to(boxes.dtype)), dim=1
    )
