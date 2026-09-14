"""A transparent temporal baseline using DINO appearance and bounding-box IoU."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from vision_pipeline.perception.contracts import BoundingBox2D
from vision_pipeline.perception.regions import RegionDescriptorBatch
from vision_pipeline.perception.tracking.config import AppearanceTrackerConfig, TrackingError
from vision_pipeline.perception.tracking.contracts import TrackedDetection2D, TrackingBatch


@dataclass(slots=True)
class _TrackState:
    track_id: int
    class_id: int
    box: BoundingBox2D
    descriptor: torch.Tensor
    age_frames: int = 1
    hits: int = 1
    missed_frames: int = 0


def _box_tensor(boxes: list[BoundingBox2D], reference: torch.Tensor) -> torch.Tensor:
    return reference.new_tensor(
        [[box.x_min, box.y_min, box.x_max, box.y_max] for box in boxes]
    )


def _pairwise_iou(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(left[:, None, :2], right[None, :, :2])
    bottom_right = torch.minimum(left[:, None, 2:], right[None, :, 2:])
    intersection_size = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    left_area = (left[:, 2] - left[:, 0]) * (left[:, 3] - left[:, 1])
    right_area = (right[:, 2] - right[:, 0]) * (right[:, 3] - right[:, 1])
    union = left_area[:, None] + right_area[None, :] - intersection
    return intersection / union.clamp(min=torch.finfo(union.dtype).eps)


class GreedyAppearanceTracker:
    """Assign persistent IDs with a scored, one-to-one greedy association baseline.

    This intentionally exposes a simple algorithm before introducing motion filters,
    Hungarian assignment, or established trackers such as ByteTrack.
    """

    def __init__(self, config: AppearanceTrackerConfig | None = None) -> None:
        self.config = config or AppearanceTrackerConfig()
        self._tracks: dict[int, _TrackState] = {}
        self._next_track_id = 1
        self._source_id: str | None = None
        self._last_sequence: int | None = None

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._source_id = None
        self._last_sequence = None

    def update(
        self, regions: RegionDescriptorBatch[torch.Tensor]
    ) -> TrackingBatch[torch.Tensor]:
        source = regions.detections.source
        self._validate_sequence(source.source_id, source.sequence_number)
        descriptors = regions.descriptors
        detection_count = len(regions.detections.detections)
        if descriptors.ndim != 2 or descriptors.shape[0] != detection_count:
            raise TrackingError("descriptors must have [detections, embedding] shape")

        for track in self._tracks.values():
            track.age_frames += 1

        track_ids = sorted(self._tracks)
        matches: dict[int, tuple[int, float | None]] = {}
        matched_track_ids: set[int] = set()
        if track_ids and detection_count:
            matches = self._associate(track_ids, regions)
            matched_track_ids = {track_id for track_id, _score in matches.values()}

        for detection_index, (track_id, _score) in matches.items():
            state = self._tracks[track_id]
            detection = regions.detections.detections[detection_index]
            descriptor = descriptors[detection_index]
            mixed = (
                self.config.descriptor_momentum * state.descriptor
                + (1 - self.config.descriptor_momentum) * descriptor
            )
            state.descriptor = functional.normalize(mixed, dim=0)
            state.box = detection.box
            state.class_id = detection.class_id
            state.hits += 1
            state.missed_frames = 0

        for track_id in track_ids:
            if track_id not in matched_track_ids:
                self._tracks[track_id].missed_frames += 1
        self._tracks = {
            track_id: state
            for track_id, state in self._tracks.items()
            if state.missed_frames <= self.config.max_missed_frames
        }

        for detection_index in range(detection_count):
            if detection_index in matches:
                continue
            detection = regions.detections.detections[detection_index]
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _TrackState(
                track_id=track_id,
                class_id=detection.class_id,
                box=detection.box,
                descriptor=descriptors[detection_index].detach().clone(),
            )
            matches[detection_index] = (track_id, None)

        tracked = tuple(
            TrackedDetection2D(
                detection_index=detection_index,
                track_id=matches[detection_index][0],
                age_frames=self._tracks[matches[detection_index][0]].age_frames,
                hits=self._tracks[matches[detection_index][0]].hits,
                association_score=matches[detection_index][1],
            )
            for detection_index in range(detection_count)
        )
        self._source_id = source.source_id
        self._last_sequence = source.sequence_number
        return TrackingBatch(
            regions=regions,
            tracked_detections=tracked,
            active_track_count=len(self._tracks),
            association="greedy(dino-cosine+iou)",
        )

    def _validate_sequence(self, source_id: str, sequence_number: int) -> None:
        if self._source_id is not None and source_id != self._source_id:
            raise TrackingError(
                f"tracker source changed from {self._source_id!r} to {source_id!r}; call reset()"
            )
        if self._last_sequence is not None and sequence_number <= self._last_sequence:
            raise TrackingError(
                f"sequence must increase beyond {self._last_sequence}; got {sequence_number}"
            )

    def _associate(
        self,
        track_ids: list[int],
        regions: RegionDescriptorBatch[torch.Tensor],
    ) -> dict[int, tuple[int, float | None]]:
        descriptors = functional.normalize(regions.descriptors, dim=1)
        track_descriptors = torch.stack(
            [self._tracks[track_id].descriptor.to(descriptors.device) for track_id in track_ids]
        )
        cosine = functional.normalize(track_descriptors, dim=1) @ descriptors.T

        detections = regions.detections.detections
        track_boxes = _box_tensor(
            [self._tracks[track_id].box for track_id in track_ids], descriptors
        )
        detection_boxes = _box_tensor([item.box for item in detections], descriptors)
        iou = _pairwise_iou(track_boxes, detection_boxes)
        score = (
            self.config.appearance_weight * cosine.clamp(min=0, max=1)
            + (1 - self.config.appearance_weight) * iou
        )
        gate = (cosine >= self.config.minimum_cosine_similarity) | (
            iou >= self.config.minimum_iou
        )
        gate &= score >= self.config.minimum_score
        if self.config.class_aware:
            track_classes = torch.tensor(
                [self._tracks[track_id].class_id for track_id in track_ids],
                device=descriptors.device,
            )
            detection_classes = torch.tensor(
                [item.class_id for item in detections],
                device=descriptors.device,
            )
            gate &= track_classes[:, None] == detection_classes[None, :]

        score_rows = score.masked_fill(~gate, -1).detach().to(device="cpu").tolist()
        candidates = sorted(
            (
                (float(candidate_score), track_index, detection_index)
                for track_index, row in enumerate(score_rows)
                for detection_index, candidate_score in enumerate(row)
                if candidate_score >= 0
            ),
            reverse=True,
        )
        matches: dict[int, tuple[int, float | None]] = {}
        used_tracks: set[int] = set()
        for candidate_score, track_index, detection_index in candidates:
            track_id = track_ids[track_index]
            if track_id in used_tracks or detection_index in matches:
                continue
            matches[detection_index] = (track_id, candidate_score)
            used_tracks.add(track_id)
        return matches


__all__ = ["GreedyAppearanceTracker"]
