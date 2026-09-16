"""Deterministic tracking-quality evaluation for promptable video segmenters.

A static recording has no motion to track and no ground truth. This module derives a
scripted sequence from real recorded RGB-D frames whose per-frame reference mask is
known exactly:

1. the backend is seeded with one click on the untouched first frame;
2. the seeded object is removed from each later frame by inpainting its (dilated) mask,
   and its real pixels are pasted back at a scripted offset;
3. phases move the object, sweep a textured occluder over it and hold it fully covered,
   move it out of the image and back, and paste an identical copy of it elsewhere as a
   distractor while the original is away.

The reference is the seed mask moved by the same offset, minus the occluder, the
distractor, and the image border. Because the reference comes from each backend's own
seed, part-versus-whole ambiguity of a single click does not bias the tracking scores;
seed quality is judged separately (visually and by cross-backend agreement).

Scores come from the raw backend (no resets). :func:`emulate_session` then replays the
:class:`~.selection.SelectionPolicy` over those outputs, which is exact up to the first
``LOST`` because a session only resets its backend on loss.

This is a synthetic stress test built from real imagery, not a substitute for recorded
object motion.
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import SensorSample
from vision_pipeline.image import ImageFrame
from vision_pipeline.rgbd import AlignedRgbdFrame, DepthFrame

from .selection import Click, LossReason, PromptableVideoSegmenter, SelectionPolicy

_PRESENT_FRACTION = 0.5
_HIDDEN_FRACTION = 0.05


class Phase(StrEnum):
    STATIC = "static"
    MOTION = "motion"
    OCCLUSION = "occlusion"
    EXIT = "exit"
    RETURN = "return"


@dataclass(frozen=True, slots=True)
class ScriptedFrame:
    """Where the object, occluder, and distractor are in one generated frame."""

    index: int
    phase: Phase
    offset: tuple[int, int]
    occluder: tuple[int, int, int, int] | None
    distractor_offset: tuple[int, int] | None


def _box(mask: NDArray[np.bool_]) -> tuple[int, int, int, int]:
    rows = np.flatnonzero(mask.any(axis=1))
    columns = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:
        raise ValueError("seed mask is empty")
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def build_script(
    seed_mask: NDArray[np.bool_], *, motion_pixels: int = 40
) -> tuple[ScriptedFrame, ...]:
    """A 150-frame schedule adapted to the seeded object's size and image position."""

    height, width = seed_mask.shape
    x0, y0, x1, y1 = _box(seed_mask)
    box_width, box_height = x1 - x0, y1 - y0
    # Leave through the nearer vertical edge; the distractor goes on the other side.
    leave_left = (x0 + x1) / 2 < width / 2
    exit_dx = -(x1 + 4) if leave_left else width - x0 + 4
    gap = max(24, box_width // 4)
    distractor_dx = x1 - x0 + gap if leave_left else -(box_width + gap)
    distractor: tuple[int, int] | None = (distractor_dx, 0)
    if not (x0 + distractor_dx >= 0 and x1 + distractor_dx <= width):
        distractor = None
    occluder_width = int(box_width * 1.3) + 8
    occluder_height = box_height + 16
    occluder_top = max(0, y0 - 8)
    sweep_start = x0 - occluder_width
    covered = x0 - (occluder_width - box_width) // 2
    sweep_end = x1

    frames: list[ScriptedFrame] = []

    def add(phase: Phase, offset: tuple[int, int], **extra: object) -> None:
        frames.append(
            ScriptedFrame(
                len(frames),
                phase,
                offset,
                cast(tuple[int, int, int, int] | None, extra.get("occluder")),
                cast(tuple[int, int] | None, extra.get("distractor")),
            )
        )

    for _ in range(30):
        add(Phase.STATIC, (0, 0))
    for step in range(40):
        angle = 2 * math.pi * step / 40
        add(
            Phase.MOTION,
            (
                round(motion_pixels * math.sin(angle)),
                round(motion_pixels / 2 * math.sin(2 * angle)),
            ),
        )

    def occluder_at(left: float) -> tuple[int, int, int, int]:
        left_px = round(left)
        return (left_px, occluder_top, left_px + occluder_width, occluder_top + occluder_height)

    for step in range(12):
        add(
            Phase.OCCLUSION,
            (0, 0),
            occluder=occluder_at(np.interp(step, [0, 11], [sweep_start, covered])),
        )
    for _ in range(8):
        add(Phase.OCCLUSION, (0, 0), occluder=occluder_at(covered))
    for step in range(10):
        add(
            Phase.OCCLUSION,
            (0, 0),
            occluder=occluder_at(np.interp(step, [0, 9], [covered, sweep_end])),
        )
    for step in range(12):
        add(Phase.EXIT, (round(np.interp(step, [0, 11], [0, exit_dx])), 0), distractor=distractor)
    for _ in range(16):
        add(Phase.EXIT, (exit_dx, 0), distractor=distractor)
    for step in range(12):
        add(Phase.RETURN, (round(np.interp(step, [0, 11], [exit_dx, 0])), 0), distractor=distractor)
    for _ in range(10):
        add(Phase.RETURN, (0, 0), distractor=distractor)
    return tuple(frames)


def _shift(image: NDArray[np.generic], dx: int, dy: int) -> NDArray[np.generic]:
    shifted = np.zeros_like(image)
    height, width = image.shape[:2]
    src_x, dst_x = (
        (slice(0, width - dx), slice(dx, width))
        if dx >= 0
        else (slice(-dx, width), slice(0, width + dx))
    )
    src_y, dst_y = (
        (slice(0, height - dy), slice(dy, height))
        if dy >= 0
        else (slice(-dy, height), slice(0, height + dy))
    )
    if abs(dx) < width and abs(dy) < height:
        shifted[dst_y, dst_x] = image[src_y, src_x]
    return shifted


@dataclass(frozen=True, slots=True)
class GeneratedFrame:
    frame: AlignedRgbdFrame
    script: ScriptedFrame
    reference: NDArray[np.bool_]
    distractor: NDArray[np.bool_]


class PerturbationSequence:
    """Generates scripted frames from recorded frames and one seed mask."""

    def __init__(
        self,
        base_frames: Sequence[AlignedRgbdFrame],
        seed_mask: NDArray[np.bool_],
        *,
        motion_pixels: int = 40,
    ) -> None:
        if not base_frames:
            raise ValueError("at least one base frame is required")
        self._frames = base_frames
        self._seed = seed_mask
        self.script = build_script(seed_mask, motion_pixels=motion_pixels)
        self._hole = cv2.dilate(seed_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        first = base_frames[0]
        self._background = cv2.inpaint(
            first.color.payload.data, self._hole.astype(np.uint8), 5, cv2.INPAINT_TELEA
        )
        # Real scene texture for the occluder, taken from the object-free background half
        # an image away so the occluder never shows a copy of the tracked object.
        self._occluder_texture = np.ascontiguousarray(
            np.roll(self._background, seed_mask.shape[1] // 2, axis=1)
        )

    def __len__(self) -> int:
        return len(self.script)

    def generate(self, index: int) -> GeneratedFrame:
        script = self.script[index]
        base = self._frames[index % len(self._frames)]
        color = base.color.payload.data
        depth = base.depth.payload.data
        seed = self._seed
        if index == 0:
            return GeneratedFrame(base, script, seed.copy(), np.zeros_like(seed))

        out_color = color.copy()
        out_depth = depth.copy()
        out_color[self._hole] = self._background[self._hole]
        out_depth[self._hole] = 0.0

        def paste(dx: int, dy: int) -> NDArray[np.bool_]:
            moved = cast(NDArray[np.bool_], _shift(seed, dx, dy))
            out_color[moved] = cast(NDArray[np.uint8], _shift(color, dx, dy))[moved]
            out_depth[moved] = cast(NDArray[np.float32], _shift(depth, dx, dy))[moved]
            return moved

        reference = paste(*script.offset)
        distractor = np.zeros_like(seed)
        if script.distractor_offset is not None:
            distractor = paste(*script.distractor_offset)
            reference &= ~distractor
        if script.occluder is not None:
            left, top, right, bottom = script.occluder
            left, right = max(0, left), min(seed.shape[1], right)
            top, bottom = max(0, top), min(seed.shape[0], bottom)
            if left < right and top < bottom:
                out_color[top:bottom, left:right] = self._occluder_texture[top:bottom, left:right]
                out_depth[top:bottom, left:right] = 0.35
                reference[top:bottom, left:right] = False
                distractor[top:bottom, left:right] = False

        frame = dataclasses.replace(
            base,
            color=SensorSample(
                base.color.header,
                ImageFrame(out_color, base.color.payload.pixel_format),
                base.color.payload_memory,
            ),
            depth=SensorSample(base.depth.header, DepthFrame(out_depth), base.depth.payload_memory),
        )
        return GeneratedFrame(frame, script, reference, distractor)


def _iou(left: NDArray[np.bool_], right: NDArray[np.bool_]) -> float:
    union = np.count_nonzero(left | right)
    return 1.0 if union == 0 else float(np.count_nonzero(left & right) / union)


@dataclass(frozen=True, slots=True)
class FrameScore:
    index: int
    phase: Phase
    reference_fraction: float
    mask_pixels: int
    object_score: float
    iou: float
    distractor_iou: float
    track_ms: float


@dataclass(frozen=True, slots=True)
class SessionEmulation:
    first_lost_index: int | None
    first_lost_phase: Phase | None
    loss_reason: LossReason | None
    tracking_while_hidden: int
    hidden_frames_before_loss: int


def emulate_session(
    scores: Sequence[FrameScore], policy: SelectionPolicy, seed_pixels: int
) -> SessionEmulation:
    previous = seed_pixels
    tracking_while_hidden = 0
    hidden_before_loss = 0
    for score in scores[1:]:
        reason: LossReason | None = None
        if score.object_score < policy.minimum_object_score:
            reason = LossReason.LOW_OBJECT_SCORE
        elif score.mask_pixels < policy.minimum_mask_pixels:
            reason = LossReason.MASK_TOO_SMALL
        elif policy.max_area_ratio is not None and previous > 0:
            ratio = score.mask_pixels / previous
            if ratio > policy.max_area_ratio or ratio < 1 / policy.max_area_ratio:
                reason = LossReason.MASK_AREA_JUMP
        hidden = score.reference_fraction < _HIDDEN_FRACTION
        if reason is not None:
            return SessionEmulation(
                score.index, score.phase, reason, tracking_while_hidden, hidden_before_loss
            )
        previous = score.mask_pixels
        if hidden:
            tracking_while_hidden += 1
            hidden_before_loss += 1
    return SessionEmulation(None, None, None, tracking_while_hidden, hidden_before_loss)


def run_sequence(
    backend: PromptableVideoSegmenter,
    base_frames: Sequence[AlignedRgbdFrame],
    click: Click,
    *,
    motion_pixels: int = 40,
    synchronize: Callable[[], None] = lambda: None,
    keep_frames: Sequence[int] = (),
) -> tuple[list[FrameScore], dict[int, tuple[GeneratedFrame, NDArray[np.bool_]]]]:
    """Seed on ``base_frames[0]``, track the scripted sequence, and score every frame."""

    backend.reset()
    first = base_frames[0]
    synchronize()
    started = time.perf_counter()
    seed = backend.seed(first, (click,))
    synchronize()
    seed_ms = (time.perf_counter() - started) * 1e3
    if seed.mask_pixels == 0:
        raise ValueError("backend produced an empty seed mask")
    sequence = PerturbationSequence(base_frames, seed.mask, motion_pixels=motion_pixels)
    seed_pixels = seed.mask_pixels
    scores = [FrameScore(0, Phase.STATIC, 1.0, seed_pixels, seed.object_score, 1.0, 0.0, seed_ms)]
    kept: dict[int, tuple[GeneratedFrame, NDArray[np.bool_]]] = {}
    if 0 in keep_frames:
        kept[0] = (sequence.generate(0), seed.mask)
    for index in range(1, len(sequence)):
        generated = sequence.generate(index)
        synchronize()
        started = time.perf_counter()
        result = backend.track(generated.frame)
        synchronize()
        track_ms = (time.perf_counter() - started) * 1e3
        scores.append(
            FrameScore(
                index=index,
                phase=generated.script.phase,
                reference_fraction=np.count_nonzero(generated.reference) / seed_pixels,
                mask_pixels=result.mask_pixels,
                object_score=result.object_score,
                iou=_iou(result.mask, generated.reference),
                distractor_iou=_iou(result.mask, generated.distractor)
                if generated.distractor.any()
                else 0.0,
                track_ms=track_ms,
            )
        )
        if index in keep_frames:
            kept[index] = (generated, result.mask)
    return scores, kept


def summarize(scores: Sequence[FrameScore], policy: SelectionPolicy) -> dict[str, object]:
    """Aggregate scores into the comparison metrics used for backend selection."""

    seed_pixels = scores[0].mask_pixels
    tracked = scores[1:]

    def present(score: FrameScore) -> bool:
        return (
            score.object_score >= policy.minimum_object_score
            and score.mask_pixels >= policy.minimum_mask_pixels
        )

    visible = [s for s in tracked if s.reference_fraction >= _PRESENT_FRACTION]
    hidden = [s for s in tracked if s.reference_fraction < _HIDDEN_FRACTION]
    by_phase: dict[str, float | None] = {}
    for phase in Phase:
        values = [s.iou for s in visible if s.phase is phase]
        by_phase[phase.value] = float(np.mean(values)) if values else None
    returned = [s for s in tracked if s.phase is Phase.RETURN and s.reference_fraction >= 0.9]
    times = np.array([s.track_ms for s in tracked[3:]])
    emulation = emulate_session(scores, policy, seed_pixels)
    return {
        "seed_mask_pixels": seed_pixels,
        "seed_object_score": scores[0].object_score,
        "visible_frames": len(visible),
        "mean_iou_visible": float(np.mean([s.iou for s in visible])) if visible else None,
        "p10_iou_visible": float(np.percentile([s.iou for s in visible], 10)) if visible else None,
        "mean_iou_by_phase": by_phase,
        "hidden_frames": len(hidden),
        "hidden_frames_reported_present": sum(present(s) for s in hidden),
        "hidden_frames_on_distractor": sum(present(s) and s.distractor_iou > 0.3 for s in hidden),
        "return_frames": len(returned),
        "return_frames_reacquired_raw": sum(present(s) and s.iou >= 0.5 for s in returned),
        "session_first_lost_index": emulation.first_lost_index,
        "session_first_lost_phase": (
            emulation.first_lost_phase.value if emulation.first_lost_phase else None
        ),
        "session_loss_reason": emulation.loss_reason.value if emulation.loss_reason else None,
        "session_tracking_while_hidden": emulation.tracking_while_hidden,
        "track_ms_p50": float(np.percentile(times, 50)) if times.size else None,
        "track_ms_p95": float(np.percentile(times, 95)) if times.size else None,
    }


__all__ = [
    "FrameScore",
    "GeneratedFrame",
    "PerturbationSequence",
    "Phase",
    "ScriptedFrame",
    "SessionEmulation",
    "build_script",
    "emulate_session",
    "run_sequence",
    "summarize",
]
