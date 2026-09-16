"""Click-prompted single-object selection: the backend contract and its state machine.

A :class:`PromptableVideoSegmenter` turns operator clicks on one exact RGB-D frame into
a mask and then propagates that one instance through later frames. It is model
independent: EfficientTAM and SAM 2 adapters live in ``selection_backend.py`` and are
imported only when a caller asks for them.

:class:`SelectionSession` owns the policy the brief requires:

- ``UNSELECTED`` until the operator gives at least one positive click;
- ``TRACKING`` while every propagated mask passes the loss checks;
- ``LOST`` as soon as one does not. ``LOST`` latches: the session never re-acquires or
  substitutes an object by itself, and only a new selection leaves it.

Every mask names the frameset and color sample it was computed from, and the session
refuses a backend result that does not match the frame it was given, so a mask can
never be paired with a different depth image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import SampleHeader
from vision_pipeline.rgbd import AlignedRgbdFrame


class SelectionState(StrEnum):
    UNSELECTED = "UNSELECTED"
    TRACKING = "TRACKING"
    LOST = "LOST"


class LossReason(StrEnum):
    """Why a tracked selection was declared ``LOST``."""

    LOW_OBJECT_SCORE = "low_object_score"
    MASK_TOO_SMALL = "mask_too_small"
    MASK_AREA_JUMP = "mask_area_jump"


class StalePromptError(ValueError):
    """A refinement click refers to a frame the segmenter no longer holds in memory."""


@dataclass(frozen=True, slots=True)
class Click:
    """One point prompt in source-image pixel coordinates.

    ``x`` is the column and ``y`` the row of a pixel index; ``(0, 0)`` is the top-left
    pixel's centre. ``positive`` clicks mark the object, negative clicks mark background.
    """

    x: float
    y: float
    positive: bool = True

    def __post_init__(self) -> None:
        for name in ("x", "y"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if not isinstance(self.positive, bool):
            raise TypeError("positive must be a bool")

    def inside(self, width: int, height: int) -> bool:
        return 0 <= self.x <= width - 1 and 0 <= self.y <= height - 1


def map_click(
    x: float,
    y: float,
    *,
    viewport: tuple[int, int, int, int],
    image_size: tuple[int, int],
    positive: bool = True,
) -> Click:
    """Map a window pixel inside a displayed image rectangle to a source pixel index.

    ``viewport`` is ``(left, top, width, height)`` of the drawn image in window pixels and
    ``image_size`` is the source ``(width, height)``. Pixel centres map to pixel centres,
    so an unscaled viewport is the identity. Clicks in letterbox borders are rejected
    instead of being clamped onto an image edge the operator did not click.
    """

    left, top, width, height = viewport
    image_width, image_height = image_size
    if min(width, height, image_width, image_height) <= 0:
        raise ValueError("image and viewport dimensions must be positive")
    if not (left <= x < left + width and top <= y < top + height):
        raise ValueError("click is outside the displayed image")
    source_x = (x - left + 0.5) * image_width / width - 0.5
    source_y = (y - top + 0.5) * image_height / height - 0.5
    return Click(
        min(max(source_x, 0.0), image_width - 1.0),
        min(max(source_y, 0.0), image_height - 1.0),
        positive,
    )


@dataclass(frozen=True, slots=True)
class SegmentationResult:
    """One backend mask for exactly one input frame.

    ``mask`` has the aligned frame's ``(height, width)`` and is therefore pixel-registered
    with its depth plane. ``object_score`` is the model's object-presence probability,
    not a calibrated geometric confidence.
    """

    frameset_id: str
    source: SampleHeader
    mask: NDArray[np.bool_]
    object_score: float
    backend: str

    def __post_init__(self) -> None:
        if not isinstance(self.frameset_id, str) or not self.frameset_id.strip():
            raise ValueError("frameset_id must be a non-empty string")
        if not isinstance(self.source, SampleHeader):
            raise TypeError("source must be a SampleHeader")
        if not isinstance(self.mask, np.ndarray) or self.mask.dtype != np.bool_:
            raise TypeError("mask must be a boolean numpy.ndarray")
        if self.mask.ndim != 2:
            raise ValueError("mask must have HxW shape")
        if not math.isfinite(self.object_score) or not 0 <= self.object_score <= 1:
            raise ValueError("object_score must be finite and between zero and one")
        if not isinstance(self.backend, str) or not self.backend.strip():
            raise ValueError("backend must be a non-empty string")

    @property
    def mask_pixels(self) -> int:
        return int(np.count_nonzero(self.mask))


def require_matching_frame(result: SegmentationResult, frame: AlignedRgbdFrame) -> None:
    """Raise unless ``result`` was computed from ``frame``'s exact color/depth pair."""

    if result.frameset_id != frame.frameset_id or result.source != frame.color.header:
        raise ValueError("segmentation result does not belong to this RGB-D frame")
    if result.mask.shape != frame.depth.payload.data.shape:
        raise ValueError("segmentation mask and aligned depth dimensions differ")


@runtime_checkable
class PromptableVideoSegmenter(Protocol):
    """Single-object promptable video segmentation on one explicit frame at a time.

    - ``seed`` discards all temporal memory and starts a new object from clicks on
      ``frame``; at least one click must be positive.
    - ``refine`` adds clicks to a frame this object was already seeded or tracked on,
      correcting the current memory rather than starting over. Implementations may
      hold only a bounded number of recent frames and raise :class:`StalePromptError`
      for older ones.
    - ``track`` propagates the object to a new frame without prompts.
    - ``reset`` releases temporal memory.

    Results always refer to the frame passed in. Temporal memory must be bounded
    independently of how long the stream runs.
    """

    @property
    def name(self) -> str: ...

    def seed(self, frame: AlignedRgbdFrame, clicks: tuple[Click, ...]) -> SegmentationResult: ...

    def refine(self, frame: AlignedRgbdFrame, clicks: tuple[Click, ...]) -> SegmentationResult: ...

    def track(self, frame: AlignedRgbdFrame) -> SegmentationResult: ...

    def reset(self) -> None: ...


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Loss checks applied to every seeded, refined, or tracked mask.

    ``max_area_ratio`` rejects a mask whose pixel area grows or shrinks by more than
    that factor between consecutive accepted frames, which is how a jump to a different
    object usually looks in the image. ``None`` disables that check.
    """

    minimum_object_score: float = 0.5
    minimum_mask_pixels: int = 50
    max_area_ratio: float | None = 4.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_object_score) or not (
            0 <= self.minimum_object_score <= 1
        ):
            raise ValueError("minimum_object_score must be between zero and one")
        if isinstance(self.minimum_mask_pixels, bool) or not isinstance(
            self.minimum_mask_pixels, int
        ):
            raise TypeError("minimum_mask_pixels must be an integer")
        if self.minimum_mask_pixels < 1:
            raise ValueError("minimum_mask_pixels must be positive")
        if self.max_area_ratio is not None and (
            not math.isfinite(self.max_area_ratio) or self.max_area_ratio <= 1
        ):
            raise ValueError("max_area_ratio must be greater than one or None")


@dataclass(frozen=True, slots=True)
class SelectionUpdate:
    """The session's decision for one frame.

    ``segmentation`` is present whenever a backend ran on this frame, including the
    frame that caused ``LOST`` (so a viewer can show what was rejected); only
    ``TRACKING`` updates may be converted to geometry. ``selection_id`` increases with
    every new selection and is 0 before the first one.
    """

    state: SelectionState
    frameset_id: str
    source: SampleHeader
    selection_id: int
    segmentation: SegmentationResult | None = None
    loss_reason: LossReason | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, SelectionState):
            raise TypeError("state must be a SelectionState")
        # The frame that causes LOST carries its rejected mask and the reason; later
        # LOST frames run no backend and carry neither.
        deciding_loss = self.state is SelectionState.LOST and self.segmentation is not None
        if deciding_loss != (self.loss_reason is not None):
            raise ValueError("loss_reason must be given exactly for the frame that caused LOST")
        if self.state is SelectionState.TRACKING and self.segmentation is None:
            raise ValueError("a TRACKING update requires a segmentation result")
        if self.segmentation is not None and self.segmentation.frameset_id != self.frameset_id:
            raise ValueError("segmentation belongs to a different frameset")

    @property
    def is_tracking(self) -> bool:
        return self.state is SelectionState.TRACKING


class SelectionSession:
    """Explicit ``UNSELECTED``/``TRACKING``/``LOST`` policy around one segmenter."""

    def __init__(
        self,
        backend: PromptableVideoSegmenter,
        policy: SelectionPolicy | None = None,
    ) -> None:
        policy = SelectionPolicy() if policy is None else policy
        if not isinstance(policy, SelectionPolicy):
            raise TypeError("policy must be a SelectionPolicy")
        self._backend = backend
        self._policy = policy
        self._state = SelectionState.UNSELECTED
        self._selection_id = 0
        self._last_accepted: SegmentationResult | None = None
        self._loss_reason: LossReason | None = None

    @property
    def state(self) -> SelectionState:
        return self._state

    @property
    def selection_id(self) -> int:
        return self._selection_id

    @property
    def loss_reason(self) -> LossReason | None:
        return self._loss_reason

    @property
    def backend(self) -> PromptableVideoSegmenter:
        return self._backend

    def clear(self) -> None:
        """Forget the selection and release backend memory."""

        self._backend.reset()
        self._last_accepted = None
        self._loss_reason = None
        self._state = SelectionState.UNSELECTED

    def select(self, frame: AlignedRgbdFrame, clicks: tuple[Click, ...]) -> SelectionUpdate:
        """Start a new selection on ``frame``, replacing any current or lost one."""

        self._require_clicks(frame, clicks)
        if not any(click.positive for click in clicks):
            raise ValueError("a new selection requires at least one positive click")
        self._backend.reset()
        self._selection_id += 1
        self._last_accepted = None
        self._loss_reason = None
        return self._accept(frame, self._backend.seed(frame, clicks))

    def refine(self, frame: AlignedRgbdFrame, clicks: tuple[Click, ...]) -> SelectionUpdate:
        """Add correction clicks to a recently tracked frame of the current selection."""

        if self._state is not SelectionState.TRACKING:
            raise RuntimeError("refinement is only possible while TRACKING; reselect instead")
        self._require_clicks(frame, clicks)
        # A correction deliberately changes the mask, so it is not an identity jump.
        self._last_accepted = None
        return self._accept(frame, self._backend.refine(frame, clicks))

    def update(self, frame: AlignedRgbdFrame) -> SelectionUpdate:
        """Propagate the selection to ``frame``; does nothing unless ``TRACKING``."""

        if self._state is not SelectionState.TRACKING:
            return SelectionUpdate(
                self._state, frame.frameset_id, frame.color.header, self._selection_id
            )
        return self._accept(frame, self._backend.track(frame))

    def _require_clicks(self, frame: AlignedRgbdFrame, clicks: tuple[Click, ...]) -> None:
        if not isinstance(clicks, tuple) or not clicks:
            raise ValueError("at least one click is required")
        if not all(isinstance(click, Click) for click in clicks):
            raise TypeError("clicks must be Click instances")
        width, height = frame.depth.payload.width, frame.depth.payload.height
        if not all(click.inside(width, height) for click in clicks):
            raise ValueError("click is outside the source frame")

    def _loss_reason_for(self, result: SegmentationResult) -> LossReason | None:
        policy = self._policy
        if result.object_score < policy.minimum_object_score:
            return LossReason.LOW_OBJECT_SCORE
        pixels = result.mask_pixels
        if pixels < policy.minimum_mask_pixels:
            return LossReason.MASK_TOO_SMALL
        previous = self._last_accepted
        if previous is not None and policy.max_area_ratio is not None:
            ratio = pixels / previous.mask_pixels
            if ratio > policy.max_area_ratio or ratio < 1 / policy.max_area_ratio:
                return LossReason.MASK_AREA_JUMP
        return None

    def _accept(self, frame: AlignedRgbdFrame, result: SegmentationResult) -> SelectionUpdate:
        try:
            require_matching_frame(result, frame)
        except ValueError:
            self.clear()
            raise
        reason = self._loss_reason_for(result)
        if reason is None:
            self._state = SelectionState.TRACKING
            self._last_accepted = result
        else:
            self._backend.reset()
            self._state = SelectionState.LOST
            self._last_accepted = None
        self._loss_reason = reason
        return SelectionUpdate(
            self._state,
            frame.frameset_id,
            frame.color.header,
            self._selection_id,
            segmentation=result,
            loss_reason=reason,
        )


__all__ = [
    "Click",
    "LossReason",
    "PromptableVideoSegmenter",
    "SegmentationResult",
    "SelectionPolicy",
    "SelectionSession",
    "SelectionState",
    "SelectionUpdate",
    "StalePromptError",
    "map_click",
    "require_matching_frame",
]
