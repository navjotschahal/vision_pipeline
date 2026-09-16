from __future__ import annotations

import dataclasses
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray
from rgbd_fixtures import HEIGHT, WIDTH, make_frame

from vision_pipeline.perception.objects.selection import (
    Click,
    LossReason,
    PromptableVideoSegmenter,
    SegmentationResult,
    SelectionPolicy,
    SelectionSession,
    SelectionState,
    SelectionUpdate,
    map_click,
)
from vision_pipeline.rgbd import AlignedRgbdFrame

MaskFunction = Callable[[AlignedRgbdFrame], tuple[NDArray[np.bool_], float]]


def square(size: int = 20, x: int = 40, y: int = 30) -> NDArray[np.bool_]:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    mask[y : y + size, x : x + size] = True
    return mask


class ScriptedSegmenter:
    """A PromptableVideoSegmenter whose masks come from a function of the frame."""

    def __init__(self, masks: MaskFunction) -> None:
        self.masks = masks
        self.calls: list[tuple[str, str]] = []
        self.resets = 0

    @property
    def name(self) -> str:
        return "scripted"

    def _result(self, frame: AlignedRgbdFrame) -> SegmentationResult:
        mask, score = self.masks(frame)
        return SegmentationResult(frame.frameset_id, frame.color.header, mask, score, self.name)

    def seed(self, frame: AlignedRgbdFrame, clicks: tuple[Click, ...]) -> SegmentationResult:
        self.calls.append(("seed", frame.frameset_id))
        return self._result(frame)

    def refine(self, frame: AlignedRgbdFrame, clicks: tuple[Click, ...]) -> SegmentationResult:
        self.calls.append(("refine", frame.frameset_id))
        return self._result(frame)

    def track(self, frame: AlignedRgbdFrame) -> SegmentationResult:
        self.calls.append(("track", frame.frameset_id))
        return self._result(frame)

    def reset(self) -> None:
        self.resets += 1


def test_scripted_segmenter_satisfies_the_protocol() -> None:
    assert isinstance(ScriptedSegmenter(lambda frame: (square(), 1.0)), PromptableVideoSegmenter)


def test_map_click_is_identity_for_an_unscaled_viewport() -> None:
    click = map_click(12, 34, viewport=(0, 0, 640, 480), image_size=(640, 480))
    assert (click.x, click.y, click.positive) == (12, 34, True)


def test_map_click_scales_pixel_centres_and_offsets() -> None:
    # A 2x display drawn at (100, 50): window pixels 100 and 101 are halves of column 0.
    first = map_click(100, 50, viewport=(100, 50, 1280, 960), image_size=(640, 480))
    second = map_click(101, 51, viewport=(100, 50, 1280, 960), image_size=(640, 480))
    last = map_click(1379, 1009, viewport=(100, 50, 1280, 960), image_size=(640, 480))
    negative = map_click(
        740, 530, viewport=(100, 50, 1280, 960), image_size=(640, 480), positive=False
    )

    assert first.x == pytest.approx(0.0) and first.y == pytest.approx(0.0)
    assert second.x == pytest.approx(0.25) and second.y == pytest.approx(0.25)
    assert (last.x, last.y) == (639.0, 479.0)
    assert (negative.x, negative.y, negative.positive) == (319.75, 239.75, False)


@pytest.mark.parametrize("point", [(99, 60), (1380, 60), (500, 49), (500, 1010)])
def test_map_click_rejects_letterbox_clicks(point: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="outside"):
        map_click(*point, viewport=(100, 50, 1280, 960), image_size=(640, 480))


def test_click_rejects_non_finite_and_non_numeric_coordinates() -> None:
    with pytest.raises(ValueError):
        Click(float("nan"), 1.0)
    with pytest.raises(TypeError):
        Click(True, 1.0)  # type: ignore[arg-type]


def test_selection_starts_unselected_and_tracks_after_a_positive_click() -> None:
    backend = ScriptedSegmenter(lambda frame: (square(), 0.9))
    session = SelectionSession(backend)
    frame = make_frame(0)

    idle = session.update(frame)
    selected = session.select(frame, (Click(45, 35),))
    tracked = session.update(make_frame(1))

    assert idle.state is SelectionState.UNSELECTED and idle.segmentation is None
    assert backend.calls == [("seed", frame.frameset_id), ("track", "run-1/frameset/1")]
    assert selected.state is SelectionState.TRACKING and selected.selection_id == 1
    assert tracked.state is SelectionState.TRACKING
    assert tracked.segmentation is not None
    assert tracked.segmentation.frameset_id == "run-1/frameset/1"
    assert tracked.source == make_frame(1).color.header


def test_selection_requires_a_positive_click_inside_the_frame() -> None:
    session = SelectionSession(ScriptedSegmenter(lambda frame: (square(), 1.0)))
    frame = make_frame(0)

    with pytest.raises(ValueError, match="positive"):
        session.select(frame, (Click(10, 10, positive=False),))
    with pytest.raises(ValueError, match="outside"):
        session.select(frame, (Click(WIDTH, 10),))
    assert session.state is SelectionState.UNSELECTED


def test_low_score_latches_lost_and_never_reacquires_automatically() -> None:
    scores = {0: 0.9, 1: 0.9, 2: 0.2, 3: 0.95, 4: 0.95}
    backend = ScriptedSegmenter(
        lambda frame: (square(), scores[frame.color.header.sequence_number])
    )
    session = SelectionSession(backend)
    session.select(make_frame(0), (Click(45, 35),))
    session.update(make_frame(1))

    lost = session.update(make_frame(2))
    after = [session.update(make_frame(sequence)) for sequence in (3, 4)]

    assert lost.state is SelectionState.LOST
    assert lost.loss_reason is LossReason.LOW_OBJECT_SCORE
    assert lost.segmentation is not None  # the rejected mask stays visible
    assert all(update.state is SelectionState.LOST for update in after)
    assert all(update.segmentation is None and update.loss_reason is None for update in after)
    # The object came back with a high score, but the backend was never asked again.
    assert [call for call, _ in backend.calls].count("track") == 2
    assert backend.resets >= 2  # select() reset and the loss reset


def test_empty_mask_and_area_jump_are_losses() -> None:
    masks = {0: square(20), 1: np.zeros((HEIGHT, WIDTH), np.bool_)}
    session = SelectionSession(
        ScriptedSegmenter(lambda frame: (masks[frame.color.header.sequence_number], 0.99))
    )
    session.select(make_frame(0), (Click(45, 35),))
    assert session.update(make_frame(1)).loss_reason is LossReason.MASK_TOO_SMALL

    jumps = {0: square(10), 1: square(60, x=10, y=10)}
    session = SelectionSession(
        ScriptedSegmenter(lambda frame: (jumps[frame.color.header.sequence_number], 0.99)),
        SelectionPolicy(minimum_mask_pixels=10, max_area_ratio=4.0),
    )
    session.select(make_frame(0), (Click(45, 35),))
    assert session.update(make_frame(1)).loss_reason is LossReason.MASK_AREA_JUMP


def test_reselection_after_loss_starts_a_new_selection() -> None:
    scores = {0: 0.9, 1: 0.1, 2: 0.9, 3: 0.9}
    backend = ScriptedSegmenter(
        lambda frame: (square(), scores[frame.color.header.sequence_number])
    )
    session = SelectionSession(backend)
    session.select(make_frame(0), (Click(45, 35),))
    assert session.update(make_frame(1)).state is SelectionState.LOST

    reselected = session.select(make_frame(2), (Click(45, 35),))
    tracked = session.update(make_frame(3))

    assert reselected.state is SelectionState.TRACKING and reselected.selection_id == 2
    assert tracked.state is SelectionState.TRACKING and tracked.selection_id == 2
    assert session.loss_reason is None


def test_clear_returns_to_unselected_and_releases_backend_memory() -> None:
    backend = ScriptedSegmenter(lambda frame: (square(), 0.9))
    session = SelectionSession(backend)
    session.select(make_frame(0), (Click(45, 35),))
    resets = backend.resets

    session.clear()

    assert session.state is SelectionState.UNSELECTED
    assert backend.resets == resets + 1
    assert session.update(make_frame(1)).state is SelectionState.UNSELECTED


def test_refine_requires_tracking_and_uses_the_clicked_frame() -> None:
    backend = ScriptedSegmenter(lambda frame: (square(), 0.9))
    session = SelectionSession(backend)
    with pytest.raises(RuntimeError, match="TRACKING"):
        session.refine(make_frame(0), (Click(1, 1, positive=False),))

    session.select(make_frame(0), (Click(45, 35),))
    session.update(make_frame(1))
    refined = session.refine(make_frame(1), (Click(1, 1, positive=False),))

    assert refined.state is SelectionState.TRACKING
    assert backend.calls[-1] == ("refine", "run-1/frameset/1")


def test_result_for_a_different_frame_is_refused_and_clears_the_selection() -> None:
    other = make_frame(99, run="other-run")

    class WrongFrame(ScriptedSegmenter):
        def track(self, frame: AlignedRgbdFrame) -> SegmentationResult:
            return SegmentationResult(
                other.frameset_id, other.color.header, square(), 0.9, self.name
            )

    session = SelectionSession(WrongFrame(lambda frame: (square(), 0.9)))
    session.select(make_frame(0), (Click(45, 35),))

    with pytest.raises(ValueError, match="does not belong"):
        session.update(make_frame(1))
    assert session.state is SelectionState.UNSELECTED


def test_segmentation_result_validates_mask_and_score() -> None:
    frame = make_frame(0)
    with pytest.raises(TypeError):
        SegmentationResult(frame.frameset_id, frame.color.header, square().astype(np.uint8), 1, "x")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        SegmentationResult(frame.frameset_id, frame.color.header, square(), 1.5, "x")


def test_selection_update_rejects_inconsistent_states() -> None:
    frame = make_frame(0)
    result = SegmentationResult(frame.frameset_id, frame.color.header, square(), 0.9, "x")
    with pytest.raises(ValueError):
        SelectionUpdate(SelectionState.TRACKING, frame.frameset_id, frame.color.header, 1)
    with pytest.raises(ValueError):
        SelectionUpdate(SelectionState.LOST, frame.frameset_id, frame.color.header, 1, result)
    with pytest.raises(ValueError, match="different frameset"):
        SelectionUpdate(
            SelectionState.TRACKING,
            "another",
            frame.color.header,
            1,
            dataclasses.replace(result),
        )
