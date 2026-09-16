from __future__ import annotations

import numpy as np
from rgbd_fixtures import HEIGHT, WIDTH, make_frame
from test_selection import ScriptedSegmenter

from vision_pipeline.perception.objects.selection import (
    Click,
    LossReason,
    SegmentationResult,
    SelectionPolicy,
)
from vision_pipeline.perception.objects.selection_evaluation import (
    FrameScore,
    PerturbationSequence,
    Phase,
    build_script,
    emulate_session,
    run_sequence,
    summarize,
)
from vision_pipeline.rgbd import AlignedRgbdFrame


def _seed() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), np.bool_)
    mask[50:70, 30:50] = True
    return mask


def test_script_has_every_phase_and_hides_the_object() -> None:
    script = build_script(_seed())

    assert len(script) == 150
    assert {frame.phase for frame in script} == set(Phase)
    covered = [frame for frame in script if frame.phase is Phase.OCCLUSION]
    assert any(
        frame.occluder is not None and frame.occluder[0] <= 30 and frame.occluder[2] >= 50
        for frame in covered
    )
    assert min(frame.offset[0] for frame in script if frame.phase is Phase.EXIT) <= -50


def test_generated_reference_tracks_the_pasted_object() -> None:
    frames = [make_frame(sequence) for sequence in range(4)]
    sequence = PerturbationSequence(frames, _seed())

    for index in (0, 40, 90, 120, 145):
        generated = sequence.generate(index)
        dx, dy = generated.script.offset
        expected = np.roll(np.roll(_seed(), dy, axis=0), dx, axis=1)
        if abs(dx) >= 50:
            expected[:] = False
        visible = generated.reference
        assert not (visible & ~expected).any()
        assert not (visible & generated.distractor).any()
        assert generated.frame.color.header == frames[index % 4].color.header


class Oracle(ScriptedSegmenter):
    """Returns the exact visible reference, like a perfect tracker would."""

    def __init__(self, sequence_holder: list[PerturbationSequence]) -> None:
        super().__init__(lambda frame: (_seed(), 1.0))
        self.holder = sequence_holder
        self.index = 0

    def track(self, frame: AlignedRgbdFrame) -> SegmentationResult:
        self.index += 1
        reference = self.holder[0].generate(self.index).reference
        score = 1.0 if reference.any() else 0.0
        return SegmentationResult(frame.frameset_id, frame.color.header, reference, score, "o")


def test_perfect_tracker_scores_perfectly_and_is_lost_only_when_hidden() -> None:
    frames = [make_frame(sequence) for sequence in range(4)]
    holder = [PerturbationSequence(frames, _seed())]

    scores, _ = run_sequence(Oracle(holder), frames, Click(40, 60))
    summary = summarize(scores, SelectionPolicy(minimum_mask_pixels=10, max_area_ratio=None))

    assert summary["mean_iou_visible"] == 1.0
    assert summary["hidden_frames_reported_present"] == 0
    assert summary["hidden_frames_on_distractor"] == 0
    assert summary["session_first_lost_phase"] in (Phase.OCCLUSION.value, Phase.EXIT.value)
    assert summary["session_tracking_while_hidden"] == 0


def test_emulated_session_latches_the_first_loss() -> None:
    def score(index: int, pixels: int, value: float, fraction: float = 1.0) -> FrameScore:
        return FrameScore(index, Phase.STATIC, fraction, pixels, value, 1.0, 0.0, 1.0)

    scores = [score(0, 100, 1.0), score(1, 100, 0.9), score(2, 100, 0.9, 0.0), score(3, 5, 0.9)]
    emulation = emulate_session(scores, SelectionPolicy(minimum_mask_pixels=10), 100)

    assert emulation.first_lost_index == 3
    assert emulation.loss_reason is LossReason.MASK_TOO_SMALL
    assert emulation.tracking_while_hidden == 1
