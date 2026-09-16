from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import pytest
from rgbd_fixtures import FRAME_ID, make_frame, tabletop_scene
from test_selection import ScriptedSegmenter

from vision_pipeline.perception.objects import WorkspaceBounds
from vision_pipeline.perception.objects.selected_geometry import (
    SelectedGeometryConfig,
    SelectedGeometryEstimator,
)
from vision_pipeline.perception.objects.selection import Click, SelectionSession, SelectionState
from vision_pipeline.rgbd import AlignedRgbdFrame
from vision_pipeline.runtime.latest_rgbd import LatestRgbdCapture, LatestSlot, SlotClosedError
from vision_pipeline.runtime.selection_pipeline import (
    CommandKind,
    GeometryOutput,
    MaskOutput,
    OperatorCommand,
    SelectionPipeline,
)
from vision_pipeline.sources.rgbd_recording import (
    EndOfRecordingError,
    ReplayRgbdSource,
    RgbdRecorder,
)


def test_latest_slot_overwrites_instead_of_queueing() -> None:
    slot: LatestSlot[int] = LatestSlot()
    for value in range(1000):
        slot.put(value)

    assert slot.take(timeout=0) == 999
    assert slot.take(timeout=0) is None
    assert slot.stats.overwritten == 999
    slot.close()
    with pytest.raises(SlotClosedError):
        slot.take(timeout=0)


class ListSource:
    def __init__(self, frames: list[AlignedRgbdFrame], delay_s: float = 0.0) -> None:
        self.frames = list(frames)
        self.delay_s = delay_s

    def read(self) -> AlignedRgbdFrame:
        if not self.frames:
            raise EndOfRecordingError("done")
        time.sleep(self.delay_s)
        return self.frames.pop(0)


def test_capture_counts_sequence_gaps_and_reports_end_of_stream() -> None:
    capture = LatestRgbdCapture(ListSource([make_frame(sequence) for sequence in (0, 1, 4, 5)]))
    capture.start()
    frames = []
    with pytest.raises(SlotClosedError) as closed:
        while True:
            frame = capture.take(timeout=2)
            if frame is not None:
                frames.append(frame)
    capture.stop()

    assert isinstance(closed.value.__cause__, EndOfRecordingError)
    assert capture.stats.color_sequence_gaps == 2
    assert capture.stats.frames_read == 4
    assert frames[-1].color.header.sequence_number == 5
    assert len(frames) + capture.stats.frames_overwritten == 4


def _geometry() -> SelectedGeometryEstimator:
    return SelectedGeometryEstimator(
        SelectedGeometryConfig(
            workspace=WorkspaceBounds(FRAME_ID, (-1, -1, 0.2), (1, 1, 2)),
            min_depth_metres=0.2,
            minimum_points=20,
        )
    )


def _run(
    source: object,
    commands: list[tuple[int, CommandKind, tuple[Click, ...]]],
    *,
    stale_frameset: bool = False,
) -> tuple[list[MaskOutput], list[GeometryOutput], SelectionPipeline]:
    scene_mask = tabletop_scene().box_mask
    backend = ScriptedSegmenter(lambda frame: (scene_mask, 0.95))
    masks: list[MaskOutput] = []
    geometries: list[GeometryOutput] = []
    pending = dict((index, (kind, clicks)) for index, kind, clicks in commands)

    def on_mask(output: MaskOutput) -> None:
        masks.append(output)
        command = pending.pop(output.index, None)
        if command is not None:
            kind, clicks = command
            frameset = "never-displayed" if stale_frameset else output.frame.frameset_id
            pipeline.submit(
                OperatorCommand(kind, None if kind is CommandKind.CLEAR else frameset, clicks)
            )

    pipeline = SelectionPipeline(
        LatestRgbdCapture(source),  # type: ignore[arg-type]
        SelectionSession(backend),
        _geometry(),
        live_timing=False,
        on_mask=on_mask,
        on_geometry=geometries.append,
    )
    pipeline.start()
    assert pipeline.wait(timeout=20)
    pipeline.stop()
    assert pipeline.error is None
    return masks, geometries, pipeline


def test_pipeline_selects_on_the_clicked_frame_and_keeps_provenance(tmp_path: Path) -> None:
    recorder = RgbdRecorder(tmp_path / "replay")
    for sequence in range(12):
        recorder.write(tabletop_scene(sequence).frame)
    # The source is slower than the worker, so every frame reaches the worker.
    source = ReplayRgbdSource(tmp_path / "replay", preload=True)
    source.open()
    slow = ListSource([source.read() for _ in range(12)], delay_s=0.02)

    masks, geometries, pipeline = _run(
        slow,
        [(1, CommandKind.SELECT, (Click(80, 60),)), (8, CommandKind.CLEAR, ())],
    )

    states = [output.update.state for output in masks]
    assert states[:2] == [SelectionState.UNSELECTED] * 2
    assert SelectionState.TRACKING in states
    assert states[-1] is SelectionState.UNSELECTED
    tracked = [output for output in geometries if output.geometry is not None]
    assert tracked
    for output in tracked:
        assert output.geometry is not None
        assert output.geometry.frameset_id == output.mask.frame.frameset_id
        assert output.geometry.depth_source == output.mask.frame.depth.header
        assert output.geometry.color_source == output.mask.frame.color.header
        assert output.geometry.centroid_metres[2] == pytest.approx(0.922, abs=3e-3)
    stats = pipeline.stats()
    assert stats.frames_processed == len(masks) == 12
    assert stats.geometry_computed == len(tracked)


def test_click_on_a_frame_the_worker_no_longer_holds_is_rejected() -> None:
    frames = [tabletop_scene(sequence).frame for sequence in range(6)]
    masks, _, _ = _run(
        ListSource(frames, delay_s=0.02),
        [(1, CommandKind.SELECT, (Click(80, 60),))],
        stale_frameset=True,
    )

    assert all(output.update.state is SelectionState.UNSELECTED for output in masks)
    assert any("no longer held" in error for output in masks for error in output.command_errors)


def test_slow_consumer_drops_frames_instead_of_building_a_backlog() -> None:
    frames = [make_frame(sequence) for sequence in range(200)]
    gate = threading.Event()
    processed: list[int] = []

    class SlowSegmenter(ScriptedSegmenter):
        def track(self, frame: AlignedRgbdFrame):  # type: ignore[no-untyped-def]
            gate.wait(0.01)
            processed.append(frame.color.header.sequence_number)
            return super().track(frame)

    backend = SlowSegmenter(lambda frame: (np.ones((120, 160), np.bool_), 0.9))
    session = SelectionSession(backend)
    session.select(frames[0], (Click(80, 60),))
    pipeline = SelectionPipeline(
        LatestRgbdCapture(ListSource(frames[1:])),
        session,
        None,
        live_timing=False,
    )
    pipeline.start()
    assert pipeline.wait(timeout=20)
    pipeline.stop()

    stats = pipeline.stats()
    assert stats.capture_frames_read == 199
    assert stats.frames_processed + stats.capture_frames_overwritten == 199
    assert stats.capture_frames_overwritten > 0
    assert processed == sorted(processed)  # never out of order, never replayed
