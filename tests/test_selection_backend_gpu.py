"""Opt-in integration test of the pinned research adapters on real replay frames.

Run with ``VISION_SELECTION_GPU_TESTS=1 pytest tests/test_selection_backend_gpu.py``.
It needs CUDA, ``../EfficientTAM``, ``../sam2``, ``../model-checkpoints``, and the
``recordings/d435i_table_static`` recording.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
RECORDING = REPOSITORY / "recordings" / "d435i_table_static"

pytestmark = pytest.mark.skipif(
    os.environ.get("VISION_SELECTION_GPU_TESTS") != "1",
    reason="set VISION_SELECTION_GPU_TESTS=1 to run CUDA selection backend tests",
)


@pytest.mark.parametrize("backend", ["efficienttam-ti", "sam2.1-hiera-tiny"])
def test_adapter_tracks_refines_and_bounds_memory(backend: str) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available() or not RECORDING.is_dir():
        pytest.skip("CUDA or the table recording is unavailable")
    from vision_pipeline.perception.objects.selection import (
        Click,
        SelectionSession,
        SelectionState,
        StalePromptError,
    )
    from vision_pipeline.perception.objects.selection_backend import build_research_segmenter
    from vision_pipeline.sources.rgbd_recording import replay_rgbd

    frames = []
    for frame in replay_rgbd(RECORDING):
        frames.append(frame)
        if len(frames) == 40:
            break
    segmenter = build_research_segmenter(backend, models_root=REPOSITORY.parent)
    session = SelectionSession(segmenter)

    seeded = session.select(frames[0], (Click(218, 432),))  # the white ball
    assert seeded.state is SelectionState.TRACKING
    assert seeded.segmentation is not None
    ball_pixels = seeded.segmentation.mask_pixels
    assert 500 < ball_pixels < 5000
    assert seeded.segmentation.mask[432, 218]

    memory = []
    for frame in frames[1:36]:
        update = session.update(frame)
        assert update.state is SelectionState.TRACKING
        assert update.segmentation is not None
        assert update.segmentation.frameset_id == frame.frameset_id
        memory.append(segmenter.memory_frame_count)
    assert max(memory) <= 1 + segmenter.config.memory_window_frames

    refined = session.refine(frames[33], (Click(218, 432),))
    assert refined.state is SelectionState.TRACKING
    with pytest.raises(StalePromptError):
        session.refine(frames[1], (Click(218, 432),))
    for frame in frames[36:]:
        assert session.update(frame).state is SelectionState.TRACKING
