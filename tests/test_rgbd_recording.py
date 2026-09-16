from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from rgbd_fixtures import make_frame, tabletop_scene

from vision_pipeline.sources.rgbd_recording import (
    EndOfRecordingError,
    ReplayRgbdSource,
    RgbdRecorder,
    RgbdRecordingError,
    count_rgbd_frames,
    read_rgbd_manifest,
    replay_rgbd,
)


def _record(path: Path, count: int = 3) -> list:
    frames = [tabletop_scene(sequence).frame for sequence in range(count)]
    recorder = RgbdRecorder(path, metadata={"device": {"serial_number": "test"}})
    for frame in frames:
        recorder.write(frame)
    return frames


def test_replay_is_bit_exact_and_keeps_all_provenance(tmp_path: Path) -> None:
    frames = _record(tmp_path / "recording")

    replayed = list(replay_rgbd(tmp_path / "recording"))

    assert len(replayed) == len(frames) == count_rgbd_frames(tmp_path / "recording")
    for original, copy in zip(frames, replayed, strict=True):
        assert copy.frameset_id == original.frameset_id
        assert copy.color.header == original.color.header
        assert copy.depth.header == original.depth.header
        assert copy.color_intrinsics == original.color_intrinsics
        assert copy.aligned_depth_intrinsics == original.aligned_depth_intrinsics
        assert copy.synchronization_method == original.synchronization_method
        assert copy.color.payload.pixel_format is original.color.payload.pixel_format
        assert np.array_equal(copy.color.payload.data, original.color.payload.data)
        # Bit-exact float comparison, including zero (invalid) depth.
        assert copy.depth.payload.data.tobytes() == original.depth.payload.data.tobytes()
    assert read_rgbd_manifest(tmp_path / "recording")["metadata"]["device"] == {
        "serial_number": "test"
    }


def test_recorder_refuses_to_overwrite_an_existing_recording(tmp_path: Path) -> None:
    _record(tmp_path / "recording", count=1)

    with pytest.raises(FileExistsError):
        RgbdRecorder(tmp_path / "recording")


def test_replay_rejects_modified_frame_bytes(tmp_path: Path) -> None:
    _record(tmp_path / "recording", count=2)
    target = tmp_path / "recording" / "000001.npz"
    payload = bytearray(target.read_bytes())
    payload[-10] ^= 0xFF
    target.write_bytes(bytes(payload))

    with pytest.raises(RgbdRecordingError, match="checksum mismatch"):
        list(replay_rgbd(tmp_path / "recording"))


def test_replay_rejects_unknown_schema_and_path_traversal(tmp_path: Path) -> None:
    _record(tmp_path / "recording", count=1)
    manifest = tmp_path / "recording" / "manifest.json"
    manifest.write_text(json.dumps({"schema": "something-else"}))
    with pytest.raises(RgbdRecordingError, match="schema"):
        list(replay_rgbd(tmp_path / "recording"))

    manifest.write_text(json.dumps({"schema": "aligned-rgbd-v1"}))
    checksums = tmp_path / "recording" / "checksums.ndjson"
    checksums.write_text(json.dumps({"file": "../escape.npz", "sha256": "0"}) + "\n")
    with pytest.raises(RgbdRecordingError, match="invalid recorded filename"):
        list(replay_rgbd(tmp_path / "recording"))


def test_replay_source_paces_by_recorded_receive_intervals(tmp_path: Path) -> None:
    _record(tmp_path / "recording", count=3)
    now = [0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += round(seconds * 1e9)

    source = ReplayRgbdSource(
        tmp_path / "recording", pacing=True, clock_ns=lambda: now[0], sleep=sleep
    )
    with source:
        first = source.read()
        second = source.read()
        third = source.read()
        with pytest.raises(EndOfRecordingError):
            source.read()

    assert [frame.color.header.sequence_number for frame in (first, second, third)] == [0, 1, 2]
    # Fixture frames were received 33 ms apart.
    assert sleeps == pytest.approx([0.033, 0.033])
    assert source.frames_read == 3


def test_replay_source_without_pacing_never_sleeps(tmp_path: Path) -> None:
    recorder = RgbdRecorder(tmp_path / "recording")
    recorder.write(make_frame(7))

    def fail(_: float) -> None:
        raise AssertionError("unpaced replay must not sleep")

    with ReplayRgbdSource(tmp_path / "recording", sleep=fail, preload=True) as source:
        assert source.read().frameset_id == "run-1/frameset/7"
