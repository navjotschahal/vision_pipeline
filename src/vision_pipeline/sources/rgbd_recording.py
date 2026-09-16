"""Lossless, versioned recording and deterministic replay of aligned RGB-D frames.

Every frame is one uncompressed ``.npz`` written without pickle: the color pixels, the
metric depth plane, and a JSON copy of both sample headers, the intrinsics, and the
synchronization/alignment provenance. Replay reconstructs exactly those values. Nothing
is re-stamped: frameset IDs, sample IDs, sequence numbers, and every clock domain are the
recorded ones, so a replayed frame differs from the captured frame only in where its
bytes live.

``checksums.ndjson`` lists frames in recorded order with their SHA-256; replay refuses a
recording whose bytes changed.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Self

import numpy as np

from vision_pipeline.contracts import (
    CalibrationRef,
    ClockDomain,
    ClockKind,
    ComputeKind,
    ComputePlacement,
    FrameId,
    MeasurementKind,
    MemoryKind,
    MemoryPlacement,
    SampleHeader,
    SensorSample,
    TimePoint,
)
from vision_pipeline.geometry.camera import PinholeIntrinsics
from vision_pipeline.image import ImageFrame, PixelFormat
from vision_pipeline.rgbd import AlignedRgbdFrame, DepthFrame

SCHEMA = "aligned-rgbd-v1"
_MANIFEST = "manifest.json"
_CHECKSUMS = "checksums.ndjson"


class RgbdRecordingError(ValueError):
    """A recording is missing, corrupt, or uses an unsupported schema."""


class EndOfRecordingError(EOFError):
    """A replay source has returned every recorded frame."""


def _time_from_json(value: Mapping[str, Any] | None) -> TimePoint | None:
    if value is None:
        return None
    clock = value["clock"]
    return TimePoint(
        int(value["nanoseconds"]),
        ClockDomain(str(clock["domain_id"]), ClockKind(clock["kind"])),
    )


def _header_from_json(value: Mapping[str, Any]) -> SampleHeader:
    received_at = _time_from_json(value["received_at"])
    if received_at is None:
        raise RgbdRecordingError("recorded header has no received_at timestamp")
    calibration = value["calibration"]
    produced_on = value["produced_on"]
    return SampleHeader(
        sample_id=str(value["sample_id"]),
        source_id=str(value["source_id"]),
        sequence_number=int(value["sequence_number"]),
        measurement_kind=MeasurementKind(value["measurement_kind"]),
        captured_at=_time_from_json(value["captured_at"]),
        received_at=received_at,
        frame_id=FrameId(str(value["frame_id"]["value"])),
        calibration=(
            CalibrationRef(str(calibration["calibration_id"]), str(calibration["revision"]))
            if calibration is not None
            else None
        ),
        producer=str(value["producer"]),
        produced_on=ComputePlacement(
            ComputeKind(produced_on["kind"]), str(produced_on["device_id"])
        ),
    )


def _intrinsics_from_json(value: Mapping[str, Any]) -> PinholeIntrinsics:
    return PinholeIntrinsics(
        width=int(value["width"]),
        height=int(value["height"]),
        fx=float(value["fx"]),
        fy=float(value["fy"]),
        cx=float(value["cx"]),
        cy=float(value["cy"]),
        calibration_id=value["calibration_id"],
    )


class RgbdRecorder:
    """Append complete frames to a new directory; never overwrite an existing recording.

    Each frame file is written under a temporary name and renamed into place, so a
    crashed recording contains only complete frames. ``metadata`` (for example device
    identity and negotiated stream modes) is stored verbatim in the manifest.
    """

    def __init__(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=False)
        self._count = 0
        manifest = {"schema": SCHEMA, "metadata": dict(metadata or {})}
        (self._path / _MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True))

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._count

    def write(self, frame: AlignedRgbdFrame) -> None:
        if not isinstance(frame, AlignedRgbdFrame):
            raise TypeError("frame must be an AlignedRgbdFrame")
        metadata = {
            "frameset_id": frame.frameset_id,
            "color": asdict(frame.color.header),
            "depth": asdict(frame.depth.header),
            "pixel_format": frame.color.payload.pixel_format.value,
            "color_intrinsics": asdict(frame.color_intrinsics),
            "aligned_depth_intrinsics": asdict(frame.aligned_depth_intrinsics),
            "synchronization_method": frame.synchronization_method,
            "alignment_method": frame.alignment_method,
        }
        buffer = io.BytesIO()
        np.savez(
            buffer,
            color=frame.color.payload.data,
            depth=frame.depth.payload.data,
            metadata=np.array(json.dumps(metadata, sort_keys=True)),
        )
        payload = buffer.getvalue()
        target = self._path / f"{self._count:06d}.npz"
        temporary = target.with_suffix(".tmp")
        temporary.write_bytes(payload)
        temporary.rename(target)
        entry = {"file": target.name, "sha256": hashlib.sha256(payload).hexdigest()}
        with (self._path / _CHECKSUMS).open("a") as stream:
            stream.write(json.dumps(entry) + "\n")
        self._count += 1


def read_rgbd_manifest(path: str | Path) -> dict[str, Any]:
    """Return a checked manifest; recordings made before metadata existed get ``{}``."""

    manifest_path = Path(path) / _MANIFEST
    if not manifest_path.is_file():
        raise RgbdRecordingError(f"{path} is not an RGB-D recording (no {_MANIFEST})")
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA:
        raise RgbdRecordingError(f"{path}: unsupported RGB-D recording schema")
    manifest.setdefault("metadata", {})
    return manifest


def _frame_entries(root: Path) -> list[tuple[str, str]]:
    checksums = root / _CHECKSUMS
    if not checksums.is_file():
        raise RgbdRecordingError(f"{root}: recording has no {_CHECKSUMS}")
    entries = []
    for line in checksums.read_text().splitlines():
        entry = json.loads(line)
        name = str(entry["file"])
        if Path(name).name != name or not name.endswith(".npz"):
            raise RgbdRecordingError(f"{root}: invalid recorded filename {name!r}")
        entries.append((name, str(entry["sha256"])))
    if not entries:
        raise RgbdRecordingError(f"{root}: recording contains no frames")
    return entries


def _decode_frame(payload: bytes) -> AlignedRgbdFrame:
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        color = np.ascontiguousarray(data["color"])
        depth = np.ascontiguousarray(data["depth"])
    return AlignedRgbdFrame(
        frameset_id=str(meta["frameset_id"]),
        color=SensorSample(
            header=_header_from_json(meta["color"]),
            payload=ImageFrame(color, PixelFormat(meta["pixel_format"])),
            payload_memory=MemoryPlacement(MemoryKind.HOST),
        ),
        depth=SensorSample(
            header=_header_from_json(meta["depth"]),
            payload=DepthFrame(depth),
            payload_memory=MemoryPlacement(MemoryKind.HOST),
        ),
        color_intrinsics=_intrinsics_from_json(meta["color_intrinsics"]),
        aligned_depth_intrinsics=_intrinsics_from_json(meta["aligned_depth_intrinsics"]),
        synchronization_method=str(meta["synchronization_method"]),
        alignment_method=str(meta["alignment_method"]),
    )


def count_rgbd_frames(path: str | Path) -> int:
    root = Path(path)
    read_rgbd_manifest(root)
    return len(_frame_entries(root))


def replay_rgbd(path: str | Path, *, verify_checksums: bool = True) -> Iterator[AlignedRgbdFrame]:
    """Yield recorded frames in recorded order, with all identities and clocks intact."""

    root = Path(path)
    read_rgbd_manifest(root)
    for name, expected in _frame_entries(root):
        payload = (root / name).read_bytes()
        if verify_checksums and hashlib.sha256(payload).hexdigest() != expected:
            raise RgbdRecordingError(f"{root}: checksum mismatch for {name}")
        yield _decode_frame(payload)


class ReplayRgbdSource:
    """A ``read()`` source over a recording, for code written against a live camera.

    With ``pacing`` enabled, ``read()`` blocks so frames are released with their recorded
    host-receive intervals, which lets latest-frame scheduling drop frames the way it
    would live. Without pacing every frame is returned immediately, so a consumer can
    process all of them deterministically. ``read()`` raises
    :class:`EndOfRecordingError` after the last frame.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        pacing: bool = False,
        preload: bool = False,
        verify_checksums: bool = True,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._path = Path(path)
        self._pacing = pacing
        self._preload = preload
        self._verify = verify_checksums
        self._clock_ns = clock_ns
        self._sleep = sleep
        self._frames: Iterator[AlignedRgbdFrame] | None = None
        self._started_ns = 0
        self._first_received_ns: int | None = None
        self._count = 0
        self._metadata: dict[str, Any] = {}

    @property
    def is_open(self) -> bool:
        return self._frames is not None

    @property
    def frames_read(self) -> int:
        return self._count

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def open(self) -> None:
        if self.is_open:
            raise RgbdRecordingError("replay source is already open")
        self._metadata = read_rgbd_manifest(self._path)["metadata"]
        frames = replay_rgbd(self._path, verify_checksums=self._verify)
        self._frames = iter(list(frames)) if self._preload else frames
        self._first_received_ns = None
        self._count = 0

    def read(self) -> AlignedRgbdFrame:
        frames = self._frames
        if frames is None:
            raise RgbdRecordingError("open() must be called before read()")
        try:
            frame = next(frames)
        except StopIteration:
            raise EndOfRecordingError(f"{self._path}: end of recording") from None
        if self._pacing:
            received_ns = frame.color.header.received_at.nanoseconds
            if self._first_received_ns is None:
                self._first_received_ns = received_ns
                self._started_ns = self._clock_ns()
            due_ns = self._started_ns + (received_ns - self._first_received_ns)
            remaining_ns = due_ns - self._clock_ns()
            if remaining_ns > 0:
                self._sleep(remaining_ns / 1e9)
        self._count += 1
        return frame

    def close(self) -> None:
        self._frames = None

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "SCHEMA",
    "EndOfRecordingError",
    "ReplayRgbdSource",
    "RgbdRecorder",
    "RgbdRecordingError",
    "count_rgbd_frames",
    "read_rgbd_manifest",
    "replay_rgbd",
]
