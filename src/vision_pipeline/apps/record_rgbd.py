"""Record a bounded live RealSense RGB-D sequence for deterministic replay.

Every frame is read and written in order (no latest-frame dropping), so the recording's
sequence gaps are the camera/driver's, not the recorder's. Capture diagnostics are
printed and saved next to the frames as ``capture.json``.

Capture-to-host latency is reported only when a frame timestamp is in librealsense's
``global_time`` domain, which the SDK maps to host realtime; it is ``null`` otherwise,
because device hardware clocks cannot be subtracted from host clocks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from vision_pipeline.sources.realsense import (
    RealSenseConfig,
    RealSenseError,
    RealSenseSource,
    sdk_global_time_age_ms,
)
from vision_pipeline.sources.rgbd_recording import RgbdRecorder

DEFAULT_SERIAL = "146322071961"


def _percentiles(values: list[float]) -> list[float] | None:
    return [float(value) for value in np.percentile(values, [50, 95])] if values else None


def record(path: Path, *, frames: int, warmup: int, serial: str | None) -> dict[str, Any]:
    source = RealSenseSource(RealSenseConfig(serial_number=serial))
    source.open()
    try:
        info = asdict(source.info)
        recorder = RgbdRecorder(path, metadata={"device": info, "recorded_by": __name__})
        for _ in range(warmup):
            source.read()
        received_ns: list[int] = []
        sequences: list[int] = []
        latencies: list[float] = []
        started = time.monotonic()
        for _ in range(frames):
            frame = source.read()
            latency = sdk_global_time_age_ms(frame, time.time_ns())
            if latency is not None:
                latencies.append(latency)
            received_ns.append(frame.color.header.received_at.nanoseconds)
            sequences.append(frame.color.header.sequence_number)
            recorder.write(frame)
        wall_s = time.monotonic() - started
    finally:
        source.close()
    receive_span_ns = received_ns[-1] - received_ns[0]
    return {
        "path": str(path),
        "device": info,
        "frames": len(received_ns),
        "warmup_frames": warmup,
        "wall_fps_including_disk": len(received_ns) / wall_s,
        "capture_receive_fps": (
            (len(received_ns) - 1) * 1e9 / receive_span_ns if receive_span_ns > 0 else None
        ),
        "color_sequence_gaps": int(np.maximum(0, np.diff(sequences) - 1).sum()),
        "sdk_global_capture_to_host_ms_p50_p95": _percentiles(latencies),
        "latency_note": (
            "librealsense global_time is the SDK's device-to-host-realtime mapping; "
            "its mapping error is not measured here"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="new directory; existing paths are refused")
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--serial", default=DEFAULT_SERIAL, help="empty string for any device")
    args = parser.parse_args(argv)
    if args.frames <= 1:
        parser.error("--frames must be at least 2")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    try:
        report = record(
            args.path, frames=args.frames, warmup=args.warmup, serial=args.serial or None
        )
    except RealSenseError as error:
        print(f"RealSense error: {error}", file=sys.stderr)
        return 2
    (args.path / "capture.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key != "device"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
