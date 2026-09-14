"""Native camera capability discovery and normalized capability reporting."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class CameraProbeError(RuntimeError):
    """Camera capabilities could not be discovered."""


@dataclass(frozen=True, slots=True)
class CameraProfile:
    width: int
    height: int
    min_fps: float
    max_fps: float
    pixel_format: str

    @property
    def pixels(self) -> int:
        return self.width * self.height


@dataclass(frozen=True, slots=True)
class CameraDevice:
    index: int
    name: str
    unique_id: str
    position: str
    is_connected: bool
    is_suspended: bool
    profiles: tuple[CameraProfile, ...]


ProbeRunner = Callable[[Sequence[str]], str]


def _run_command(command: Sequence[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise CameraProbeError("xcrun/swift is required for native macOS probing") from error
    except subprocess.TimeoutExpired as error:
        raise CameraProbeError("native macOS camera probe timed out") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        raise CameraProbeError(f"native macOS camera probe failed: {detail}") from error
    return completed.stdout


def parse_camera_probe_json(output: str) -> tuple[CameraDevice, ...]:
    try:
        raw_devices: Any = json.loads(output)
        if not isinstance(raw_devices, list):
            raise TypeError("top-level value must be a list")
        devices = []
        for raw_device in raw_devices:
            if not isinstance(raw_device, dict):
                raise TypeError("each camera must be an object")
            raw_profiles = raw_device["profiles"]
            if not isinstance(raw_profiles, list):
                raise TypeError("profiles must be a list")
            profiles = tuple(
                CameraProfile(
                    width=int(profile["width"]),
                    height=int(profile["height"]),
                    min_fps=float(profile["min_fps"]),
                    max_fps=float(profile["max_fps"]),
                    pixel_format=str(profile["pixel_format"]),
                )
                for profile in raw_profiles
            )
            devices.append(
                CameraDevice(
                    index=int(raw_device["index"]),
                    name=str(raw_device["name"]),
                    unique_id=str(raw_device["unique_id"]),
                    position=str(raw_device["position"]),
                    is_connected=bool(raw_device["is_connected"]),
                    is_suspended=bool(raw_device["is_suspended"]),
                    profiles=profiles,
                )
            )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CameraProbeError(f"invalid native camera probe output: {error}") from error
    return tuple(devices)


def probe_macos_cameras(*, runner: ProbeRunner = _run_command) -> tuple[CameraDevice, ...]:
    """Enumerate native AVFoundation formats and their frame-rate ranges."""

    if sys.platform != "darwin":
        raise CameraProbeError("native AVFoundation probing is only available on macOS")
    script = Path(__file__).resolve().parents[1] / "native" / "macos_camera_probe.swift"
    if not script.is_file():
        raise CameraProbeError(f"native camera probe is missing: {script}")
    return parse_camera_probe_json(runner(("xcrun", "swift", str(script))))


def camera_by_index(devices: Sequence[CameraDevice], index: int) -> CameraDevice:
    for device in devices:
        if device.index == index:
            return device
    available = ", ".join(str(device.index) for device in devices) or "none"
    raise CameraProbeError(f"camera index {index} not found; available indexes: {available}")


def render_macos_camera_report(
    devices: Sequence[CameraDevice],
    *,
    generated_at: datetime | None = None,
) -> str:
    """Render an AVFoundation inventory as durable, human-readable Markdown."""

    timestamp = generated_at or datetime.now(UTC)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=UTC)
    lines = [
        "# macOS camera capability snapshot",
        "",
        f"Generated: `{timestamp.astimezone(UTC).isoformat()}`",
        "",
        "Probe method: macOS AVFoundation device formats and their supported " "frame-rate ranges.",
        "",
        "This is an inventory, not a recommendation. Choose one row and copy its "
        "`width`, `height`, and a supported `fps` into `configs/webcam.yaml`. OpenCV "
        "still reports the mode actually negotiated by the device.",
        "",
    ]
    if not devices:
        lines.extend(("No AVFoundation video cameras were found.", ""))
        return "\n".join(lines)

    for device in devices:
        status = "connected" if device.is_connected else "disconnected"
        if device.is_suspended:
            status += ", suspended"
        lines.extend(
            (
                f"## Device {device.index}: {device.name}",
                "",
                f"- OpenCV/AVFoundation index: `{device.index}`",
                f"- Stable native ID: `{device.unique_id}`",
                f"- Position: `{device.position}`",
                f"- Status: {status}",
                "",
                "| Width | Height | Minimum FPS | Maximum FPS | Native format |",
                "| ---: | ---: | ---: | ---: | :--- |",
            )
        )
        seen: set[tuple[int, int, float, float, str]] = set()
        for profile in sorted(
            device.profiles,
            key=lambda item: (item.width, item.height, item.max_fps, item.pixel_format),
        ):
            identity = (
                profile.width,
                profile.height,
                profile.min_fps,
                profile.max_fps,
                profile.pixel_format,
            )
            if identity in seen:
                continue
            seen.add(identity)
            lines.append(
                f"| {profile.width} | {profile.height} | {profile.min_fps:g} | "
                f"{profile.max_fps:g} | `{profile.pixel_format}` |"
            )
        if not seen:
            lines.append("| _No video profiles reported_ | | | | |")
        lines.append("")

    lines.extend(
        (
            "## Raw configuration fields",
            "",
            "```yaml",
            "webcam:",
            "  device_index: 0  # choose a device above",
            "  width: 1920      # copy from the same table row",
            "  height: 1080",
            "  fps: 30",
            "```",
            "",
            "Regenerate this snapshot after an OS update, camera firmware change, or "
            "hardware change. The native format identifies capture transport; the current "
            "OpenCV adapter emits decoded BGR8 frames in host memory.",
            "",
        )
    )
    return "\n".join(lines)
