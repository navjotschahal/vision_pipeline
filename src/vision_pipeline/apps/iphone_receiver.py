"""Live dual-socket receiver and recorder for the iPhone sensor gateway."""

from __future__ import annotations

import json
import socket
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

import numpy as np

from vision_pipeline.sources.iphone_gateway import (
    IphoneGatewayProtocolError,
    IphoneImuSample,
    IphoneImuTimeline,
    IphoneRGBDFrame,
    StreamContinuityMonitor,
    parse_imu_line,
    read_rgbd_frame,
)


@dataclass(frozen=True, slots=True)
class IphoneReceiverConfig:
    bind_host: str = "0.0.0.0"
    imu_port: int = 5001
    max_frames: int = 0
    headless: bool = False
    record_directory: Path | None = None
    minimum_depth_confidence: int = 1
    raw_cloud: bool = False
    world_cloud: bool = False
    point_stride: int = 2
    voxel_size_metres: float = 0.02
    max_world_voxels: int = 250_000

    def __post_init__(self) -> None:
        if not self.bind_host:
            raise ValueError("bind_host must be non-empty")
        if not 1 <= self.imu_port < 65_535:
            raise ValueError("imu_port must be between 1 and 65534")
        if self.max_frames < 0:
            raise ValueError("max_frames must be non-negative")
        if not 0 <= self.minimum_depth_confidence <= 2:
            raise ValueError("minimum_depth_confidence must be between 0 and 2")
        if self.point_stride <= 0:
            raise ValueError("point_stride must be positive")
        if self.voxel_size_metres <= 0:
            raise ValueError("voxel_size_metres must be positive")
        if self.max_world_voxels <= 0:
            raise ValueError("max_world_voxels must be positive")
        if self.raw_cloud and self.world_cloud:
            raise ValueError("raw_cloud and world_cloud are mutually exclusive")

    @property
    def rgbd_port(self) -> int:
        return self.imu_port + 1


class IphoneDatasetRecorder:
    """Record decoded, analysis-friendly RGB, depth, confidence, and metadata."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"record directory is not empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "rgb").mkdir()
        (self.root / "depth").mkdir()
        (self.root / "confidence").mkdir()
        self._imu_file = (self.root / "imu.ndjson").open("w", encoding="utf-8")
        self._frame_file = (self.root / "frames.ndjson").open("w", encoding="utf-8")
        self._imu_lock = threading.Lock()
        self._frame_lock = threading.Lock()
        manifest = {
            "schema": "vision_pipeline.iphone_dataset.v1",
            "created_utc": datetime.now(UTC).isoformat(),
            "time_semantics": {
                "device_timestamp_ns": "iPhone monotonic device time",
                "host_received_monotonic_ns": "host time after complete packet read",
            },
            "coordinate_conventions": {
                "camera_to_world": "ARKit column-major; +X right, +Y up, view -Z",
                "depth": "metres in optical frame; +X right, +Y down, +Z forward",
            },
        }
        (self.root / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    def record_imu(self, sample: IphoneImuSample, host_received_ns: int) -> None:
        document = sample.to_wire_dict()
        document["host_received_monotonic_ns"] = host_received_ns
        with self._imu_lock:
            self._imu_file.write(json.dumps(document, separators=(",", ":")) + "\n")

    def record_frame(
        self,
        frame: IphoneRGBDFrame,
        host_received_ns: int,
        nearest_imu_delta_ns: int | None,
    ) -> None:
        stem = f"{frame.session_id}_{frame.sequence_number:010d}"
        rgb_path = Path("rgb") / f"{stem}.jpg"
        depth_path = Path("depth") / f"{stem}.npy"
        confidence_path = Path("confidence") / f"{stem}.npy"
        with self._frame_lock:
            (self.root / rgb_path).write_bytes(frame.rgb_jpeg)
            np.save(self.root / depth_path, frame.depth_metres, allow_pickle=False)
            if frame.confidence is not None:
                np.save(self.root / confidence_path, frame.confidence, allow_pickle=False)
            metadata = frame.metadata_dict()
            metadata.update(
                {
                    "host_received_monotonic_ns": host_received_ns,
                    "nearest_imu_delta_ns": nearest_imu_delta_ns,
                    "rgb_path": str(rgb_path),
                    "depth_path": str(depth_path),
                    "confidence_path": (
                        str(confidence_path) if frame.confidence is not None else None
                    ),
                }
            )
            self._frame_file.write(json.dumps(metadata, separators=(",", ":")) + "\n")
            self._frame_file.flush()

    def close(self) -> None:
        with self._imu_lock:
            self._imu_file.flush()
            self._imu_file.close()
        with self._frame_lock:
            self._frame_file.flush()
            self._frame_file.close()

    def __enter__(self) -> IphoneDatasetRecorder:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


@dataclass(slots=True)
class _ReceiverCounters:
    imu_received: int = 0
    first_imu_received_ns: int | None = None
    last_imu_received_ns: int | None = None
    imu_sequence_drops: int = 0
    imu_order_errors: int = 0
    frames_received: int = 0
    first_frame_received_ns: int | None = None
    last_frame_received_ns: int | None = None
    frame_sequence_drops: int = 0
    frame_order_errors: int = 0
    frame_timestamp_gaps: int = 0


class _ConnectionSlot:
    """Allow the owner thread to interrupt a worker's blocking socket read."""

    def __init__(self) -> None:
        self._connection: socket.socket | None = None
        self._lock = threading.Lock()

    def set(self, connection: socket.socket) -> None:
        with self._lock:
            self._connection = connection

    def clear(self, connection: socket.socket) -> None:
        with self._lock:
            if self._connection is connection:
                self._connection = None

    def close(self) -> None:
        with self._lock:
            connection = self._connection
            self._connection = None
        if connection is not None:
            with suppress(OSError):
                connection.shutdown(socket.SHUT_RDWR)
            connection.close()


def _listening_socket(host: str, port: int) -> socket.socket:
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, port))
    listener.listen(1)
    listener.settimeout(0.5)
    return listener


def _observed_rate(count: int, first_ns: int | None, last_ns: int | None) -> float:
    if count < 2 or first_ns is None or last_ns is None or last_ns <= first_ns:
        return 0.0
    return (count - 1) / ((last_ns - first_ns) / 1e9)


def _accept_until_stopped(
    listener: socket.socket, stop: threading.Event
) -> tuple[socket.socket, tuple[str, int]] | None:
    while not stop.is_set():
        try:
            return listener.accept()
        except TimeoutError:
            continue
    return None


def _receive_imu(
    listener: socket.socket,
    stop: threading.Event,
    timeline: IphoneImuTimeline,
    counters: _ReceiverCounters,
    counter_lock: threading.Lock,
    recorder: IphoneDatasetRecorder | None,
    connection_slot: _ConnectionSlot,
) -> None:
    monitor = StreamContinuityMonitor()
    accepted = _accept_until_stopped(listener, stop)
    if accepted is None:
        return
    connection, address = accepted
    connection_slot.set(connection)
    print(f"IMU connected from {address[0]}:{address[1]}")
    try:
        with connection, connection.makefile("rb") as stream:
            for line in stream:
                if stop.is_set():
                    break
                received_ns = time.monotonic_ns()
                try:
                    sample = parse_imu_line(line)
                except IphoneGatewayProtocolError as error:
                    print(f"IMU protocol error: {error}")
                    continue
                observation = monitor.observe(
                    sample.session_id, sample.sequence_number, sample.device_timestamp_ns
                )
                timeline.append(sample)
                if recorder is not None:
                    recorder.record_imu(sample, received_ns)
                with counter_lock:
                    if counters.first_imu_received_ns is None:
                        counters.first_imu_received_ns = received_ns
                    counters.last_imu_received_ns = received_ns
                    counters.imu_received += 1
                    counters.imu_sequence_drops += observation.dropped
                    counters.imu_order_errors += int(
                        observation.duplicate_or_reordered
                        or observation.timestamp_regressed
                    )
    except OSError as error:
        if not stop.is_set():
            print(f"IMU connection error: {error}")
    finally:
        connection_slot.clear(connection)
        print("IMU connection closed")


def _decode_rgb(frame: IphoneRGBDFrame) -> np.ndarray:
    import cv2

    encoded = np.frombuffer(frame.rgb_jpeg, dtype=np.uint8)
    rgb = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if rgb is None or (rgb.shape[1], rgb.shape[0]) != (frame.rgb_width, frame.rgb_height):
        raise IphoneGatewayProtocolError("JPEG dimensions do not match RGB-D header")
    return rgb


def _decode_preview(frame: IphoneRGBDFrame, status: tuple[str, ...]) -> np.ndarray:
    import cv2

    rgb = _decode_rgb(frame)
    valid = np.isfinite(frame.depth_metres) & (frame.depth_metres > 0)
    normalized = np.zeros(frame.depth_metres.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(frame.depth_metres, 0.15, 5.0)
        normalized[valid] = np.asarray(
            255.0 * (1.0 - (clipped[valid] - 0.15) / (5.0 - 0.15)),
            dtype=np.uint8,
        )
    depth_color = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    depth_color[~valid] = 0
    depth_color = cv2.resize(
        depth_color, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST
    )
    preview_width = 720
    scale = preview_width / rgb.shape[1]
    preview_size = (preview_width, round(rgb.shape[0] * scale))
    rgb = cv2.resize(rgb, preview_size, interpolation=cv2.INTER_AREA)
    depth_color = cv2.resize(depth_color, preview_size, interpolation=cv2.INTER_NEAREST)
    canvas = np.hstack((rgb, depth_color))
    for index, line in enumerate(status):
        cv2.putText(
            canvas,
            line,
            (15, 30 + 27 * index),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "RGB (native buffer orientation) | LiDAR depth     Q: quit",
        (15, canvas.shape[0] - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return canvas


def _depth_aligned_colors(
    frame: IphoneRGBDFrame,
    rgb_bgr: np.ndarray,
    *,
    sampling_stride: int,
    minimum_confidence: int,
) -> np.ndarray:
    """Select RGB colors in the same row-major order as frame.world_points."""

    import cv2

    depth_height, depth_width = frame.depth_metres.shape
    aligned_bgr = cv2.resize(
        rgb_bgr, (depth_width, depth_height), interpolation=cv2.INTER_AREA
    )
    sampled_depth = frame.depth_metres[::sampling_stride, ::sampling_stride]
    valid = (
        np.isfinite(sampled_depth)
        & (sampled_depth >= 0.05)
        & (sampled_depth <= 8.0)
    )
    if frame.confidence is not None:
        valid &= (
            frame.confidence[::sampling_stride, ::sampling_stride] >= minimum_confidence
        )
    sampled_rgb = aligned_bgr[::sampling_stride, ::sampling_stride, ::-1]
    return np.ascontiguousarray(sampled_rgb[valid], dtype=np.uint8)


def run_iphone_receiver(config: IphoneReceiverConfig) -> int:
    """Receive both streams until Q, Ctrl-C, disconnect, or max_frames."""

    stop = threading.Event()
    timeline = IphoneImuTimeline()
    counters = _ReceiverCounters()
    counter_lock = threading.Lock()
    recorder = (
        IphoneDatasetRecorder(config.record_directory)
        if config.record_directory is not None
        else None
    )
    imu_connection_slot = _ConnectionSlot()
    imu_listener = _listening_socket(config.bind_host, config.imu_port)
    rgbd_listener = _listening_socket(config.bind_host, config.rgbd_port)
    imu_thread = threading.Thread(
        target=_receive_imu,
        name="iphone-imu-receiver",
        args=(
            imu_listener,
            stop,
            timeline,
            counters,
            counter_lock,
            recorder,
            imu_connection_slot,
        ),
        daemon=True,
    )
    imu_thread.start()
    print(
        f"listening for iPhone IMU on {config.bind_host}:{config.imu_port} "
        f"and RGB-D on {config.bind_host}:{config.rgbd_port}"
    )

    frame_monitor = StreamContinuityMonitor()
    previous_frame_timestamp_ns: int | None = None
    world_map = None
    world_renderer = None
    world_session_id: str | None = None
    if config.raw_cloud or config.world_cloud:
        from vision_pipeline.geometry.voxel_map import VoxelMapConfig, VoxelWorldMap
        from vision_pipeline.geometry.world_visualization import (
            WorldCloudRenderer,
            WorldCloudViewConfig,
        )

        world_map = VoxelWorldMap(
            VoxelMapConfig(
                voxel_size_metres=config.voxel_size_metres,
                max_voxels=config.max_world_voxels,
            )
        )
        world_renderer = WorldCloudRenderer(WorldCloudViewConfig())
    try:
        accepted = _accept_until_stopped(rgbd_listener, stop)
        if accepted is None:
            return 0
        connection, address = accepted
        print(f"RGB-D connected from {address[0]}:{address[1]}")
        with connection, connection.makefile("rb") as stream:
            frame_stream: BinaryIO = stream
            while not stop.is_set():
                try:
                    frame = read_rgbd_frame(frame_stream)
                except EOFError:
                    print("RGB-D connection closed")
                    break
                received_ns = time.monotonic_ns()
                observation = frame_monitor.observe(
                    frame.session_id, frame.sequence_number, frame.device_timestamp_ns
                )
                timestamp_gap = (
                    previous_frame_timestamp_ns is not None
                    and frame.device_timestamp_ns - previous_frame_timestamp_ns > 200_000_000
                )
                previous_frame_timestamp_ns = frame.device_timestamp_ns
                _nearest_imu, imu_delta_ns = timeline.nearest(frame)
                world_points = frame.world_points(
                    sampling_stride=config.point_stride,
                    minimum_confidence=config.minimum_depth_confidence,
                )
                displayed_xyz = np.empty((0, 3), dtype=np.float32)
                displayed_colors = np.empty((0, 3), dtype=np.uint8)
                raw_rgb_bgr: np.ndarray | None = None
                if config.raw_cloud:
                    raw_rgb_bgr = _decode_rgb(frame)
                    displayed_xyz = frame.optical_points(
                        sampling_stride=config.point_stride,
                        minimum_confidence=config.minimum_depth_confidence,
                    )
                    displayed_colors = _depth_aligned_colors(
                        frame,
                        raw_rgb_bgr,
                        sampling_stride=config.point_stride,
                        minimum_confidence=config.minimum_depth_confidence,
                    )
                if world_map is not None:
                    if frame.session_id != world_session_id:
                        world_map.clear()
                        world_session_id = frame.session_id
                    if frame.tracking_state == "normal":
                        rgb_bgr = raw_rgb_bgr if raw_rgb_bgr is not None else _decode_rgb(frame)
                        colors = _depth_aligned_colors(
                            frame,
                            rgb_bgr,
                            sampling_stride=config.point_stride,
                            minimum_confidence=config.minimum_depth_confidence,
                        )
                        if len(colors) != len(world_points):
                            raise IphoneGatewayProtocolError(
                                "world points and aligned RGB colors differ in length"
                            )
                        world_map.integrate(world_points, colors)
                    displayed_xyz, displayed_colors = world_map.snapshot()
                with counter_lock:
                    if counters.first_frame_received_ns is None:
                        counters.first_frame_received_ns = received_ns
                    counters.last_frame_received_ns = received_ns
                    counters.frames_received += 1
                    counters.frame_sequence_drops += observation.dropped
                    counters.frame_order_errors += int(
                        observation.duplicate_or_reordered
                        or observation.timestamp_regressed
                    )
                    counters.frame_timestamp_gaps += int(timestamp_gap)
                    frame_count = counters.frames_received
                    imu_count = counters.imu_received
                    frame_rate = _observed_rate(
                        frame_count,
                        counters.first_frame_received_ns,
                        counters.last_frame_received_ns,
                    )
                    imu_rate = _observed_rate(
                        imu_count,
                        counters.first_imu_received_ns,
                        counters.last_imu_received_ns,
                    )
                if recorder is not None:
                    recorder.record_frame(frame, received_ns, imu_delta_ns)

                alignment = (
                    "unavailable"
                    if imu_delta_ns is None
                    else f"{imu_delta_ns / 1e6:+.2f} ms"
                )
                position = frame.camera_position_world_metres
                status = (
                    f"session={frame.session_id[:8]} frame={frame.sequence_number} "
                    f"tracking={frame.tracking_state}",
                    f"RGB={frame.rgb_width}x{frame.rgb_height} "
                    f"depth={frame.depth_metres.shape[1]}x{frame.depth_metres.shape[0]}",
                    f"received RGB-D={frame_rate:.1f} Hz "
                    f"IMU={imu_rate:.1f} Hz "
                    f"nearest IMU={alignment}",
                    f"world points={len(world_points)} camera xyz="
                    f"({position[0]:+.2f}, {position[1]:+.2f}, {position[2]:+.2f}) m",
                )
                if config.headless:
                    if frame_count == 1 or frame_count % 10 == 0:
                        print(" | ".join(status))
                else:
                    import cv2

                    if world_renderer is not None:
                        raw_mode = config.raw_cloud
                        canvas = world_renderer.render(
                            displayed_xyz,
                            displayed_colors,
                            status_lines=(
                                f"session={frame.session_id[:8]} "
                                f"tracking={frame.tracking_state}",
                                f"frame={frame.sequence_number} "
                                f"current points={len(displayed_xyz)} "
                                + (
                                    "no accumulation / no voxelization"
                                    if raw_mode
                                    else f"voxel={config.voxel_size_metres * 100:.1f} cm"
                                ),
                                f"camera xyz=({position[0]:+.2f}, "
                                f"{position[1]:+.2f}, {position[2]:+.2f}) m "
                                f"nearest IMU={alignment}",
                            ),
                            title=(
                                "RAW CURRENT-FRAME OPTICAL CLOUD"
                                if raw_mode
                                else "ACCUMULATED ARKIT WORLD CLOUD"
                            ),
                            vertical_axis_up=not raw_mode,
                        )
                        cv2.imshow(
                            (
                                "iPhone raw point cloud"
                                if raw_mode
                                else "iPhone live world cloud"
                            ),
                            canvas,
                        )
                    else:
                        cv2.imshow("iPhone RGB-D receiver", _decode_preview(frame, status))
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        stop.set()
                    elif (
                        world_renderer is not None
                        and world_map is not None
                        and world_renderer.handle_key(key)
                    ):
                        world_map.clear()
                if config.max_frames and frame_count >= config.max_frames:
                    stop.set()
    except KeyboardInterrupt:
        stop.set()
    except (IphoneGatewayProtocolError, OSError) as error:
        print(f"iPhone receiver error: {error}")
        return 2
    finally:
        stop.set()
        imu_connection_slot.close()
        imu_listener.close()
        rgbd_listener.close()
        imu_thread.join(timeout=2)
        if recorder is not None:
            recorder.close()
        if not config.headless:
            try:
                import cv2

                cv2.destroyAllWindows()
            except ModuleNotFoundError:
                pass

    frame_rate = _observed_rate(
        counters.frames_received,
        counters.first_frame_received_ns,
        counters.last_frame_received_ns,
    )
    imu_rate = _observed_rate(
        counters.imu_received,
        counters.first_imu_received_ns,
        counters.last_imu_received_ns,
    )
    print(
        f"received {counters.frames_received} RGB-D frames "
        f"({frame_rate:.2f} fps) and "
        f"{counters.imu_received} IMU samples "
        f"({imu_rate:.2f} Hz)"
    )
    print(
        f"continuity: frame sequence gaps={counters.frame_sequence_drops}, "
        f"frame timestamp gaps>200ms={counters.frame_timestamp_gaps}, "
        f"frame order errors={counters.frame_order_errors}, "
        f"IMU sequence gaps={counters.imu_sequence_drops}, "
        f"IMU order errors={counters.imu_order_errors}"
    )
    if recorder is not None:
        print(f"recorded dataset: {recorder.root}")
    return 0


__all__ = ["IphoneDatasetRecorder", "IphoneReceiverConfig", "run_iphone_receiver"]
