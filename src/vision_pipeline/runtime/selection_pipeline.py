"""Live click-selection pipeline: capture -> mask/tracking -> metric geometry.

Three threads are joined only by one-value slots, so no stage can build a backlog:

- the capture thread publishes the newest aligned RGB-D frame
  (:class:`~vision_pipeline.runtime.latest_rgbd.LatestRgbdCapture`);
- the selection worker applies operator commands, runs the segmenter on the newest
  frame, and publishes a :class:`MaskOutput`;
- the geometry worker converts the newest ``TRACKING`` mask plus *that output's own*
  depth into a :class:`GeometryOutput`.

Operator clicks name the frameset they were made on. The worker keeps a short ring of
recent frames so a click is applied to exactly the image the operator saw, even though
newer frames have arrived meanwhile; a click on a frame that has aged out is rejected
with a message instead of being moved to a different image.

Displays read :attr:`SelectionPipeline.latest_mask`/``latest_geometry`` references and
never block either worker.
"""

from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from vision_pipeline.perception.objects.selected_geometry import (
    GeometryRejection,
    SelectedGeometryDiagnostics,
    SelectedGeometryError,
    SelectedGeometryEstimator,
    SelectedObjectGeometry,
)
from vision_pipeline.perception.objects.selection import (
    Click,
    SelectionSession,
    SelectionState,
    SelectionUpdate,
)
from vision_pipeline.rgbd import AlignedRgbdFrame
from vision_pipeline.sources.realsense import sdk_global_time_age_ms

from .latest_rgbd import LatestRgbdCapture, LatestSlot, SlotClosedError

_COMMAND_CAPACITY = 32
_LATENCY_HISTORY = 20_000


class CommandKind(StrEnum):
    SELECT = "select"
    REFINE = "refine"
    CLEAR = "clear"


@dataclass(frozen=True, slots=True)
class OperatorCommand:
    """An operator action; clicks are pixel indices on frameset ``frameset_id``."""

    kind: CommandKind
    frameset_id: str | None = None
    clicks: tuple[Click, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, CommandKind):
            raise TypeError("kind must be a CommandKind")
        if self.kind is CommandKind.CLEAR:
            if self.clicks:
                raise ValueError("clear takes no clicks")
        elif not self.clicks or self.frameset_id is None:
            raise ValueError(f"{self.kind.value} requires clicks and the clicked frameset_id")


@dataclass(frozen=True, slots=True)
class MaskOutput:
    """The selection decision for one frame, with the frame it was computed on.

    Latencies are ``None`` unless the pipeline runs on a live source in this process:
    ``host_receive_to_mask_ms`` uses the source's host-monotonic receive stamp and
    ``capture_to_mask_ms`` the SDK global-time capture stamp.
    """

    index: int
    frame: AlignedRgbdFrame
    update: SelectionUpdate
    worker_ms: float
    host_receive_to_mask_ms: float | None
    capture_to_mask_ms: float | None
    command_errors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class GeometryOutput:
    mask: MaskOutput
    geometry: SelectedObjectGeometry | None
    rejection: GeometryRejection | None
    rejection_message: str | None
    geometry_ms: float
    capture_to_geometry_ms: float | None
    diagnostics: SelectedGeometryDiagnostics | None


@dataclass(frozen=True, slots=True)
class LatencySummary:
    count: int
    p50: float | None
    p95: float | None
    maximum: float | None


def _summary(values: deque[float]) -> LatencySummary:
    if not values:
        return LatencySummary(0, None, None, None)
    array = np.fromiter(values, dtype=np.float64)
    p50, p95 = np.percentile(array, [50, 95])
    return LatencySummary(int(array.size), float(p50), float(p95), float(array.max()))


@dataclass(frozen=True, slots=True)
class SelectionPipelineStats:
    capture_frames_read: int
    capture_frames_overwritten: int
    capture_sequence_gaps: int
    frames_processed: int
    tracking_outputs: int
    lost_events: int
    geometry_computed: int
    geometry_rejected: dict[str, int]
    geometry_inputs_overwritten: int
    worker_ms: LatencySummary
    host_receive_to_mask_ms: LatencySummary
    capture_to_mask_ms: LatencySummary
    geometry_ms: LatencySummary
    capture_to_geometry_ms: LatencySummary


class SelectionPipeline:
    """Owns the selection and geometry worker threads for one capture."""

    def __init__(
        self,
        capture: LatestRgbdCapture,
        session: SelectionSession,
        geometry: SelectedGeometryEstimator | None,
        *,
        live_timing: bool,
        recent_frames: int = 30,
        on_mask: Callable[[MaskOutput], None] | None = None,
        on_geometry: Callable[[GeometryOutput], None] | None = None,
    ) -> None:
        if recent_frames < 1:
            raise ValueError("recent_frames must be positive")
        self._capture = capture
        self._session = session
        self._geometry = geometry
        self._live_timing = live_timing
        self._recent_capacity = recent_frames
        self._on_mask = on_mask
        self._on_geometry = on_geometry
        self._commands: deque[OperatorCommand] = deque(maxlen=_COMMAND_CAPACITY)
        self._commands_lock = threading.Lock()
        self._geometry_slot: LatestSlot[MaskOutput] = LatestSlot()
        self._stopping = threading.Event()
        self._finished = threading.Event()
        self._error: BaseException | None = None
        self._stats_lock = threading.Lock()
        self._frames_processed = 0
        self._tracking_outputs = 0
        self._lost_events = 0
        self._geometry_computed = 0
        self._geometry_rejected: dict[str, int] = {}
        self._worker_ms: deque[float] = deque(maxlen=_LATENCY_HISTORY)
        self._receive_to_mask_ms: deque[float] = deque(maxlen=_LATENCY_HISTORY)
        self._capture_to_mask_ms: deque[float] = deque(maxlen=_LATENCY_HISTORY)
        self._geometry_ms: deque[float] = deque(maxlen=_LATENCY_HISTORY)
        self._capture_to_geometry_ms: deque[float] = deque(maxlen=_LATENCY_HISTORY)
        self.latest_mask: MaskOutput | None = None
        self.latest_geometry: GeometryOutput | None = None
        self._selection_thread = threading.Thread(
            target=self._guard(self._run_selection), name="selection-worker", daemon=True
        )
        self._geometry_thread = threading.Thread(
            target=self._guard(self._run_geometry), name="geometry-worker", daemon=True
        )

    @property
    def finished(self) -> bool:
        """True once the source ended or a worker failed."""

        return self._finished.is_set()

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        self._capture.start()
        self._selection_thread.start()
        self._geometry_thread.start()

    def submit(self, command: OperatorCommand) -> None:
        """Queue an operator command; the oldest is dropped if 32 are already pending."""

        if not isinstance(command, OperatorCommand):
            raise TypeError("command must be an OperatorCommand")
        with self._commands_lock:
            self._commands.append(command)

    def wait(self, timeout: float | None = None) -> bool:
        return self._finished.wait(timeout)

    def stop(self) -> None:
        self._stopping.set()
        self._capture.stop()
        self._selection_thread.join(timeout=10)
        self._geometry_slot.close()
        self._geometry_thread.join(timeout=10)

    def stats(self) -> SelectionPipelineStats:
        capture = self._capture.stats
        with self._stats_lock:
            return SelectionPipelineStats(
                capture_frames_read=capture.frames_read,
                capture_frames_overwritten=capture.frames_overwritten,
                capture_sequence_gaps=capture.color_sequence_gaps,
                frames_processed=self._frames_processed,
                tracking_outputs=self._tracking_outputs,
                lost_events=self._lost_events,
                geometry_computed=self._geometry_computed,
                geometry_rejected=dict(self._geometry_rejected),
                geometry_inputs_overwritten=self._geometry_slot.stats.overwritten,
                worker_ms=_summary(self._worker_ms),
                host_receive_to_mask_ms=_summary(self._receive_to_mask_ms),
                capture_to_mask_ms=_summary(self._capture_to_mask_ms),
                geometry_ms=_summary(self._geometry_ms),
                capture_to_geometry_ms=_summary(self._capture_to_geometry_ms),
            )

    def _guard(self, target: Callable[[], None]) -> Callable[[], None]:
        def run() -> None:
            try:
                target()
            except BaseException as error:
                self._error = error
                self._stopping.set()
                self._geometry_slot.close(error)
            finally:
                self._finished.set()

        return run

    def _drain_commands(self) -> list[OperatorCommand]:
        with self._commands_lock:
            commands = list(self._commands)
            self._commands.clear()
        return commands

    def _run_selection(self) -> None:
        recent: OrderedDict[str, AlignedRgbdFrame] = OrderedDict()
        index = 0
        while not self._stopping.is_set():
            try:
                frame = self._capture.take(timeout=0.5)
            except SlotClosedError:
                break
            if frame is None:
                continue
            recent[frame.frameset_id] = frame
            while len(recent) > self._recent_capacity:
                recent.popitem(last=False)
            started = time.perf_counter()
            update, errors = self._apply(frame, recent)
            worker_ms = (time.perf_counter() - started) * 1e3
            receive_ms = capture_ms = None
            if self._live_timing:
                received_ns = frame.color.header.received_at.nanoseconds
                receive_ms = (time.monotonic_ns() - received_ns) / 1e6
                capture_ms = sdk_global_time_age_ms(frame, time.time_ns())
            output = MaskOutput(index, frame, update, worker_ms, receive_ms, capture_ms, errors)
            index += 1
            with self._stats_lock:
                self._frames_processed += 1
                self._worker_ms.append(worker_ms)
                if update.is_tracking:
                    self._tracking_outputs += 1
                    if receive_ms is not None:
                        self._receive_to_mask_ms.append(receive_ms)
                    if capture_ms is not None:
                        self._capture_to_mask_ms.append(capture_ms)
                if update.loss_reason is not None:
                    self._lost_events += 1
            self.latest_mask = output
            self._geometry_slot.put(output)
            if self._on_mask is not None:
                self._on_mask(output)
        self._geometry_slot.close()

    def _apply(
        self, frame: AlignedRgbdFrame, recent: OrderedDict[str, AlignedRgbdFrame]
    ) -> tuple[SelectionUpdate, tuple[str, ...]]:
        session = self._session
        errors: list[str] = []
        current: SelectionUpdate | None = None
        for command in self._drain_commands():
            if command.kind is CommandKind.CLEAR:
                session.clear()
                current = None
                continue
            assert command.frameset_id is not None
            target = recent.get(command.frameset_id)
            if target is None:
                errors.append(f"{command.kind.value} ignored: clicked frame is no longer held")
                continue
            try:
                if command.kind is CommandKind.SELECT:
                    update = session.select(target, command.clicks)
                else:
                    update = session.refine(target, command.clicks)
            except (ValueError, RuntimeError) as error:
                errors.append(f"{command.kind.value} ignored: {error}")
                continue
            current = update if target is frame else None
        if current is None:
            current = session.update(frame)
        return current, tuple(errors)

    def _run_geometry(self) -> None:
        estimator = self._geometry
        selection_id = 0
        while True:
            try:
                item = self._geometry_slot.take(timeout=0.5)
            except SlotClosedError:
                return
            if item is None:
                continue
            update = item.update
            geometry: SelectedObjectGeometry | None = None
            rejection: GeometryRejection | None = None
            message: str | None = None
            diagnostics: SelectedGeometryDiagnostics | None = None
            started = time.perf_counter()
            if estimator is not None and update.is_tracking:
                assert update.segmentation is not None
                if update.selection_id != selection_id:
                    estimator.reset()
                    selection_id = update.selection_id
                try:
                    geometry = estimator.estimate(
                        item.frame, update.segmentation, selection_id=update.selection_id
                    )
                except SelectedGeometryError as error:
                    rejection = error.reason
                    message = str(error)
                diagnostics = estimator.last_diagnostics
            geometry_ms = (time.perf_counter() - started) * 1e3
            capture_ms = (
                sdk_global_time_age_ms(item.frame, time.time_ns()) if self._live_timing else None
            )
            output = GeometryOutput(
                item, geometry, rejection, message, geometry_ms, capture_ms, diagnostics
            )
            with self._stats_lock:
                if geometry is not None:
                    self._geometry_computed += 1
                    self._geometry_ms.append(geometry_ms)
                    if capture_ms is not None and math.isfinite(capture_ms):
                        self._capture_to_geometry_ms.append(capture_ms)
                if rejection is not None:
                    key = rejection.value
                    self._geometry_rejected[key] = self._geometry_rejected.get(key, 0) + 1
            self.latest_geometry = output
            if self._on_geometry is not None:
                self._on_geometry(output)


def selection_state_of(output: MaskOutput | None) -> SelectionState:
    return SelectionState.UNSELECTED if output is None else output.update.state


__all__ = [
    "CommandKind",
    "GeometryOutput",
    "LatencySummary",
    "MaskOutput",
    "OperatorCommand",
    "SelectionPipeline",
    "SelectionPipelineStats",
    "selection_state_of",
]
