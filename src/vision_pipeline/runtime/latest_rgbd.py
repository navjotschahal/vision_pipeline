"""Bounded latest-value handoff between live pipeline stages.

A producer that outpaces its consumer overwrites the one pending value instead of
queueing it, so a slow consumer always works on the freshest data and memory never grows
with a backlog. Every overwrite is counted, so dropped work is measured rather than
hidden.

:class:`LatestRgbdCapture` runs a ``read()`` source (a RealSense camera or a recording) on
its own thread and publishes into such a slot.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from vision_pipeline.rgbd import AlignedRgbdFrame

ValueT = TypeVar("ValueT")


class SlotClosedError(RuntimeError):
    """The producer finished (or failed) and no pending value remains."""


@dataclass(frozen=True, slots=True)
class SlotStats:
    published: int
    taken: int
    overwritten: int


class LatestSlot(Generic[ValueT]):
    """A thread-safe single-value mailbox whose producer never blocks."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._value: ValueT | None = None
        self._pending = False
        self._closed = False
        self._error: BaseException | None = None
        self._published = 0
        self._taken = 0
        self._overwritten = 0

    @property
    def stats(self) -> SlotStats:
        with self._condition:
            return SlotStats(self._published, self._taken, self._overwritten)

    def put(self, value: ValueT) -> None:
        with self._condition:
            if self._closed:
                raise SlotClosedError("cannot publish to a closed slot")
            if self._pending:
                self._overwritten += 1
            self._value = value
            self._pending = True
            self._published += 1
            self._condition.notify_all()

    def take(self, timeout: float | None = None) -> ValueT | None:
        """Return and clear the pending value; ``None`` on timeout.

        Raises :class:`SlotClosedError` once the slot is closed and drained, chaining the
        producer's exception if it failed.
        """

        with self._condition:
            if not self._condition.wait_for(lambda: self._pending or self._closed, timeout):
                return None
            if not self._pending:
                raise SlotClosedError("producer closed the slot") from self._error
            value = self._value
            self._value = None
            self._pending = False
            self._taken += 1
            return value

    def peek(self) -> ValueT | None:
        """The pending value without consuming it, for non-blocking displays."""

        with self._condition:
            return self._value if self._pending else None

    def close(self, error: BaseException | None = None) -> None:
        with self._condition:
            self._closed = True
            if error is not None and self._error is None:
                self._error = error
            self._condition.notify_all()


class RgbdSource(Protocol):
    def read(self) -> AlignedRgbdFrame: ...


@dataclass(frozen=True, slots=True)
class CaptureStats:
    frames_read: int
    frames_overwritten: int
    color_sequence_gaps: int


class LatestRgbdCapture:
    """Read an opened source continuously on a background thread.

    Sequence gaps count color frame numbers the source skipped; overwrites count frames
    that were read but replaced before any consumer took them. The source's own
    open/close lifecycle stays with the caller. Any exception from ``read()`` (including
    a replay's end of recording) stops the thread and is re-raised, chained, by
    :meth:`take` once the last frame has been consumed.
    """

    def __init__(self, source: RgbdSource, *, thread_name: str = "rgbd-capture") -> None:
        self._source = source
        self._slot: LatestSlot[AlignedRgbdFrame] = LatestSlot()
        self._stopping = threading.Event()
        self._lock = threading.Lock()
        self._frames_read = 0
        self._sequence_gaps = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=thread_name, daemon=True)

    @property
    def stats(self) -> CaptureStats:
        with self._lock:
            return CaptureStats(
                self._frames_read, self._slot.stats.overwritten, self._sequence_gaps
            )

    @property
    def error(self) -> BaseException | None:
        return self._error

    def start(self) -> None:
        self._thread.start()

    def take(self, timeout: float | None = None) -> AlignedRgbdFrame | None:
        return self._slot.take(timeout)

    def peek(self) -> AlignedRgbdFrame | None:
        return self._slot.peek()

    def stop(self, timeout: float = 6.0) -> None:
        self._stopping.set()
        if self._thread.is_alive():
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError("RGB-D capture thread did not stop")

    def _run(self) -> None:
        previous: int | None = None
        try:
            while not self._stopping.is_set():
                frame = self._source.read()
                sequence = frame.color.header.sequence_number
                with self._lock:
                    if previous is not None:
                        self._sequence_gaps += max(0, sequence - previous - 1)
                    previous = sequence
                    self._frames_read += 1
                self._slot.put(frame)
        except BaseException as error:
            self._error = error
            self._slot.close(error)
            return
        self._slot.close()


__all__ = [
    "CaptureStats",
    "LatestRgbdCapture",
    "LatestSlot",
    "RgbdSource",
    "SlotClosedError",
    "SlotStats",
]
