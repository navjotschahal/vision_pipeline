"""A single-writer seqlock over a fixed-size POSIX shared-memory block.

One writer publishes a fixed-size payload without ever waiting on a reader: it bumps a
leading sequence counter to odd, writes the payload, then bumps the counter to even. A
reader takes an optimistic copy and retries if the sequence changed underneath it, so a
slow or absent reader never stalls the writer and the writer never blocks on a reader.
This is the discipline design rule 4 asks for, applied across a process boundary rather
than within one process's threads.
"""

from __future__ import annotations

import struct
from multiprocessing import resource_tracker, shared_memory

_SEQUENCE_STRUCT = struct.Struct("<Q")
_HEADER_SIZE = _SEQUENCE_STRUCT.size


class SeqlockTornReadError(RuntimeError):
    """A reader could not obtain one consistent snapshot within its retry budget."""


class SeqlockWriter:
    """Creates and owns the shared-memory segment; call :meth:`close` exactly once."""

    def __init__(self, name: str, payload_size: int) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if isinstance(payload_size, bool) or not isinstance(payload_size, int):
            raise TypeError("payload_size must be an integer")
        if payload_size <= 0:
            raise ValueError("payload_size must be positive")
        self._payload_size = payload_size
        self._sequence = 0
        try:
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=_HEADER_SIZE + payload_size
            )
        except FileExistsError:
            # A previous run of this same channel crashed without unlinking. Reclaim
            # the stale segment rather than fail: nothing else has a valid reason to
            # hold a segment under this exact channel name.
            stale = shared_memory.SharedMemory(name=name, create=False)
            stale.close()
            stale.unlink()
            self._shm = shared_memory.SharedMemory(
                name=name, create=True, size=_HEADER_SIZE + payload_size
            )
        _SEQUENCE_STRUCT.pack_into(self._shm.buf, 0, 0)

    @property
    def name(self) -> str:
        return self._shm.name

    def write(self, payload: bytes | bytearray | memoryview) -> None:
        if len(payload) != self._payload_size:
            raise ValueError(
                f"payload is {len(payload)} bytes; channel expects {self._payload_size}"
            )
        self._sequence += 1
        _SEQUENCE_STRUCT.pack_into(self._shm.buf, 0, self._sequence)
        self._shm.buf[_HEADER_SIZE : _HEADER_SIZE + self._payload_size] = payload
        self._sequence += 1
        _SEQUENCE_STRUCT.pack_into(self._shm.buf, 0, self._sequence)

    def close(self) -> None:
        self._shm.close()
        self._shm.unlink()


class SeqlockReader:
    """Attaches to a segment a :class:`SeqlockWriter` already created."""

    def __init__(self, name: str, payload_size: int) -> None:
        if isinstance(payload_size, bool) or not isinstance(payload_size, int):
            raise TypeError("payload_size must be an integer")
        if payload_size <= 0:
            raise ValueError("payload_size must be positive")
        self._payload_size = payload_size
        self._shm = shared_memory.SharedMemory(name=name, create=False)
        # CPython (as of 3.12) registers every SharedMemory instance -- attached or
        # created -- with the multiprocessing resource tracker for cleanup-on-exit.
        # Only the writer that created this segment should ever unlink it; without
        # this, this reader's own tracker would unlink the segment out from under a
        # still-running writer the moment this reader process exits.
        resource_tracker.unregister(self._shm._name, "shared_memory")
        expected = _HEADER_SIZE + payload_size
        if self._shm.size != expected:
            self._shm.close()
            raise ValueError(
                f"shared memory segment {name!r} is {self._shm.size} bytes; "
                f"expected {expected} for this channel's payload size"
            )

    def read(self, *, max_attempts: int = 32) -> tuple[int, bytes] | None:
        """Return ``(sequence, payload)`` for the latest write, or ``None`` if none yet.

        Raises :class:`SeqlockTornReadError` if the writer keeps updating faster than
        ``max_attempts`` retries can outrun. That is a real condition worth surfacing,
        not one to paper over with unbounded spinning.
        """

        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int):
            raise TypeError("max_attempts must be an integer")
        if max_attempts <= 0:
            raise ValueError("max_attempts must be positive")
        for _ in range(max_attempts):
            before = _SEQUENCE_STRUCT.unpack_from(self._shm.buf, 0)[0]
            if before == 0:
                return None
            if before % 2 == 1:
                continue
            payload = bytes(self._shm.buf[_HEADER_SIZE : _HEADER_SIZE + self._payload_size])
            after = _SEQUENCE_STRUCT.unpack_from(self._shm.buf, 0)[0]
            if before == after:
                return before, payload
        raise SeqlockTornReadError(f"no consistent read after {max_attempts} attempts")

    def close(self) -> None:
        self._shm.close()


__all__ = ["SeqlockReader", "SeqlockTornReadError", "SeqlockWriter"]
