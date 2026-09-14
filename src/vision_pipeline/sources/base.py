"""Stable source boundary that Python, C++, and vendor adapters can implement."""

from __future__ import annotations

from typing import Protocol, TypeVar

from vision_pipeline.contracts import SensorSample

PayloadT = TypeVar("PayloadT")


class SensorSource(Protocol[PayloadT]):
    """A pull-based physical measurement source."""

    def open(self) -> None:
        """Acquire the underlying device."""

    def read(self) -> SensorSample[PayloadT]:
        """Return the next physical measurement."""

    def close(self) -> None:
        """Release the underlying device; repeated calls are safe."""
