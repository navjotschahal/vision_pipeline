"""Camera projection models independent of a camera SDK."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PinholeIntrinsics:
    """Pinhole intrinsics at one explicit image resolution."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    calibration_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("fx", "fy", "cx", "cy"):
            value = getattr(self, name)
            if not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if self.calibration_id is not None and not self.calibration_id.strip():
            raise ValueError("calibration_id must be non-empty or None")

    @classmethod
    def from_horizontal_fov(
        cls,
        width: int,
        height: int,
        horizontal_fov_degrees: float,
    ) -> PinholeIntrinsics:
        """Construct an explicitly approximate model assuming square pixels."""

        if not math.isfinite(horizontal_fov_degrees) or not 0 < horizontal_fov_degrees < 180:
            raise ValueError("horizontal field of view must be between 0 and 180 degrees")
        focal_length = (width / 2) / math.tan(math.radians(horizontal_fov_degrees) / 2)
        return cls(
            width=width,
            height=height,
            fx=focal_length,
            fy=focal_length,
            cx=(width - 1) / 2,
            cy=(height - 1) / 2,
            calibration_id=None,
        )


__all__ = ["PinholeIntrinsics"]
