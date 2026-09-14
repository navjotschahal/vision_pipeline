"""Configuration shared by DINOv2 feature-extraction implementations."""

from __future__ import annotations

from dataclasses import dataclass


class DinoV2Error(RuntimeError):
    """A DINOv2 implementation could not load or extract features."""


@dataclass(frozen=True, slots=True)
class DinoV2Config:
    """Explicit research-inference parameters for an official DINOv2 backbone."""

    model: str = "dinov2_vits14_reg"
    device: str = "cpu"
    image_size: int = 518

    def __post_init__(self) -> None:
        for name in ("model", "device"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.image_size, bool) or not isinstance(self.image_size, int):
            raise TypeError("image_size must be an integer")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")


__all__ = ["DinoV2Config", "DinoV2Error"]
