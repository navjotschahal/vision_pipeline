"""Synthetic aligned RGB-D frames with known geometry for selection tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from vision_pipeline.contracts import (
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

WIDTH = 160
HEIGHT = 120
INTRINSICS = PinholeIntrinsics(WIDTH, HEIGHT, fx=150.0, fy=150.0, cx=79.5, cy=59.5)
FRAME_ID = FrameId("camera_color_optical")
TABLE_DEPTH = 1.0
BOX_TOP_DEPTH = 0.92
BOX_RELIEF = 0.003
BOX = (60, 40, 100, 80)  # x0, y0, x1, y1 pixels
ROD = (100, 58, 140, 60)  # a two-pixel-high rod leaving the box to the right
FAR_PATCH = (40, 44, 52, 56)  # mask leakage onto a surface far behind the table
_DEVICE_CLOCK = ClockDomain("device/test-camera/global_time/run-1", ClockKind.DEVICE)
_HOST_CLOCK = ClockDomain("host/monotonic/test", ClockKind.HOST_MONOTONIC)


@dataclass(frozen=True, slots=True)
class Scene:
    frame: AlignedRgbdFrame
    box_mask: NDArray[np.bool_]
    rod_mask: NDArray[np.bool_]
    far_patch_mask: NDArray[np.bool_]
    flying_mask: NDArray[np.bool_]


def _rect(region: tuple[int, int, int, int]) -> NDArray[np.bool_]:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.bool_)
    x0, y0, x1, y1 = region
    mask[y0:y1, x0:x1] = True
    return mask


def header(
    kind: MeasurementKind, sequence: int, *, run: str = "run-1", frame_id: FrameId = FRAME_ID
) -> SampleHeader:
    stream = "color" if kind is MeasurementKind.RGB_IMAGE else "depth"
    return SampleHeader(
        sample_id=f"{run}/{stream}/{sequence}",
        source_id=f"test-camera/{stream}",
        sequence_number=sequence,
        measurement_kind=kind,
        captured_at=TimePoint(1_000_000_000 + sequence * 33_000_000, _DEVICE_CLOCK),
        received_at=TimePoint(5_000_000_000 + sequence * 33_000_000, _HOST_CLOCK),
        frame_id=frame_id,
        calibration=None,
        producer="synthetic-test-camera",
        produced_on=ComputePlacement(ComputeKind.HOST_CPU, "test"),
    )


def make_frame(
    sequence: int,
    color: NDArray[np.uint8] | None = None,
    depth: NDArray[np.float32] | None = None,
    *,
    run: str = "run-1",
) -> AlignedRgbdFrame:
    if color is None:
        color = np.full((HEIGHT, WIDTH, 3), 80, dtype=np.uint8)
        color[..., 0] = sequence % 256
    if depth is None:
        depth = np.full((HEIGHT, WIDTH), TABLE_DEPTH, dtype=np.float32)
    return AlignedRgbdFrame(
        frameset_id=f"{run}/frameset/{sequence}",
        color=SensorSample(
            header(MeasurementKind.RGB_IMAGE, sequence, run=run),
            ImageFrame(color, PixelFormat.BGR8),
            MemoryPlacement(MemoryKind.HOST),
        ),
        depth=SensorSample(
            header(MeasurementKind.DEPTH_IMAGE, sequence, run=run),
            DepthFrame(depth),
            MemoryPlacement(MemoryKind.HOST),
        ),
        color_intrinsics=INTRINSICS,
        aligned_depth_intrinsics=INTRINSICS,
        synchronization_method="synthetic",
        alignment_method="synthetic",
    )


def tabletop_scene(sequence: int = 0) -> Scene:
    """Top-down view: table at 1.0 m, an 8 cm box, a thin rod, and depth artefacts.

    The rod sits at the box-top height and is only two pixels high, so any filter that
    erodes the mask or requires wide neighbourhoods would delete it. Flying pixels are
    isolated depths halfway between the box and the table, the far patch is mask
    leakage onto a surface 0.6 m behind the table, and a few box pixels have zero depth.
    """

    depth = np.full((HEIGHT, WIDTH), TABLE_DEPTH, dtype=np.float32)
    box = _rect(BOX)
    rod = _rect(ROD)
    far = _rect(FAR_PATCH)
    # Seeded measurement noise: a perfectly flat top would give a zero-thickness PCA box.
    noise = np.random.default_rng(sequence).uniform(0.0, BOX_RELIEF, (HEIGHT, WIDTH))
    relief = (BOX_TOP_DEPTH + noise).astype(np.float32)
    depth[box] = relief[box]
    depth[rod] = relief[rod]
    depth[far] = 1.6
    flying = np.zeros_like(box)
    for x, y in ((56, 30), (104, 90), (30, 100)):
        flying[y, x] = True
    depth[flying] = 0.95
    depth[50, 70] = 0.0
    depth[52, 72] = 0.0
    color = np.full((HEIGHT, WIDTH, 3), 90, dtype=np.uint8)
    color[box | rod] = (30, 60, 200)
    return Scene(make_frame(sequence, color, depth), box, rod, far, flying)


__all__ = [
    "BOX",
    "BOX_RELIEF",
    "BOX_TOP_DEPTH",
    "FRAME_ID",
    "HEIGHT",
    "INTRINSICS",
    "ROD",
    "TABLE_DEPTH",
    "WIDTH",
    "Scene",
    "header",
    "make_frame",
    "tabletop_scene",
]
