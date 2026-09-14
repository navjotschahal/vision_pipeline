import numpy as np

from vision_pipeline.geometry.voxel_map import VoxelMapConfig, VoxelWorldMap
from vision_pipeline.geometry.world_visualization import (
    WorldCloudRenderer,
    WorldCloudViewConfig,
)


def test_voxel_map_fuses_repeated_points_and_bounds_memory() -> None:
    world_map = VoxelWorldMap(
        VoxelMapConfig(voxel_size_metres=0.1, max_voxels=2, update_alpha=0.5)
    )
    points = np.array(
        [[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [0.15, 0.0, 0.0]],
        dtype=np.float32,
    )
    colors = np.array([[255, 0, 0], [250, 0, 0], [0, 255, 0]], dtype=np.uint8)

    assert world_map.integrate(points, colors) == 2
    assert len(world_map) == 2
    world_map.integrate(
        np.array([[0.09, 0.01, 0.01], [0.25, 0.0, 0.0]], dtype=np.float32),
        np.array([[200, 0, 0], [0, 0, 255]], dtype=np.uint8),
    )
    xyz, rgb = world_map.snapshot()

    assert len(world_map) == 2
    assert xyz.shape == (2, 3)
    assert rgb.shape == (2, 3)
    # The oldest untouched voxel was evicted when the third voxel was added.
    assert np.any(np.isclose(xyz[:, 0], 0.25))


def test_world_cloud_renderer_produces_large_visible_canvas() -> None:
    renderer = WorldCloudRenderer(
        WorldCloudViewConfig(width=640, height=480, point_size=1)
    )
    xyz = np.array([[-0.2, 0.0, 0.0], [0.2, 0.1, 0.1]], dtype=np.float32)
    rgb = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

    canvas = renderer.render(xyz, rgb, status_lines=("tracking=normal",))

    assert canvas.shape == (480, 640, 3)
    assert canvas.dtype == np.uint8
    assert np.any(canvas != 18)
    assert renderer.handle_key(ord("c"))
