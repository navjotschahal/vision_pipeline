import numpy as np
import pytest

from vision_pipeline.image import ImageFrame, PixelFormat


def test_bgr_frame_preserves_array_without_copying() -> None:
    pixels = np.zeros((4, 6, 3), dtype=np.uint8)
    image = ImageFrame(pixels, PixelFormat.BGR8)

    assert image.data is pixels
    assert image.width == 6
    assert image.height == 4
    assert image.row_stride_bytes == pixels.strides[0]
    assert image.size_bytes == pixels.nbytes


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((4, 6), dtype=np.uint8),
        np.zeros((4, 6, 4), dtype=np.uint8),
        np.zeros((4, 6, 3), dtype=np.float32),
    ],
)
def test_bgr_frame_rejects_wrong_shape_or_dtype(pixels: np.ndarray) -> None:
    with pytest.raises(ValueError):
        ImageFrame(pixels, PixelFormat.BGR8)
