from __future__ import annotations

import cv2
import numpy as np

from depth_encoding import jet_palette_rgb
from p88_build_depth_sequence_cache import read_decoded_depth, resize_crop


def test_read_decoded_depth_preserves_jet_indices(tmp_path) -> None:
    indices = np.zeros((480, 640), dtype=np.uint8)
    indices[:240, 320:] = 17
    indices[240:, :320] = 128
    indices[240:, 320:] = 255
    rgb = jet_palette_rgb()[indices]
    path = tmp_path / "depth.png"
    assert cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    decoded, valid, repaired = read_decoded_depth(path)
    assert np.array_equal(decoded, indices)
    assert np.asarray(valid, dtype=bool).all()
    assert repaired == 0


def test_resize_crop_keeps_invalid_pixels_masked() -> None:
    depth = np.full((8, 8), 120, dtype=np.uint8)
    valid = np.ones((8, 8), dtype=np.uint8)
    valid[:4] = 0
    resized, resized_valid = resize_crop(
        depth, valid, np.asarray([0, 0, 7, 7], dtype=np.float32), 1.0, 16
    )
    assert resized.shape == (16, 16)
    assert resized_valid.shape == (16, 16)
    assert np.all(resized[~resized_valid.astype(bool)] == 0)
