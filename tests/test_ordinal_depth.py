from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.data.ordinal_depth import (
    JET_LUT_BGR,
    UnexpectedJetColorError,
    invert_opencv_jet_bgr,
    mask_aware_resize_ordinal,
)


def test_exact_inverse_jet_recovers_all_256_indices() -> None:
    encoded = cv2.applyColorMap(
        np.arange(256, dtype=np.uint8).reshape(16, 16),
        cv2.COLORMAP_JET,
    )
    decoded = invert_opencv_jet_bgr(encoded)
    np.testing.assert_array_equal(
        decoded.values,
        np.arange(256, dtype=np.uint8).reshape(16, 16),
    )
    assert decoded.pixel_valid.all()
    assert decoded.unexpected_count == 0


def test_black_is_invalid_but_jet_index_zero_is_valid() -> None:
    image = np.asarray([[JET_LUT_BGR[0], [0, 0, 0]]], dtype=np.uint8)
    decoded = invert_opencv_jet_bgr(image)
    np.testing.assert_array_equal(decoded.values, [[0, 0]])
    np.testing.assert_array_equal(decoded.pixel_valid, [[True, False]])


def test_unknown_non_black_color_is_rejected() -> None:
    known = {tuple(color) for color in JET_LUT_BGR.tolist()}
    unknown = next(
        np.asarray([b, g, r], dtype=np.uint8)
        for b in range(1, 8)
        for g in range(1, 8)
        for r in range(1, 8)
        if (b, g, r) not in known
    )
    with pytest.raises(UnexpectedJetColorError, match="outside the OpenCV JET LUT") as caught:
        invert_opencv_jet_bgr(unknown.reshape(1, 1, 3))
    assert caught.value.count == 1


def test_tolerated_unknown_color_remains_explicitly_invalid() -> None:
    image = np.asarray([[JET_LUT_BGR[42], [1, 2, 3]]], dtype=np.uint8)
    decoded = invert_opencv_jet_bgr(image, max_unexpected_pixels=1)
    np.testing.assert_array_equal(decoded.pixel_valid, [[True, False]])
    np.testing.assert_array_equal(decoded.unexpected_mask, [[False, True]])
    assert decoded.unexpected_count == 1


def test_mask_aware_resize_does_not_blend_invalid_zeros_into_depth() -> None:
    values = np.asarray([[200, 0], [0, 0]], dtype=np.uint8)
    valid = np.asarray([[True, False], [False, False]])
    resized, resized_valid = mask_aware_resize_ordinal(values, valid, (4, 4))
    assert resized.shape == (4, 4)
    assert resized_valid.shape == (4, 4)
    assert np.all(resized[resized_valid] == 200)
    assert np.all(resized[~resized_valid] == 0)


def test_mask_aware_downsample_uses_valid_weight_normalization() -> None:
    values = np.asarray([[160, 0], [0, 0]], dtype=np.uint8)
    valid = np.asarray([[True, False], [False, False]])
    resized, resized_valid = mask_aware_resize_ordinal(values, valid, (1, 1))
    np.testing.assert_array_equal(resized, [[160]])
    np.testing.assert_array_equal(resized_valid, [[True]])


def test_all_invalid_resize_stays_zero_and_invalid() -> None:
    resized, resized_valid = mask_aware_resize_ordinal(
        np.full((3, 5), 255, dtype=np.uint8),
        np.zeros((3, 5), dtype=bool),
        (7, 9),
    )
    assert not resized_valid.any()
    assert not resized.any()


def test_resize_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="matching 2D arrays"):
        mask_aware_resize_ordinal(
            np.zeros((2, 2), dtype=np.uint8),
            np.zeros((2, 3), dtype=bool),
            (4, 4),
        )
