from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class UnexpectedJetColorError(ValueError):
    def __init__(self, count: int, examples: np.ndarray) -> None:
        self.count = int(count)
        self.examples = np.asarray(examples, dtype=np.uint8)
        super().__init__(
            f"Depth_Color contains {self.count} non-black pixels outside the OpenCV JET LUT; "
            f"examples (BGR): {self.examples.tolist()}"
        )


@dataclass(frozen=True)
class OrdinalDepthFrame:
    values: np.ndarray
    pixel_valid: np.ndarray
    unexpected_mask: np.ndarray

    @property
    def unexpected_count(self) -> int:
        return int(np.count_nonzero(self.unexpected_mask))


def opencv_jet_lut_bgr() -> np.ndarray:
    indices = np.arange(256, dtype=np.uint8).reshape(256, 1)
    lut = cv2.applyColorMap(indices, cv2.COLORMAP_JET).reshape(256, 3)
    if len(np.unique(lut, axis=0)) != 256:
        raise RuntimeError("OpenCV JET LUT is not one-to-one")
    if np.any(np.all(lut == 0, axis=1)):
        raise RuntimeError("OpenCV JET LUT unexpectedly contains pure black")
    return lut


JET_LUT_BGR = opencv_jet_lut_bgr()


def _pack_bgr(colors: np.ndarray) -> np.ndarray:
    values = colors.astype(np.uint32, copy=False)
    return values[..., 0] | (values[..., 1] << 8) | (values[..., 2] << 16)


_JET_PACKED = _pack_bgr(JET_LUT_BGR)
_JET_SORT_ORDER = np.argsort(_JET_PACKED)
_JET_SORTED_KEYS = _JET_PACKED[_JET_SORT_ORDER]
_JET_SORTED_INDICES = np.arange(256, dtype=np.uint8)[_JET_SORT_ORDER]


def invert_opencv_jet_bgr(
    image_bgr: np.ndarray,
    *,
    max_unexpected_pixels: int = 0,
) -> OrdinalDepthFrame:
    """Recover ordinal values from an exact OpenCV JET image.

    Pure black is reserved for invalid depth. Non-black colors outside the exact
    256-color JET LUT are rejected above ``max_unexpected_pixels`` and otherwise
    remain explicitly invalid.
    """
    image = np.asarray(image_bgr)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise TypeError("image_bgr must be a uint8 array shaped [H, W, 3]")
    if max_unexpected_pixels < 0:
        raise ValueError("max_unexpected_pixels must be non-negative")

    packed = _pack_bgr(image).reshape(-1)
    locations = np.searchsorted(_JET_SORTED_KEYS, packed)
    in_range = locations < len(_JET_SORTED_KEYS)
    matched = np.zeros_like(in_range)
    matched[in_range] = _JET_SORTED_KEYS[locations[in_range]] == packed[in_range]

    black = np.all(image.reshape(-1, 3) == 0, axis=1)
    unexpected = ~black & ~matched
    unexpected_count = int(np.count_nonzero(unexpected))
    if unexpected_count > max_unexpected_pixels:
        examples = np.unique(image.reshape(-1, 3)[unexpected], axis=0)[:8]
        raise UnexpectedJetColorError(unexpected_count, examples)

    values = np.zeros(len(packed), dtype=np.uint8)
    values[matched] = _JET_SORTED_INDICES[locations[matched]]
    shape = image.shape[:2]
    return OrdinalDepthFrame(
        values=values.reshape(shape),
        pixel_valid=matched.reshape(shape),
        unexpected_mask=unexpected.reshape(shape),
    )


def load_depth_color_ordinal(
    path: str | Path,
    *,
    max_unexpected_pixels: int = 0,
) -> OrdinalDepthFrame:
    source = Path(path)
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read Depth_Color image: {source}")
    return invert_opencv_jet_bgr(image, max_unexpected_pixels=max_unexpected_pixels)


def mask_aware_resize_ordinal(
    values: np.ndarray,
    pixel_valid: np.ndarray,
    output_size: tuple[int, int],
    *,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize ordinal values without allowing invalid zeros to dilute depth.

    ``output_size`` is ``(height, width)``. Returned values are uint8 and are
    forced to zero wherever the nearest-neighbour output mask is invalid.
    """
    depth = np.asarray(values)
    valid = np.asarray(pixel_valid)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("values and pixel_valid must be matching 2D arrays")
    if len(output_size) != 2 or output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("output_size must contain positive (height, width)")
    if eps <= 0:
        raise ValueError("eps must be positive")

    output_height, output_width = (int(value) for value in output_size)
    source_height, source_width = depth.shape
    interpolation = (
        cv2.INTER_AREA
        if output_height < source_height and output_width < source_width
        else cv2.INTER_LINEAR
    )
    valid_float = valid.astype(np.float32)
    numerator = cv2.resize(
        depth.astype(np.float32) * valid_float,
        (output_width, output_height),
        interpolation=interpolation,
    )
    coverage = cv2.resize(
        valid_float,
        (output_width, output_height),
        interpolation=interpolation,
    )
    output_valid = cv2.resize(
        valid.astype(np.uint8),
        (output_width, output_height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    resized = np.divide(
        numerator,
        np.maximum(coverage, eps),
        out=np.zeros_like(numerator),
        where=coverage > eps,
    )
    resized = np.clip(np.rint(resized), 0, 255).astype(np.uint8)
    resized[~output_valid] = 0
    return resized, output_valid


def crop_and_letterbox_ordinal(
    values: np.ndarray,
    pixel_valid: np.ndarray,
    box_xyxy: np.ndarray,
    output_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop using the stored ROI and preserve its aspect ratio while resizing."""
    depth = np.asarray(values)
    valid = np.asarray(pixel_valid)
    if depth.ndim != 2 or valid.shape != depth.shape:
        raise ValueError("values and pixel_valid must be matching 2D arrays")
    box = np.asarray(box_xyxy, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError("box_xyxy must contain four finite coordinates")
    height, width = depth.shape
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    x1, x2 = sorted((int(np.clip(x1, 0, width)), int(np.clip(x2, 0, width))))
    y1, y2 = sorted((int(np.clip(y1, 0, height)), int(np.clip(y2, 0, height))))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("box_xyxy produces an empty crop")

    output_height, output_width = (int(value) for value in output_size)
    if output_height <= 0 or output_width <= 0:
        raise ValueError("output_size must contain positive (height, width)")
    crop = depth[y1:y2, x1:x2]
    crop_valid = valid[y1:y2, x1:x2]
    scale = min(output_width / crop.shape[1], output_height / crop.shape[0])
    resized_width = max(1, min(output_width, int(round(crop.shape[1] * scale))))
    resized_height = max(1, min(output_height, int(round(crop.shape[0] * scale))))
    resized, resized_valid = mask_aware_resize_ordinal(
        crop,
        crop_valid,
        (resized_height, resized_width),
    )
    output = np.zeros((output_height, output_width), dtype=np.uint8)
    output_valid = np.zeros((output_height, output_width), dtype=bool)
    left = (output_width - resized_width) // 2
    top = (output_height - resized_height) // 2
    output[top : top + resized_height, left : left + resized_width] = resized
    output_valid[top : top + resized_height, left : left + resized_width] = resized_valid
    output[~output_valid] = 0
    return output, output_valid
