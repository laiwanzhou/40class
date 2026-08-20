from __future__ import annotations

from collections.abc import Sequence

import torch

QUALITY_NAMES = (
    "decodable_ratio",
    "unique_sampled_ratio",
    "crop_detection_hit_ratio",
    "median_detector_confidence",
    "crop_area_ratio",
    "pose_valid_step_ratio",
    "duplicate_frame_ratio",
    "short_trial_ratio",
)


def signed_grayscale_differences(rgb: torch.Tensor) -> torch.Tensor:
    if rgb.ndim != 4 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [T,3,H,W]")
    weights = rgb.new_tensor((0.299, 0.587, 0.114)).reshape(1, 3, 1, 1)
    grayscale = (rgb * weights).sum(dim=1, keepdim=True)
    differences = torch.zeros_like(grayscale)
    differences[1:] = grayscale[1:] - grayscale[:-1]
    return differences


def encode_pose_step(
    keypoints_xyc: torch.Tensor | None,
    bbox_xyxyc: torch.Tensor | None,
    frame_size: tuple[int, int],
) -> torch.Tensor:
    width, height = frame_size
    if width < 1 or height < 1:
        raise ValueError("frame dimensions must be positive")
    if keypoints_xyc is None or bbox_xyxyc is None:
        return torch.zeros(56, dtype=torch.float32)
    if keypoints_xyc.shape != (17, 3):
        raise ValueError("keypoints_xyc must have shape [17,3]")
    if bbox_xyxyc.shape != (5,):
        raise ValueError("bbox_xyxyc must have shape [5]")

    keypoints = keypoints_xyc.to(dtype=torch.float32).clone()
    keypoints[:, 0] /= width
    keypoints[:, 1] /= height
    bbox = bbox_xyxyc.to(dtype=torch.float32)
    x1, y1, x2, y2, confidence = bbox.unbind()
    box_features = torch.stack(
        (
            (x1 + x2) / (2.0 * width),
            (y1 + y2) / (2.0 * height),
            (x2 - x1) / width,
            (y2 - y1) / height,
            confidence,
        )
    )
    return torch.cat((keypoints.reshape(-1), box_features))


def masked_stream_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    if mask.ndim != 1 or mask.numel() != values.shape[dim]:
        raise ValueError("mask must be one-dimensional and match the reduced axis")
    shape = [1] * values.ndim
    shape[dim] = mask.numel()
    weights = mask.to(device=values.device, dtype=values.dtype).reshape(shape)
    numerator = (values * weights).sum(dim=dim)
    denominator = weights.sum(dim=dim).clamp_min(1.0)
    return numerator / denominator
