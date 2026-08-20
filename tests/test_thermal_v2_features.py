from __future__ import annotations

import torch

from src.data.thermal_v2_features import (
    QUALITY_NAMES,
    encode_pose_step,
    masked_stream_mean,
    signed_grayscale_differences,
)


def test_signed_grayscale_differences_preserve_direction() -> None:
    rgb = torch.zeros(3, 3, 2, 2)
    rgb[1] = 1.0
    rgb[2] = 0.25

    differences = signed_grayscale_differences(rgb)

    assert differences.shape == (3, 1, 2, 2)
    assert torch.equal(differences[0], torch.zeros_like(differences[0]))
    assert torch.allclose(differences[1], torch.ones_like(differences[1]))
    assert torch.allclose(differences[2], torch.full_like(differences[2], -0.75))


def test_pose_step_has_56_normalized_values() -> None:
    keypoints = torch.zeros(17, 3)
    keypoints[:, 0] = 160
    keypoints[:, 1] = 120
    keypoints[:, 2] = 0.8
    bbox = torch.tensor([80.0, 60.0, 240.0, 180.0, 0.9])

    encoded = encode_pose_step(keypoints, bbox, frame_size=(320, 240))

    assert encoded.shape == (56,)
    assert torch.allclose(encoded[:51].reshape(17, 3)[0], torch.tensor([0.5, 0.5, 0.8]))
    assert torch.allclose(encoded[-5:], torch.tensor([0.5, 0.5, 0.5, 0.5, 0.9]))


def test_missing_pose_is_finite_zero() -> None:
    encoded = encode_pose_step(None, None, frame_size=(320, 240))

    assert encoded.shape == (56,)
    assert torch.equal(encoded, torch.zeros(56))
    assert torch.isfinite(encoded).all()


def test_masked_mean_ignores_unavailable_steps() -> None:
    values = torch.tensor([[1.0, 3.0], [100.0, 200.0], [5.0, 7.0]])
    mask = torch.tensor([True, False, True])

    assert torch.allclose(masked_stream_mean(values, mask, dim=0), torch.tensor([3.0, 5.0]))
    assert torch.equal(
        masked_stream_mean(values, torch.zeros(3, dtype=torch.bool), dim=0),
        torch.zeros(2),
    )


def test_quality_contract_has_eight_stable_names() -> None:
    assert QUALITY_NAMES == (
        "decodable_ratio",
        "unique_sampled_ratio",
        "crop_detection_hit_ratio",
        "median_detector_confidence",
        "crop_area_ratio",
        "pose_valid_step_ratio",
        "duplicate_frame_ratio",
        "short_trial_ratio",
    )
