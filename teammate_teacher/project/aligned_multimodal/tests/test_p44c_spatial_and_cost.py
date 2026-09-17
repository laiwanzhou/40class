from __future__ import annotations

import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from p32_fused_data import BalancedCostBucketBatchSampler
from p44c_model import SpatialROIEncoder


def test_spatial_roi_encoder_preserves_mask_and_uses_difference_path() -> None:
    torch.manual_seed(44)
    encoder = SpatialROIEncoder(dropout=0.0)
    features = torch.randn(2, 5, 2, 3, 3, 3, 128, requires_grad=True)
    frame_mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, False, False]]
    )
    valid = frame_mask[:, :, None].expand(-1, -1, 3).clone()
    quality = valid.float() * 0.8
    output = encoder(features, valid, quality, torch.zeros_like(quality), frame_mask)
    assert output["embedding"].shape == (2, 256)
    assert output["sequence"].shape == (2, 5, 128)
    assert output["event_energy"].shape == (2, 5)
    assert torch.count_nonzero(output["sequence"][1, 3:]) == 0
    assert torch.allclose(output["event_energy"][:, 0], torch.zeros(2))
    output["embedding"].square().mean().backward()
    assert encoder.difference_project[1].weight.grad is not None
    assert torch.isfinite(encoder.difference_project[1].weight.grad).all()


def test_cost_sampler_keeps_all_samples_and_respects_padding_budgets() -> None:
    frames = [20, 40, 80, 236] * 16
    points = [100, 250, 500, 1324] * 16
    labels = [index % 4 for index in range(len(frames))]
    sampler = BalancedCostBucketBatchSampler(
        frames,
        points,
        labels,
        maximum_batch_size=32,
        frame_budget=2240,
        point_budget=11200,
        samples_per_epoch=len(frames),
        seed=44,
    )
    sampler.set_epoch(3)
    batches = list(sampler)
    assert sum(len(batch) for batch in batches) == len(frames)
    for batch in batches:
        assert len(batch) <= 32
        assert len(batch) * max(frames[index] for index in batch) <= 2240
        assert len(batch) * max(points[index] for index in batch) <= 11200
