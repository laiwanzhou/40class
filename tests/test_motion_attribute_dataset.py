from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.motion_attribute_dataset import (
    MotionAttributeDataset,
    compute_motion_attributes,
    fit_apply_attribute_normalization,
    resample_motion_trial,
)
from src.experiments.motion_attribute_config import load_motion_attribute_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/experiments/motion_attribute_expert.yaml"


def _pose(frame: int, moving: bool = True) -> np.ndarray:
    pose = np.zeros((17, 3), dtype=np.float64)
    pose[:, 0] = np.linspace(-0.8, 0.8, 17)
    pose[:, 1] = np.linspace(0.0, 1.6, 17)
    pose[:, 2] = np.cos(np.linspace(0.0, np.pi, 17))
    pose[1], pose[4] = [-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]
    pose[11], pose[14] = [-0.5, 1.0, 0.0], [0.5, 1.0, 0.0]
    if moving:
        pose[:, 0] += frame * 0.02
        pose[13, 1] += frame * 0.01
    return pose


def _rows(tmp_path: Path, moving: bool = True) -> pd.DataFrame:
    rows = []
    for segment, frames in ((0, [0, 1, 2]), (1, [10, 11, 12])):
        for frame in frames:
            path = tmp_path / f"pose_{frame}.json"
            path.write_text(
                json.dumps([{"keypoints": _pose(frame, moving).tolist()}]),
                encoding="utf-8",
            )
            rows.append(
                {
                    "sample_id": "sample",
                    "frame_id": frame,
                    "retained_segment_index": segment,
                    "skeleton_json_path": path.name,
                    "candidate_index": 0,
                    "use_for_frame_training": True,
                }
            )
    return pd.DataFrame(rows)


def test_resampling_masks_cross_gap_and_resets_segment_velocity(tmp_path: Path) -> None:
    result = resample_motion_trial(_rows(tmp_path), data_root=tmp_path, frames=96)

    assert result.features.shape == (96, 17, 6)
    assert result.mask.shape == (96,)
    assert result.segment_ids.shape == (96,)
    assert (~result.mask).any()
    assert torch.count_nonzero(result.features[~result.mask]) == 0
    for segment in (0, 1):
        first = torch.nonzero(result.segment_ids.eq(segment)).flatten()[0]
        assert torch.count_nonzero(result.features[first, :, 3:]) == 0
    assert result.attributes.shape == (16,)
    assert torch.isfinite(result.attributes).all()


def test_static_motion_attributes_have_zero_speed_and_full_static_fraction(
    tmp_path: Path,
) -> None:
    result = resample_motion_trial(
        _rows(tmp_path, moving=False), data_root=tmp_path, frames=96
    )
    attributes = compute_motion_attributes(result.features, result.mask)

    assert torch.allclose(attributes[2:8], torch.zeros(6), atol=1e-6)
    assert attributes[14].item() == 1.0
    assert attributes[15].item() == 0.0


def test_real_population_is_canonical_before_cache_generation() -> None:
    config = load_motion_attribute_config(CONFIG)
    train = MotionAttributeDataset(config, partition="train")
    validation = MotionAttributeDataset(config, partition="validation")

    assert len(train) == 2039
    assert len(validation) == 388
    assert train.supported_count == 1956
    assert validation.supported_count == 385
    assert set(train.labels.tolist()) == set(range(40))
    assert set(validation.labels.tolist()) == set(range(40))


def test_attribute_normalization_fits_train_supported_rows_only() -> None:
    raw = np.stack(
        (
            np.full(16, 1.0),
            np.full(16, 3.0),
            np.full(16, 100.0),
        )
    ).astype(np.float32)
    train = np.asarray([True, True, False])
    available = np.ones(3, dtype=bool)

    normalized, mean, std = fit_apply_attribute_normalization(
        raw, train_mask=train, available=available
    )

    assert np.allclose(mean, 2.0)
    assert np.allclose(std, 1.0)
    assert np.allclose(normalized[:2], [[-1.0] * 16, [1.0] * 16])
    assert np.allclose(normalized[2], 98.0)
