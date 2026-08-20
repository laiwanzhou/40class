from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from src.data.thermal_native_dataset import ThermalNativeDataset
from src.data.thermal_v2_sampling import normalized_window_indices


def write_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    trial = tmp_path / "data/Thermal/0_Action/user1/1-1-1"
    trial.mkdir(parents=True)
    for index in range(5):
        image = np.full((48, 64, 3), index * 30, dtype=np.uint8)
        assert cv2.imwrite(str(trial / f"{index}.jpg"), image)
    records = [
        {
            "sample_id": "train__c00__user1__1-1-1",
            "class_id": 0,
            "user_id": "user1",
            "development_split": "train12",
            "thermal_relative_path": "Thermal/0_Action/user1/1-1-1",
            "usable": True,
            "context_available": True,
            "bbox_xyxy": [4, 2, 44, 42],
            "decodable_frame_count": 5,
            "file_count": 5,
            "duplicate_frame_count": 0,
            "detection_hit_ratio": 0.75,
            "median_confidence": 0.8,
            "bbox_area_ratio": 0.5,
        },
        {
            "sample_id": "train__c01__user1__missing",
            "class_id": 1,
            "user_id": "user1",
            "development_split": "train12",
            "thermal_relative_path": "Thermal/1_Action/user1/missing",
            "usable": False,
            "context_available": False,
            "decodable_frame_count": 0,
            "file_count": 0,
            "duplicate_frame_count": 0,
        },
    ]
    context = tmp_path / "context.jsonl"
    context.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    normalization = tmp_path / "normalization.json"
    normalization.write_text(json.dumps({"rgb_mean": [0.5] * 3, "rgb_std": [0.25] * 3}))
    indices = sorted({i for window in normalized_window_indices(5) for i in window})
    keys = np.asarray([f"train__c00__user1__1-1-1|{i}" for i in indices])
    pose = np.ones((len(keys), 56), np.float32)
    valid = np.ones(len(keys), np.bool_)
    cache = tmp_path / "pose.npz"
    np.savez_compressed(cache, keys=keys, pose=pose, valid=valid)
    return tmp_path / "data", context, normalization, cache


def test_dataset_preserves_canonical_unavailable_and_model_shapes(tmp_path: Path) -> None:
    data_root, context, normalization, cache = write_fixture(tmp_path)
    dataset = ThermalNativeDataset(
        data_root=data_root,
        context_path=context,
        normalization_path=normalization,
        pose_cache_path=cache,
        partition="train12",
        training=False,
    )

    usable, unavailable = dataset[0], dataset[1]
    assert len(dataset) == 2
    assert usable["full_rgb"].shape == (3, 3, 16, 160, 160)
    assert usable["crop_rgb"].shape == (3, 3, 16, 160, 160)
    assert usable["motion"].shape == (3, 16, 1, 160, 160)
    assert usable["pose"].shape == (3, 16, 56)
    assert usable["window_mask"].all()
    assert usable["loss_eligible"]
    assert torch.isfinite(usable["full_rgb"]).all()
    assert unavailable["sample_id"] == "train__c01__user1__missing"
    assert not unavailable["loss_eligible"]
    assert not unavailable["availability"].any()
    assert unavailable["full_rgb"].count_nonzero() == 0


def test_dataset_rejects_incomplete_pose_cache(tmp_path: Path) -> None:
    data_root, context, normalization, cache = write_fixture(tmp_path)
    with np.load(cache) as payload:
        np.savez_compressed(
            cache,
            keys=payload["keys"][:-1],
            pose=payload["pose"][:-1],
            valid=payload["valid"][:-1],
        )

    with pytest.raises(ValueError, match="pose cache missing"):
        ThermalNativeDataset(
            data_root=data_root,
            context_path=context,
            normalization_path=normalization,
            pose_cache_path=cache,
            partition="train12",
            training=False,
        )


def test_training_geometry_is_deterministic_within_epoch(tmp_path: Path) -> None:
    data_root, context, normalization, cache = write_fixture(tmp_path)
    dataset = ThermalNativeDataset(
        data_root=data_root,
        context_path=context,
        normalization_path=normalization,
        pose_cache_path=cache,
        partition="train12",
        training=True,
        seed=17,
    )
    dataset.set_epoch(3)
    first = dataset[0]["full_rgb"]
    second = dataset[0]["full_rgb"]
    assert torch.equal(first, second)
