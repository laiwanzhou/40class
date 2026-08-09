from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.ir_primary_full_sequence_dataset import (
    FrameBudgetBatchSampler,
    IRPrimaryFullSequenceDataset,
    QUALITY_NAMES,
    collate_full_sequences,
)


def fixture_manifest(tmp_path: Path) -> pd.DataFrame:
    image = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    mask = np.full((256, 256), 255, dtype=np.uint8)
    ir_path = tmp_path / "ir.png"
    depth_path = tmp_path / "depth.png"
    mask_path = tmp_path / "mask.png"
    assert cv2.imwrite(str(ir_path), image)
    assert cv2.imwrite(str(depth_path), image)
    assert cv2.imwrite(str(mask_path), mask)
    rows: list[dict[str, object]] = []
    for class_id in range(40):
        frames = 3 if class_id == 0 else 1
        for frame in range(frames):
            row: dict[str, object] = {
                "split": "train",
                "class_id": class_id,
                "action_name": f"Action_{class_id}",
                "sample_id": f"train__c{class_id:02d}__u__trial",
                "user_id": "u",
                "source_frame_index": frame,
                "inter_frame_delta_seconds": 0.0 if frame == 0 else 0.1,
                "temporal_valid": 1,
            }
            for view in ("ir_context", "ir_left", "ir_right", "ir_relation"):
                row[f"{view}_path"] = str(ir_path)
                row[f"{view}_effective_valid"] = 1
                row[f"{view}_reliability"] = 1.0
            for view in ("depth_context", "depth_relation"):
                row[f"{view}_ordinal_path"] = str(depth_path)
                row[f"{view}_pixel_valid_path"] = str(mask_path)
                row[f"{view}_effective_valid"] = 1
                row[f"{view}_reliability"] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def test_dataset_preserves_complete_sequence_and_fixed_depth_interface(tmp_path: Path) -> None:
    dataset = IRPrimaryFullSequenceDataset(
        fixture_manifest(tmp_path), split="train", depth_representation="raw+relative",
    )
    item = dataset[0]
    assert item["length"] == 3
    assert item["ir"].shape == (3, 4, 1, 256, 256)
    assert item["depth"].shape == (3, 2, 3, 256, 256)
    assert item["depth_pixel_valid"].shape == (3, 2, 1, 256, 256)
    assert item["frame_indices"].tolist() == [0, 1, 2]
    np.testing.assert_allclose(item["timestamps"].numpy(), [0.0, 0.1, 0.2], atol=1e-6)
    assert bool(item["relative_stats_valid"])
    assert len(item["quality"]) == len(QUALITY_NAMES)
    assert dataset.class_map_hash


def test_collate_only_pads_in_memory_and_emits_temporal_mask(tmp_path: Path) -> None:
    dataset = IRPrimaryFullSequenceDataset(
        fixture_manifest(tmp_path), split="train", depth_representation="raw",
    )
    batch = collate_full_sequences([dataset[0], dataset[1]])
    assert batch["ir"].shape[:2] == (2, 3)
    assert batch["lengths"].tolist() == [3, 1]
    assert batch["temporal_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert not batch["ir"][1, 1:].any()
    assert batch["frame_indices"][1].tolist() == [0, -1, -1]


def test_frame_budget_sampler_bounds_padded_frames_and_is_epoch_deterministic() -> None:
    sampler = FrameBudgetBatchSampler(
        [10, 20, 30, 40, 80], max_frames=100, max_samples=4,
        shuffle=True, seed=9, bucket_size=3,
    )
    sampler.set_epoch(2)
    first = list(sampler)
    sampler.set_epoch(2)
    assert list(sampler) == first
    assert sorted(index for batch in first for index in batch) == list(range(5))
    lengths = [10, 20, 30, 40, 80]
    assert all(max(lengths[index] for index in batch) * len(batch) <= 100 for batch in first)
