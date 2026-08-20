from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

import src.data.x3d_clip_dataset as x3d_clip_dataset
from src.data.x3d_clip_dataset import (
    IRAugmentationConfig,
    X3DClipDataset,
    _read_fixed_context_four_channel_tensor,
    endpoint_uniform_indices,
    fixed_trial_person_context_box,
    _transform_clip_frames,
    adaptive_clip_count,
    collate_x3d_clips,
    partition_trial_windows,
    select_training_window_indices,
)


def test_endpoint_uniform_indices_match_eight_probe_contract() -> None:
    assert endpoint_uniform_indices(1, count=8).tolist() == [0]
    assert endpoint_uniform_indices(13, count=8).tolist() == [0, 2, 3, 5, 7, 9, 10, 12]
    assert endpoint_uniform_indices(236, count=8).tolist() == [0, 34, 67, 101, 134, 168, 201, 235]


def test_fixed_trial_person_context_box_uses_probe_union_and_not_unprobed_outlier() -> None:
    boxes = np.full((13, 4), np.nan, dtype=np.float32)
    boxes[0] = [100.0, 80.0, 200.0, 280.0]
    boxes[2] = [120.0, 70.0, 240.0, 300.0]
    boxes[6] = [0.0, 0.0, 640.0, 480.0]
    boxes[12] = [110.0, 90.0, 220.0, 290.0]

    box = fixed_trial_person_context_box(boxes, width=640, height=480)

    np.testing.assert_allclose(box, [9.0, 24.0, 331.0, 346.0], atol=1e-5)


def test_fixed_trial_person_context_box_applies_minimum_side_and_boundary_clamp() -> None:
    boxes = np.full((20, 4), np.nan, dtype=np.float32)
    boxes[0] = [2.0, 2.0, 22.0, 42.0]
    boxes[-1] = [4.0, 4.0, 24.0, 44.0]

    box = fixed_trial_person_context_box(boxes, width=640, height=480)

    np.testing.assert_allclose(box, [0.0, 0.0, 125.0, 135.0], atol=1e-5)


def test_fixed_trial_person_context_box_forbids_full_frame_fallback() -> None:
    boxes = np.full((16, 4), np.nan, dtype=np.float32)

    with pytest.raises(ValueError, match="no valid pose probe"):
        fixed_trial_person_context_box(boxes, width=640, height=480)


@pytest.mark.parametrize(
    ("num_windows", "expected_count"),
    [(1, 1), (2, 1), (3, 2), (8, 4)],
)
def test_training_window_selection_keeps_ceil_half_in_temporal_order(
    num_windows: int, expected_count: int
) -> None:
    selected = select_training_window_indices(
        num_windows,
        keep_fraction=0.5,
        generator=torch.Generator().manual_seed(17),
    )

    assert len(selected) == expected_count
    assert selected == sorted(selected)
    assert len(set(selected)) == expected_count
    assert all(0 <= index < num_windows for index in selected)


def test_training_window_selection_changes_with_generator_seed() -> None:
    first = select_training_window_indices(
        8,
        keep_fraction=0.5,
        generator=torch.Generator().manual_seed(1),
    )
    second = select_training_window_indices(
        8,
        keep_fraction=0.5,
        generator=torch.Generator().manual_seed(2),
    )

    assert first != second


def fixture_manifest(
    tmp_path: Path,
    *,
    primary_frames: int = 3,
    split: str = "train",
) -> pd.DataFrame:
    image = np.tile(np.arange(256, dtype=np.uint8), (256, 1))
    image_path = tmp_path / "ir_context.png"
    assert cv2.imwrite(str(image_path), image)
    rows: list[dict[str, object]] = []
    for class_id in range(40):
        frame_count = primary_frames if class_id == 0 else 1
        for frame_index in range(frame_count):
            rows.append(
                {
                    "split": split,
                    "class_id": class_id,
                    "action_name": f"Action_{class_id}",
                    "sample_id": f"{split}__c{class_id:02d}__u__trial",
                    "user_id": "u",
                    "source_frame_index": frame_index,
                    "temporal_valid": 1,
                    "ir_context_path": str(image_path),
                    "ir_context_effective_valid": 1,
                    "ir_context_reliability": 1.0,
                }
            )
    return pd.DataFrame(rows)


def add_fixed_context_sources_and_cache(
    frame: pd.DataFrame, tmp_path: Path
) -> tuple[pd.DataFrame, Path]:
    enriched = frame.copy()
    enriched["source_ir_path"] = enriched["ir_context_path"]
    sample_ids: list[str] = []
    frame_keys: list[str] = []
    boxes: list[list[float]] = []
    source_depth_paths: list[str] = []
    for row in enriched.itertuples(index=False):
        key = f"{row.sample_id}_{row.source_frame_index:04d}"
        sample_ids.append(str(row.sample_id))
        frame_keys.append(key)
        boxes.append(
            [
                40.0 + row.source_frame_index,
                30.0,
                180.0 + row.source_frame_index,
                230.0,
            ]
        )
        source_depth_paths.append(str(tmp_path / f"Depth_{key}_Color.png"))
    enriched["source_depth_path"] = source_depth_paths
    pose_cache = tmp_path / "pose_cache.npz"
    np.savez_compressed(
        pose_cache,
        sample_ids=np.asarray(sample_ids),
        frame_keys=np.asarray(frame_keys),
        bbox_xyxy=np.asarray(boxes, dtype=np.float32),
        keypoints_xy=np.zeros((len(boxes), 17, 2), dtype=np.float32),
        keypoints_confidence=np.ones((len(boxes), 17), dtype=np.float32),
    )
    return enriched, pose_cache


@pytest.mark.parametrize(
    ("length", "expected"),
    [(1, 1), (13, 1), (32, 1), (33, 2), (64, 2), (65, 3), (236, 8)],
)
def test_adaptive_clip_count_boundaries(length: int, expected: int) -> None:
    assert adaptive_clip_count(length) == expected


@pytest.mark.parametrize("length", [1, 13, 32, 33, 64, 65, 236])
def test_partition_windows_cover_trial_exactly_once(length: int) -> None:
    windows = partition_trial_windows(length)
    assert len(windows) == adaptive_clip_count(length)
    assert windows[0][0] == 0
    assert windows[-1][1] == length
    assert all(start < end for start, end in windows)
    assert all(left[1] == right[0] for left, right in zip(windows, windows[1:]))
    assert max(end - start for start, end in windows) - min(
        end - start for start, end in windows
    ) <= 1


def test_dataset_groups_complete_trials_and_preserves_frame_order(tmp_path: Path) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path), split="train", training=True, seed=17
    )
    assert len(dataset) == 40
    item = dataset[0]
    assert item["clips"].shape == (1, 1, 3, 13, 182, 182)
    assert item["sample_id"] == "train__c00__u__trial"
    assert item["source_indices"].shape == (1, 1, 13)
    assert item["clip_mask"].tolist() == [True]
    assert item["window_bounds"].tolist() == [[0, 3]]
    assert item["num_frames"] == 3
    assert item["num_clips"] == 1
    assert item["class_map_hash"] == dataset.class_map_hash


def test_fixed_context_dataset_reuses_one_trial_box_for_all_sampled_frames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, pose_cache = add_fixed_context_sources_and_cache(
        fixture_manifest(tmp_path, primary_frames=20, split="val"), tmp_path
    )
    observed_boxes: list[np.ndarray] = []

    def fake_read(_path: str | Path, box: np.ndarray) -> torch.Tensor:
        observed_boxes.append(box.copy())
        return torch.zeros((1, 256, 256), dtype=torch.float32)

    monkeypatch.setattr(x3d_clip_dataset, "_read_fixed_context_gray_tensor", fake_read)
    dataset = X3DClipDataset(
        manifest,
        split="val",
        training=False,
        temporal_sampling_mode="global_single_clip",
        spatial_crop_mode="fixed_trial_person_context",
        pose_cache_path=pose_cache,
    )

    item = dataset[0]

    assert item["num_clips"] == 1
    assert observed_boxes
    for observed in observed_boxes:
        np.testing.assert_array_equal(observed, dataset.fixed_context_boxes[0])


def test_fixed_context_four_channel_reader_orders_depth_rgb_then_ir(tmp_path: Path) -> None:
    depth = np.zeros((12, 16, 3), dtype=np.uint8)
    depth[:] = [10, 20, 30]
    ir = np.full((12, 16), 40, dtype=np.uint8)
    depth_path = tmp_path / "depth.png"
    ir_path = tmp_path / "ir.png"
    Image.fromarray(depth, mode="RGB").save(depth_path)
    Image.fromarray(ir, mode="L").save(ir_path)

    tensor = _read_fixed_context_four_channel_tensor(
        depth_path, ir_path, np.asarray([0.0, 0.0, 16.0, 12.0], dtype=np.float32)
    )

    assert tensor.shape == (4, 256, 256)
    torch.testing.assert_close(
        tensor[:, 128, 128], torch.tensor([10.0, 20.0, 30.0, 40.0]) / 255.0
    )


def test_four_channel_transform_applies_photometric_change_only_to_ir() -> None:
    frame = torch.stack(
        (
            torch.full((32, 32), 0.1),
            torch.full((32, 32), 0.2),
            torch.full((32, 32), 0.3),
            torch.full((32, 32), 0.4),
        )
    )
    frames = [frame.clone() for _ in range(13)]
    plain = _transform_clip_frames(
        frames,
        training=True,
        generator=torch.Generator().manual_seed(19),
        augmentation=IRAugmentationConfig(),
    )
    augmented = _transform_clip_frames(
        frames,
        training=True,
        generator=torch.Generator().manual_seed(19),
        augmentation=IRAugmentationConfig(brightness=(1.2, 1.2)),
    )

    torch.testing.assert_close(augmented[:3], plain[:3], atol=0.0, rtol=0.0)
    assert not torch.equal(augmented[3], plain[3])


def test_dataset_normalizes_singleton_channel_grayscale_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path), split="train", training=True, seed=17
    )
    decoded = np.zeros((256, 256, 1), dtype=np.uint8)
    monkeypatch.setattr(cv2, "imread", lambda *_args, **_kwargs: decoded.copy())

    item = dataset[0]

    assert item["clips"].shape == (1, 1, 3, 13, 182, 182)


def test_dataset_rejects_sample_on_both_split_sides(tmp_path: Path) -> None:
    frame = fixture_manifest(tmp_path)
    duplicate = frame.iloc[0].to_dict()
    duplicate["split"] = "val"
    frame.loc[len(frame)] = duplicate
    with pytest.raises(ValueError, match="both train and val"):
        X3DClipDataset(frame, split="train", training=True)


def test_dataset_rejects_non_contiguous_frame_order(tmp_path: Path) -> None:
    frame = fixture_manifest(tmp_path)
    frame.loc[frame.sample_id == "train__c00__u__trial", "source_frame_index"] += 1
    with pytest.raises(ValueError, match="Non-contiguous frame order"):
        X3DClipDataset(frame, split="train", training=True)


def test_validation_emits_one_deterministic_midpoint_view(tmp_path: Path) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, split="val"), split="val", training=False
    )
    first = dataset[0]
    second = dataset[0]
    assert first["clips"].shape == (1, 1, 3, 13, 182, 182)
    torch.testing.assert_close(first["clips"], second["clips"])
    torch.testing.assert_close(first["source_indices"], second["source_indices"])


def test_236_frame_trial_uses_eight_local_windows(tmp_path: Path) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, primary_frames=236, split="val"),
        split="val",
        training=False,
    )
    item = dataset[0]
    assert item["clips"].shape == (8, 1, 3, 13, 182, 182)
    assert item["clip_mask"].sum().item() == 8
    bounds = item["window_bounds"].tolist()
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 236
    for (start, end), indices in zip(bounds, item["source_indices"][:, 0]):
        assert all(start <= index < end for index in indices.tolist())


def test_global_single_clip_samples_thirteen_frames_across_complete_trial(
    tmp_path: Path,
) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, primary_frames=236, split="val"),
        split="val",
        training=False,
        temporal_sampling_mode="global_single_clip",
    )

    item = dataset[0]

    assert dataset.num_clips[0] == 1
    assert item["num_clips"] == 1
    assert item["clips"].shape == (1, 1, 3, 13, 182, 182)
    assert item["window_bounds"].tolist() == [[0, 236]]
    indices = item["source_indices"][0, 0].tolist()
    assert len(indices) == 13
    assert indices == sorted(indices)
    assert indices[0] < 236 // 13
    assert indices[-1] >= 236 - (236 // 13 + 1)


def test_global_single_clip_training_sampling_changes_by_epoch_without_losing_span(
    tmp_path: Path,
) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, primary_frames=236),
        split="train",
        training=True,
        temporal_sampling_mode="global_single_clip",
        seed=29,
    )

    dataset.set_epoch(3)
    first = dataset[0]
    dataset.set_epoch(4)
    changed = dataset[0]

    assert first["window_bounds"].tolist() == [[0, 236]]
    assert changed["window_bounds"].tolist() == [[0, 236]]
    assert not torch.equal(first["source_indices"], changed["source_indices"])
    assert first["source_indices"][0, 0, 0] < 19
    assert first["source_indices"][0, 0, -1] >= 217


def test_dataset_rejects_unknown_temporal_sampling_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="temporal_sampling_mode"):
        X3DClipDataset(
            fixture_manifest(tmp_path),
            split="train",
            training=True,
            temporal_sampling_mode="motion_peak",
        )


def test_training_clip_dropout_keeps_half_and_changes_windows_by_epoch(
    tmp_path: Path,
) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, primary_frames=236),
        split="train",
        training=True,
        train_clip_keep_fraction=0.5,
        seed=29,
    )

    dataset.set_epoch(3)
    first = dataset[0]
    dataset.set_epoch(4)
    changed = dataset[0]

    assert dataset.num_clips[0] == 4
    assert first["num_clips"] == 4
    assert first["window_bounds"].shape == (4, 2)
    assert first["window_bounds"].tolist() == sorted(first["window_bounds"].tolist())
    assert first["window_bounds"].tolist() != changed["window_bounds"].tolist()


def test_validation_ignores_training_clip_dropout_fraction(tmp_path: Path) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, primary_frames=236, split="val"),
        split="val",
        training=False,
        train_clip_keep_fraction=0.5,
    )

    item = dataset[0]

    assert dataset.num_clips[0] == 8
    assert item["num_clips"] == 8
    assert item["window_bounds"].shape == (8, 2)


def test_training_sampling_is_epoch_deterministic(tmp_path: Path) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, primary_frames=64),
        split="train",
        training=True,
        seed=29,
    )
    dataset.set_epoch(3)
    first = dataset[0]
    dataset.set_epoch(3)
    repeated = dataset[0]
    torch.testing.assert_close(first["clips"], repeated["clips"])
    torch.testing.assert_close(first["source_indices"], repeated["source_indices"])
    dataset.set_epoch(4)
    changed = dataset[0]
    assert not torch.equal(first["source_indices"], changed["source_indices"])


def test_training_can_disable_temporal_and_spatial_augmentation(tmp_path: Path) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, primary_frames=64),
        split="train",
        training=True,
        augmentation_enabled=False,
        seed=29,
    )
    dataset.set_epoch(1)
    first = dataset[0]
    dataset.set_epoch(20)
    last = dataset[0]

    torch.testing.assert_close(first["clips"], last["clips"], atol=0.0, rtol=0.0)
    torch.testing.assert_close(first["source_indices"], last["source_indices"])


def test_spatial_transform_is_identical_for_all_frames_in_a_clip(tmp_path: Path) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, primary_frames=3),
        split="train",
        training=True,
        seed=41,
    )
    clip = dataset[0]["clips"][0, 0]
    for frame_index in range(1, 13):
        torch.testing.assert_close(clip[:, 0], clip[:, frame_index])


def test_ir_photometric_augmentation_is_seed_deterministic() -> None:
    frames = [torch.linspace(0.0, 1.0, 64).reshape(1, 8, 8) for _ in range(13)]
    augmentation = IRAugmentationConfig(
        brightness=(0.8, 1.2),
        contrast=(0.8, 1.2),
        gamma=(0.8, 1.2),
        noise_std_max=0.025,
        blur_probability=0.5,
        blur_kernel_size=3,
        blur_sigma=(0.1, 1.2),
    )

    first = _transform_clip_frames(
        frames,
        training=True,
        generator=torch.Generator().manual_seed(123),
        augmentation=augmentation,
    )
    repeated = _transform_clip_frames(
        frames,
        training=True,
        generator=torch.Generator().manual_seed(123),
        augmentation=augmentation,
    )
    changed = _transform_clip_frames(
        frames,
        training=True,
        generator=torch.Generator().manual_seed(124),
        augmentation=augmentation,
    )

    torch.testing.assert_close(first, repeated, atol=0.0, rtol=0.0)
    assert not torch.equal(first, changed)
    assert torch.isfinite(first).all()


def test_clip_consistent_ir_factors_preserve_identical_frames_without_noise() -> None:
    frame = torch.linspace(0.0, 1.0, 256).reshape(1, 16, 16)
    augmentation = IRAugmentationConfig(
        brightness=(0.7, 1.3),
        contrast=(0.7, 1.3),
        gamma=(0.7, 1.3),
        noise_std_max=0.0,
        blur_probability=1.0,
        blur_kernel_size=3,
        blur_sigma=(0.7, 0.7),
    )

    clip = _transform_clip_frames(
        [frame.clone() for _ in range(13)],
        training=True,
        generator=torch.Generator().manual_seed(7),
        augmentation=augmentation,
    )

    for frame_index in range(1, 13):
        torch.testing.assert_close(clip[:, 0], clip[:, frame_index], atol=0.0, rtol=0.0)


def test_validation_ignores_ir_augmentation_configuration() -> None:
    frames = [torch.linspace(0.0, 1.0, 256).reshape(1, 16, 16) for _ in range(13)]
    augmentation = IRAugmentationConfig(
        brightness=(0.5, 0.5),
        contrast=(0.5, 0.5),
        gamma=(1.5, 1.5),
        noise_std_max=0.1,
        blur_probability=1.0,
        blur_kernel_size=3,
        blur_sigma=(1.0, 1.0),
    )

    plain = _transform_clip_frames(
        frames,
        training=False,
        generator=torch.Generator().manual_seed(1),
    )
    configured = _transform_clip_frames(
        frames,
        training=False,
        generator=torch.Generator().manual_seed(99),
        augmentation=augmentation,
    )

    torch.testing.assert_close(plain, configured, atol=0.0, rtol=0.0)


def test_collate_pads_only_clip_dimension_and_preserves_metadata(tmp_path: Path) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, primary_frames=33, split="val"),
        split="val",
        training=False,
    )
    batch = collate_x3d_clips([dataset[0], dataset[1]])
    assert batch["clips"].shape == (2, 2, 1, 3, 13, 182, 182)
    assert batch["clip_mask"].tolist() == [[True, True], [True, False]]
    assert batch["source_indices"].shape == (2, 2, 1, 13)
    assert batch["source_indices"][1, 1].tolist() == [[-1] * 13]
    assert batch["window_bounds"][1, 1].tolist() == [-1, -1]
    assert batch["labels"].tolist() == [0, 1]
    assert batch["num_frames"].tolist() == [33, 1]
    assert batch["num_clips"].tolist() == [2, 1]
    assert batch["sample_ids"] == (
        "val__c00__u__trial",
        "val__c01__u__trial",
    )
    assert batch["quality"].shape == (2, 6)
    assert batch["quality_mask"].shape == (2, 6)
    assert batch["availability"].shape == (2, 1)
    assert batch["class_map_hash"] == dataset.class_map_hash


def test_quality_reports_repetition_and_complete_window_coverage(tmp_path: Path) -> None:
    dataset = X3DClipDataset(
        fixture_manifest(tmp_path, primary_frames=3, split="val"),
        split="val",
        training=False,
    )
    quality = dataset[0]["quality"]
    assert quality.tolist()[:3] == pytest.approx([1.0, 1.0, 1.0])
    assert quality[3].item() == pytest.approx(3 / 13)
    assert quality[4].item() == pytest.approx(1.0)
    assert quality[5].item() == pytest.approx(3 / 256)
    assert dataset[0]["quality_mask"].tolist() == [True] * 6
    assert dataset[0]["availability"].tolist() == [True]
