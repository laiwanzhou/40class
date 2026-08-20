from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image

from src.data.thermal_native_dataset import (
    ThermalNativeDataset,
    ThermalTrialRecord,
    collate_thermal_trials,
    normalized_frame_indices,
)
from src.roi.thermal_context_locator import ThermalContextLocator


def _write_frames(root: Path, count: int, *, identical: bool = False) -> Path:
    root.mkdir(parents=True)
    for index in range(count):
        value = 80 if identical else index % 255
        Image.new("RGB", (320, 240), (value, 20, 240 - value)).save(
            root / f"frame_{index + 101:06d}.jpg"
        )
    return root


def _record(path: Path | None, count: int, *, user_id: str = "user6") -> ThermalTrialRecord:
    return ThermalTrialRecord(
        sample_id=f"train__c00__{user_id}__1-1-1",
        class_id=0,
        action_name="Wash_face",
        user_id=user_id,
        trial_id="1-1-1",
        thermal_dir=path,
        directory_present=path is not None,
        usable=path is not None and count > 0,
        file_count=count,
        decodable_frame_count=count,
        distinct_frame_ratio=1.0 if count else 0.0,
    )


@pytest.mark.parametrize("frame_count", [1, 2, 8, 16, 17, 595])
def test_normalized_sampling_is_thermal_native_and_always_16(frame_count: int) -> None:
    indices = normalized_frame_indices(frame_count, num_segments=16)

    assert len(indices) == 16
    assert indices[0] == 0
    assert indices[-1] == frame_count - 1
    assert all(0 <= index < frame_count for index in indices)
    assert indices == tuple(sorted(indices))
    assert len(set(indices)) == min(frame_count, 16)


def test_singleton_and_short_trials_repeat_explicit_source_indices(tmp_path: Path) -> None:
    singleton = ThermalNativeDataset([_record(_write_frames(tmp_path / "one", 1), 1)], training=False)[0]
    short = ThermalNativeDataset([_record(_write_frames(tmp_path / "short", 2), 2)], training=False)[0]

    assert singleton["source_indices"].tolist() == [0] * 16
    assert singleton["source_unique_mask"].sum().item() == 1
    assert short["source_unique_mask"].sum().item() == 2
    assert singleton["availability"].item() is True


def test_validation_is_deterministic_and_shared_across_time(tmp_path: Path) -> None:
    record = _record(_write_frames(tmp_path / "identical", 17, identical=True), 17)
    dataset = ThermalNativeDataset([record], training=False)

    first = dataset[0]
    second = dataset[0]

    torch.testing.assert_close(first["clips"], second["clips"])
    for frame in first["clips"][1:]:
        torch.testing.assert_close(frame, first["clips"][0])
    assert first["sample_id"] == record.sample_id
    assert first["clips"].shape == (16, 3, 224, 224)


def test_training_uses_one_spatial_transform_for_the_whole_trial(tmp_path: Path) -> None:
    record = _record(_write_frames(tmp_path / "identical", 16, identical=True), 16)
    dataset = ThermalNativeDataset([record], training=True, seed=123)
    sample = dataset[0]

    for frame in sample["clips"][1:]:
        torch.testing.assert_close(frame, sample["clips"][0])


def test_missing_trial_remains_canonical_with_false_availability() -> None:
    record = _record(None, 0)
    dataset = ThermalNativeDataset([record], training=False)

    assert len(dataset) == 1
    sample = dataset[0]
    assert sample["sample_id"] == record.sample_id
    assert sample["availability"].item() is False
    assert torch.count_nonzero(sample["clips"]) == 0


def test_dataset_rejects_sealed_users_and_cross_modal_fields(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sealed"):
        ThermalNativeDataset([_record(tmp_path, 1, user_id="user4")], training=False)

    base = _record(tmp_path, 1).__dict__.copy()
    for forbidden in ("ir_frame_indices", "ir_bbox", "motion_peak_indices", "paired_frame_position"):
        payload = dict(base)
        payload[forbidden] = [0]
        with pytest.raises((TypeError, ValueError)):
            ThermalTrialRecord(**payload)


def test_collate_preserves_trial_boundaries(tmp_path: Path) -> None:
    first = ThermalNativeDataset([_record(_write_frames(tmp_path / "a", 2), 2)], training=False)[0]
    second_record = _record(_write_frames(tmp_path / "b", 8), 8, user_id="user7")
    second = ThermalNativeDataset([second_record], training=False)[0]
    batch = collate_thermal_trials([first, second])

    assert batch["clips"].shape == (2, 16, 3, 224, 224)
    assert batch["sample_ids"] == (first["sample_id"], second["sample_id"])
    assert batch["user_ids"] == ("user6", "user7")


def test_thermal_context_locator_has_audited_fallbacks() -> None:
    locator = ThermalContextLocator(confidence_threshold=0.25, area_ratio_threshold=0.05)

    accepted = locator.choose((80, 40, 240, 220, 0.7), width=320, height=240)
    low_confidence = locator.choose((80, 40, 240, 220, 0.2), width=320, height=240)
    too_small = locator.choose((1, 1, 10, 10, 0.9), width=320, height=240)

    assert accepted.route == "thermal_yolo_context"
    assert accepted.bbox_xyxy == (40, 0, 280, 240)
    assert low_confidence.route == "full_frame"
    assert low_confidence.fallback_reason == "low_confidence"
    assert too_small.fallback_reason == "small_bbox"
