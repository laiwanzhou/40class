from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.imu_stage2_contracts import SequenceLengthSafetyError
from src.data.imu_stage2_io import build_stage2_schema


def _write_schema(path: Path) -> dict[str, object]:
    schema = build_stage2_schema(
        {
            "implementation_version": "fixture",
            "generator_script": "scripts/preprocess_imu_stage2.py",
            "git_commit": "0" * 40,
            "created_at": "2026-07-22T00:00:00Z",
            "source_stage1_manifest": "manifest.csv",
            "source_stage1_manifest_sha256": "a" * 64,
        }
    )
    path.write_text(json.dumps(schema), encoding="utf-8")
    return schema


def _normalization(tmp_path: Path, schema: dict[str, object]) -> tuple[Path, Path, dict[str, object]]:
    from scripts.compute_imu_normalization import StreamingMoments, write_normalization_artifacts

    moments = StreamingMoments()
    values = np.empty((2, 5, 16), dtype=np.float64)
    values[0] = 1.0
    values[1] = 3.0
    moments.update(values, np.ones((2, 5), dtype=bool))
    output = tmp_path / "normalization"
    metadata = write_normalization_artifacts(
        moments.finalize(),
        output,
        stage2_contract_sha256=str(schema["contract_sha256"]),
        training_index_sha256="b" * 64,
        train_sample_id_sha256="c" * 64,
        fold=0,
        train_users=["u1"],
        source_stage2_manifest_sha256="d" * 64,
    )
    return output / "imu_normalization.npz", output / "imu_normalization.json", metadata


def _write_action(
    path: Path,
    values: np.ndarray,
    valid_mask: np.ndarray,
    sensor_mask: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        values=values.astype(np.float32),
        valid_mask=valid_mask.astype(bool),
        sensor_mask=(
            np.ones(5, dtype=bool) if sensor_mask is None else sensor_mask.astype(bool)
        ),
        timestamps_ms=np.arange(len(values), dtype=np.int64) * 100,
    )


def _dataset(
    tmp_path: Path,
    lengths: tuple[int, ...] = (2, 3),
    *,
    hard_safety_limit_t: int = 10_000,
):
    from src.data.imu_stage2_dataset import IMUStage2Dataset

    stage2_root = tmp_path / "stage2"
    stage2_root.mkdir(parents=True)
    schema_path = stage2_root / "schema.json"
    schema = _write_schema(schema_path)
    normalization_npz, normalization_json, _ = _normalization(tmp_path, schema)
    rows = []
    for index, length in enumerate(lengths):
        values = np.full((length, 5, 16), 3.0 + index, dtype=np.float32)
        valid = np.ones((length, 5), dtype=bool)
        if index == 0:
            valid[-1] = False
            values[-1] = np.nan
        relpath = f"s{index}/imu_stage2.npz"
        _write_action(stage2_root / relpath, values, valid)
        rows.append(
            {
                "sample_id": f"s{index}",
                "stage2_npz_relpath": relpath,
                "status": "success" if valid.any() else "no_usable_grid_cells",
                "selected_for_run": True,
                "split": "train",
                "label_index": index,
            }
        )
    metadata = {
        "stage2_contract_sha256": schema["contract_sha256"],
        "training_index_sha256": "b" * 64,
        "train_sample_id_sha256": "c" * 64,
        "fold": 0,
    }
    return IMUStage2Dataset(
        pd.DataFrame(rows),
        stage2_root=stage2_root,
        stage2_schema=schema_path,
        normalization_npz=normalization_npz,
        normalization_json=normalization_json,
        training_index_metadata=metadata,
        hard_safety_limit_t=hard_safety_limit_t,
    )


def test_dataset_standardizes_only_valid_cells_and_keeps_full_length(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    sample = dataset[0]

    assert sample["values"].dtype == torch.float32
    assert sample["length"] == 2
    assert torch.allclose(sample["values"][0], torch.ones((5, 16)))
    assert torch.equal(sample["values"][1], torch.zeros((5, 16)))
    assert not sample["valid_mask"][1].any()
    assert sample["usable_sensor_mask"].all()
    assert sample["label"] == 0


def test_collate_right_pads_without_confusing_real_all_invalid_time(tmp_path: Path) -> None:
    from src.data.imu_stage2_dataset import collate_imu_stage2

    dataset = _dataset(tmp_path)
    batch = collate_imu_stage2([dataset[0], dataset[1]])

    assert batch["values"].shape == (2, 3, 5, 16)
    assert batch["lengths"].tolist() == [2, 3]
    assert batch["sequence_mask"].tolist() == [[True, True, False], [True, True, True]]
    assert not batch["valid_mask"][0, 1].any()
    assert batch["sequence_mask"][0, 1]
    assert torch.equal(batch["values"][0, 2], torch.zeros((5, 16)))
    assert not batch["valid_mask"][0, 2].any()
    assert batch["timestamps_ms"][0].tolist() == [0, 100, -1]
    assert torch.equal(batch["lengths"], batch["sequence_mask"].sum(dim=1))
    assert batch["labels"].tolist() == [0, 1]


def test_dataset_rejects_contract_limit_mismatch_and_oversized_sequence(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="hard_safety_limit_t"):
        _dataset(tmp_path / "mismatch", lengths=(2,), hard_safety_limit_t=9_999)

    dataset = _dataset(tmp_path / "oversized", lengths=(10_001,))
    with pytest.raises(SequenceLengthSafetyError):
        dataset[0]


def test_length_bucket_sampler_is_deterministic_budgeted_and_complete() -> None:
    from src.data.imu_stage2_dataset import LengthBucketBatchSampler

    lengths = [10, 11, 30, 31, 70, 500]
    kwargs = dict(
        lengths=lengths,
        bucket_boundaries=[24, 48, 64, 96],
        batch_feature_budget=2 * 31 * 5 * 16,
        maximum_batch_size=3,
        minimum_batch_size=1,
        shuffle_seed=123,
        drop_last=False,
    )
    batches = list(LengthBucketBatchSampler(**kwargs))
    repeated = list(LengthBucketBatchSampler(**kwargs))

    assert batches == repeated
    assert sorted(index for batch in batches for index in batch) == list(range(len(lengths)))
    assert len({index for batch in batches for index in batch}) == len(lengths)
    for batch in batches:
        budget = len(batch) * max(lengths[index] for index in batch) * 5 * 16
        assert budget <= kwargs["batch_feature_budget"] or len(batch) == 1
        bucket_ids = [next((i for i, boundary in enumerate(kwargs["bucket_boundaries"]) if lengths[index] <= boundary), 4) for index in batch]
        assert len(set(bucket_ids)) == 1
    assert [5] in batches

    dropped = LengthBucketBatchSampler(
        lengths=[10, 11, 12],
        bucket_boundaries=[24],
        batch_feature_budget=10_000,
        maximum_batch_size=2,
        minimum_batch_size=2,
        shuffle_seed=1,
        drop_last=True,
    )
    dropped_batches = list(dropped)
    assert sum(map(len, dropped_batches)) == 2
    assert len(dropped.omitted_indices) == 1


def test_v1_loader_config_has_no_legacy_fixed_sequence_length() -> None:
    config = yaml.safe_load(Path("configs/task03/imu_stage2_v1.yaml").read_text(encoding="utf-8"))

    assert config["config_version"] == "imu-stage2-loader-v1"
    assert config["hard_safety_limit_t"] == 10_000
    assert config["batch_feature_budget"] == 327_680
    assert "sequence_length" not in config
