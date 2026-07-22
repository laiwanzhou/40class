from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_npz(
    path: Path,
    values: np.ndarray,
    valid_mask: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        values=values.astype(np.float32),
        sensor_mask=np.ones(5, dtype=bool),
        valid_mask=valid_mask.astype(bool),
        timestamps_ms=np.arange(len(values), dtype=np.int64) * 100,
    )


def _sample_values(base: float, length: int = 2) -> np.ndarray:
    values = np.empty((length, 5, 16), dtype=np.float32)
    for time in range(length):
        for sensor in range(5):
            values[time, sensor] = base + time * 10 + sensor + np.arange(16) / 10
    return values


def test_streaming_moments_matches_float64_population_statistics() -> None:
    from scripts.compute_imu_normalization import StreamingMoments

    first = _sample_values(1.0).astype(np.float64)
    second = _sample_values(21.0, length=1).astype(np.float64)
    first_mask = np.ones((2, 5), dtype=bool)
    second_mask = np.ones((1, 5), dtype=bool)
    moments = StreamingMoments()
    moments.update(first, first_mask)
    moments.update(second, second_mask)
    result = moments.finalize()
    expected = np.concatenate([first, second], axis=0)

    assert np.array_equal(result["count"], np.full((5, 16), 3, dtype=np.int64))
    assert np.allclose(result["mean"], expected.mean(axis=0), rtol=0, atol=1e-12)
    assert np.allclose(result["raw_std"], expected.std(axis=0, ddof=0), rtol=0, atol=1e-12)
    assert np.allclose(result["minimum"], expected.min(axis=0), rtol=0, atol=0)
    assert np.allclose(result["maximum"], expected.max(axis=0), rtol=0, atol=0)


def test_compute_normalization_uses_only_selected_train_valid_cells(tmp_path: Path) -> None:
    from scripts.compute_imu_normalization import compute_normalization

    root = tmp_path / "stage2"
    train_values = _sample_values(1.0)
    train_valid = np.ones((2, 5), dtype=bool)
    train_valid[1, 0] = False
    train_values[1, 0] = np.nan
    validation_values = _sample_values(1000.0)
    _write_npz(root / "train/imu_stage2.npz", train_values, train_valid)
    _write_npz(
        root / "validation/imu_stage2.npz",
        validation_values,
        np.ones((2, 5), dtype=bool),
    )
    index = pd.DataFrame(
        [
            {
                "sample_id": "train",
                "stage2_npz_relpath": "train/imu_stage2.npz",
                "status": "success",
                "selected_for_run": True,
                "split": "train",
            },
            {
                "sample_id": "validation",
                "stage2_npz_relpath": "validation/imu_stage2.npz",
                "status": "success",
                "selected_for_run": True,
                "split": "validation",
            },
        ]
    )

    result = compute_normalization(index, root)

    assert np.array_equal(result["count"][0], np.ones(16, dtype=np.int64))
    assert np.array_equal(result["count"][1:], np.full((4, 16), 2, dtype=np.int64))
    assert np.allclose(result["mean"][0], train_values[0, 0], rtol=0, atol=0)
    assert float(result["mean"].max()) < 100.0
    assert np.all(result["applied_scale"][0] == 1.0)
    assert np.all(result["near_constant_mask"][0])


def test_normalization_rejects_zero_count_sensor() -> None:
    from scripts.compute_imu_normalization import StreamingMoments

    moments = StreamingMoments()
    values = _sample_values(1.0).astype(np.float64)
    valid = np.ones((2, 5), dtype=bool)
    valid[:, 4] = False
    values[:, 4] = np.nan
    moments.update(values, valid)

    with pytest.raises(ValueError, match="zero count"):
        moments.finalize()


def test_normalization_artifacts_bind_contract_and_reject_tampering(tmp_path: Path) -> None:
    from scripts.compute_imu_normalization import (
        StreamingMoments,
        validate_normalization_artifacts,
        write_normalization_artifacts,
    )

    moments = StreamingMoments()
    moments.update(_sample_values(1.0).astype(np.float64), np.ones((2, 5), dtype=bool))
    statistics = moments.finalize()
    output = tmp_path / "normalization"
    metadata = write_normalization_artifacts(
        statistics,
        output,
        stage2_contract_sha256="a" * 64,
        training_index_sha256="b" * 64,
        train_sample_id_sha256="c" * 64,
        fold=0,
        train_users=["u1", "u2"],
        source_stage2_manifest_sha256="d" * 64,
    )
    loaded = validate_normalization_artifacts(
        output / "imu_normalization.npz",
        output / "imu_normalization.json",
        expected_stage2_contract_sha256="a" * 64,
        expected_training_index_sha256="b" * 64,
        expected_train_sample_id_sha256="c" * 64,
        expected_fold=0,
    )
    assert np.array_equal(loaded["count"], statistics["count"])
    assert metadata["normalization_file_sha256"]

    metadata_path = output / "imu_normalization.json"
    original_json = metadata_path.read_text(encoding="utf-8")
    for key, replacement in (
        ("fold", 1),
        ("stage2_contract_sha256", "e" * 64),
        ("training_index_sha256", "e" * 64),
        ("train_sample_id_sha256", "e" * 64),
        ("sensor_order", ["C", "RA", "LA", "RL", "LL"]),
    ):
        payload = json.loads(original_json)
        payload["contract"][key] = replacement
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError):
            validate_normalization_artifacts(
                output / "imu_normalization.npz",
                metadata_path,
                expected_stage2_contract_sha256="a" * 64,
                expected_training_index_sha256="b" * 64,
                expected_train_sample_id_sha256="c" * 64,
                expected_fold=0,
            )
        metadata_path.write_text(original_json, encoding="utf-8")

    npz_path = output / "imu_normalization.npz"
    original_npz = npz_path.read_bytes()
    npz_path.write_bytes(original_npz + b"tamper")
    with pytest.raises(ValueError, match="normalization_file_sha256"):
        validate_normalization_artifacts(
            npz_path,
            metadata_path,
            expected_stage2_contract_sha256="a" * 64,
            expected_training_index_sha256="b" * 64,
            expected_train_sample_id_sha256="c" * 64,
            expected_fold=0,
        )
