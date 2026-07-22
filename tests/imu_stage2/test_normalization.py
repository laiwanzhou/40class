from __future__ import annotations

import hashlib
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


def _pipeline_contract_fixture(tmp_path: Path) -> dict[str, Path]:
    from scripts.build_imu_training_index import generate_training_index_artifacts
    from src.data.imu_stage2_io import build_stage2_schema

    stage2_root = tmp_path / "stage2"
    stage2_root.mkdir()
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "train",
                "class_id": "10",
                "class_name": "Ten",
                "user_id": "u_train",
                "action_id": "a1",
                "stage2_npz_relpath": "train/imu_stage2.npz",
                "status": "success",
                "imu_usable": "True",
                "sensor_mask": "[True, True, True, True, True]",
                "usable_sensor_mask": "[True, True, True, True, True]",
            },
            {
                "sample_id": "validation",
                "class_id": "30",
                "class_name": "Thirty",
                "user_id": "u_val",
                "action_id": "a2",
                "stage2_npz_relpath": "validation/imu_stage2.npz",
                "status": "success",
                "imu_usable": "True",
                "sensor_mask": "[True, True, True, True, True]",
                "usable_sensor_mask": "[True, True, True, True, True]",
            },
        ]
    )
    manifest_path = stage2_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False, encoding="utf-8-sig")
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
    schema_path = stage2_root / "schema.json"
    schema_path.write_text(json.dumps(schema), encoding="utf-8")
    _write_npz(
        stage2_root / "train" / "imu_stage2.npz",
        _sample_values(1.0),
        np.ones((2, 5), dtype=bool),
    )
    _write_npz(
        stage2_root / "validation" / "imu_stage2.npz",
        _sample_values(100.0),
        np.ones((2, 5), dtype=bool),
    )
    split_path = tmp_path / "fold.json"
    split_path.write_text(
        json.dumps(
            {"fold": 0, "train_users": ["u_train"], "val_users": ["u_val"]}
        ),
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"
    generate_training_index_artifacts(
        manifest_path,
        index_dir,
        split_path,
        repository_root=tmp_path,
    )
    return {
        "stage2_root": stage2_root,
        "manifest": manifest_path,
        "schema": schema_path,
        "split": split_path,
        "class_order": index_dir / "class_order.json",
        "index": index_dir / "training_index.csv",
        "metadata": index_dir / "training_index.json",
    }


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
        expected_train_users=["u1", "u2"],
        expected_source_stage2_manifest_sha256="d" * 64,
    )
    assert np.array_equal(loaded["count"], statistics["count"])
    assert metadata["normalization_file_sha256"]
    assert len(metadata["provenance"]["git_commit"]) == 40

    wrong_list = json.loads(original_json) if 'original_json' in locals() else json.loads(
        (output / "imu_normalization.json").read_text(encoding="utf-8")
    )
    wrong_list["near_constant_features"] = ["LL/not_a_feature"]
    (output / "imu_normalization.json").write_text(json.dumps(wrong_list), encoding="utf-8")
    with pytest.raises(ValueError, match="near_constant_features"):
        validate_normalization_artifacts(
            output / "imu_normalization.npz",
            output / "imu_normalization.json",
            expected_stage2_contract_sha256="a" * 64,
            expected_training_index_sha256="b" * 64,
            expected_train_sample_id_sha256="c" * 64,
            expected_fold=0,
            expected_train_users=["u1", "u2"],
            expected_source_stage2_manifest_sha256="d" * 64,
        )
    (output / "imu_normalization.json").write_text(json.dumps(metadata), encoding="utf-8")

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
                expected_train_users=["u1", "u2"],
                expected_source_stage2_manifest_sha256="d" * 64,
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
            expected_train_users=["u1", "u2"],
            expected_source_stage2_manifest_sha256="d" * 64,
        )


def test_normalization_metadata_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    from scripts.compute_imu_normalization import (
        StreamingMoments,
        validate_normalization_artifacts,
        write_normalization_artifacts,
    )

    moments = StreamingMoments()
    moments.update(_sample_values(1.0).astype(np.float64), np.ones((2, 5), dtype=bool))
    output = tmp_path / "normalization"
    write_normalization_artifacts(
        moments.finalize(),
        output,
        stage2_contract_sha256="a" * 64,
        training_index_sha256="b" * 64,
        train_sample_id_sha256="c" * 64,
        fold=0,
        train_users=["u1"],
        source_stage2_manifest_sha256="d" * 64,
    )
    metadata_path = output / "imu_normalization.json"
    text = metadata_path.read_text(encoding="utf-8")
    metadata_path.write_text(
        text.replace('"contract":', '"contract": {}, "contract":', 1),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        validate_normalization_artifacts(
            output / "imu_normalization.npz",
            metadata_path,
            expected_stage2_contract_sha256="a" * 64,
            expected_training_index_sha256="b" * 64,
            expected_train_sample_id_sha256="c" * 64,
            expected_fold=0,
            expected_train_users=["u1"],
            expected_source_stage2_manifest_sha256="d" * 64,
        )


def test_normalization_rejects_rehashed_incompatible_v1_contract(tmp_path: Path) -> None:
    from scripts.compute_imu_normalization import (
        StreamingMoments,
        validate_normalization_artifacts,
        write_normalization_artifacts,
    )
    from src.data.imu_stage2_contracts import canonical_json_bytes

    moments = StreamingMoments()
    moments.update(_sample_values(1.0).astype(np.float64), np.ones((2, 5), dtype=bool))
    output = tmp_path / "normalization"
    write_normalization_artifacts(
        moments.finalize(),
        output,
        stage2_contract_sha256="a" * 64,
        training_index_sha256="b" * 64,
        train_sample_id_sha256="c" * 64,
        fold=0,
        train_users=["u1", "u2"],
        source_stage2_manifest_sha256="d" * 64,
    )
    metadata_path = output / "imu_normalization.json"
    original = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key, replacement in (
        ("normalization_version", "imu-normalization-v2"),
        ("shape", [16, 5]),
        ("ddof", 1),
        ("near_constant_threshold", 1e-5),
        ("statistics_dtype", "float64"),
        ("train_users", ["other"]),
    ):
        payload = json.loads(json.dumps(original))
        payload["contract"][key] = replacement
        payload["normalization_contract_sha256"] = hashlib.sha256(
            canonical_json_bytes(payload["contract"])
        ).hexdigest()
        metadata_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            validate_normalization_artifacts(
                output / "imu_normalization.npz",
                metadata_path,
                expected_stage2_contract_sha256="a" * 64,
                expected_training_index_sha256="b" * 64,
                expected_train_sample_id_sha256="c" * 64,
                expected_fold=0,
                expected_train_users=["u1", "u2"],
                expected_source_stage2_manifest_sha256="d" * 64,
            )


def test_normalization_cli_requires_source_contract_inputs() -> None:
    from scripts.compute_imu_normalization import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(
            [
                "--training-index",
                "training_index.csv",
                "--training-index-metadata",
                "training_index.json",
                "--stage2-root",
                "stage2",
                "--stage2-schema",
                "stage2/schema.json",
                "--output-dir",
                "normalization",
            ]
        )


def test_normalization_rejects_rehashed_split_tampering_before_npz_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.build_imu_training_index import hash_training_index
    from scripts.compute_imu_normalization import generate_normalization_artifacts

    fixture = _pipeline_contract_fixture(tmp_path)
    frame = pd.read_csv(fixture["index"], encoding="utf-8-sig", keep_default_na=False)
    frame.loc[frame["sample_id"] == "validation", "split"] = "train"
    frame.to_csv(fixture["index"], index=False, encoding="utf-8-sig")
    metadata = json.loads(fixture["metadata"].read_text(encoding="utf-8"))
    metadata["training_index_sha256"] = hash_training_index(frame)
    train_ids = sorted(frame.loc[frame["split"] == "train", "sample_id"].astype(str))
    metadata["train_sample_id_sha256"] = hashlib.sha256(
        "".join(f"{sample_id}\n" for sample_id in train_ids).encode("utf-8")
    ).hexdigest()
    metadata["validation_sample_id_sha256"] = hashlib.sha256(b"").hexdigest()
    fixture["metadata"].write_text(json.dumps(metadata), encoding="utf-8")

    monkeypatch.setattr(
        "scripts.compute_imu_normalization.compute_normalization",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("NPZ computation ran before source-contract validation")
        ),
    )
    with pytest.raises(ValueError, match="split"):
        generate_normalization_artifacts(
            fixture["index"],
            fixture["metadata"],
            fixture["stage2_root"],
            fixture["schema"],
            tmp_path / "normalization",
            class_order_path=fixture["class_order"],
            split_path=fixture["split"],
            stage2_manifest_path=fixture["manifest"],
            repository_root=tmp_path,
        )


def test_normalization_binds_actual_stage2_root_manifest(tmp_path: Path) -> None:
    from scripts.compute_imu_normalization import generate_normalization_artifacts

    fixture = _pipeline_contract_fixture(tmp_path)
    fixture["manifest"].write_bytes(fixture["manifest"].read_bytes() + b"\n")

    with pytest.raises(ValueError, match="source_stage2_manifest_sha256"):
        generate_normalization_artifacts(
            fixture["index"],
            fixture["metadata"],
            fixture["stage2_root"],
            fixture["schema"],
            tmp_path / "normalization",
            class_order_path=fixture["class_order"],
            split_path=fixture["split"],
            stage2_manifest_path=fixture["manifest"],
            repository_root=tmp_path,
        )


def test_generate_normalization_validates_sources_and_writes_artifacts(
    tmp_path: Path,
) -> None:
    from scripts.compute_imu_normalization import generate_normalization_artifacts

    fixture = _pipeline_contract_fixture(tmp_path)
    output = tmp_path / "normalization"

    metadata = generate_normalization_artifacts(
        fixture["index"],
        fixture["metadata"],
        fixture["stage2_root"],
        fixture["schema"],
        output,
        class_order_path=fixture["class_order"],
        split_path=fixture["split"],
        stage2_manifest_path=fixture["manifest"],
        repository_root=tmp_path,
    )

    assert (output / "imu_normalization.npz").is_file()
    assert (output / "imu_normalization.json").is_file()
    assert metadata["provenance"]["source_stage2_manifest_sha256"] == hashlib.sha256(
        fixture["manifest"].read_bytes()
    ).hexdigest()
