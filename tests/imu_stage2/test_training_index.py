from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "sample_id": "s_train_a",
                "class_id": "10",
                "class_name": "Ten",
                "user_id": "u_train",
                "action_id": "a1",
                "stage2_npz_relpath": "10/u_train/a1/imu_stage2.npz",
                "status": "success",
                "imu_usable": "True",
                "sensor_mask": "[True, True, True, True, True]",
                "usable_sensor_mask": "[True, True, True, True, True]",
            },
            {
                "sample_id": "s_val_b",
                "class_id": "30",
                "class_name": "Thirty",
                "user_id": "u_val",
                "action_id": "a2",
                "stage2_npz_relpath": "30/u_val/a2/imu_stage2.npz",
                "status": "success_with_warnings",
                "imu_usable": "True",
                "sensor_mask": "[True, True, True, True, True]",
                "usable_sensor_mask": "[True, True, True, True, True]",
            },
            {
                "sample_id": "s_ineligible",
                "class_id": "10",
                "class_name": "Ten",
                "user_id": "u_train",
                "action_id": "a3",
                "stage2_npz_relpath": "10/u_train/a3/imu_stage2.npz",
                "status": "incomplete_sensors",
                "imu_usable": "True",
                "sensor_mask": "[True, True, True, True, False]",
                "usable_sensor_mask": "[True, True, True, True, False]",
            },
            {
                "sample_id": "s_unselected",
                "class_id": "30",
                "class_name": "Thirty",
                "user_id": "u_other",
                "action_id": "a4",
                "stage2_npz_relpath": "30/u_other/a4/imu_stage2.npz",
                "status": "success",
                "imu_usable": "True",
                "sensor_mask": "[True, True, True, True, True]",
                "usable_sensor_mask": "[True, True, True, True, True]",
            },
        ]
    )


def _split() -> dict[str, object]:
    return {"fold": 0, "train_users": ["u_train"], "val_users": ["u_val"]}


def test_class_order_uses_sorted_noncontiguous_ids_and_derived_class_count() -> None:
    from scripts.build_imu_training_index import build_class_order

    contract = build_class_order(_manifest().iloc[[1, 0, 3, 2]])

    assert contract.num_classes == 2
    assert [record["class_id"] for record in contract.classes] == [10, 30]
    assert [record["label_index"] for record in contract.classes] == [0, 1]
    assert len(contract.class_order_sha256) == 64


@pytest.mark.parametrize(
    ("column", "value"),
    [("class_name", "Other"), ("class_id", "10")],
)
def test_class_order_rejects_non_bijective_id_name_mapping(
    column: str,
    value: str,
) -> None:
    from scripts.build_imu_training_index import build_class_order

    manifest = _manifest()
    manifest.loc[1, column] = value

    with pytest.raises(ValueError, match="one-to-one"):
        build_class_order(manifest)


def test_training_index_enforces_strict_eligibility_and_split_semantics() -> None:
    from scripts.build_imu_training_index import build_class_order, build_training_index

    manifest = _manifest()
    index = build_training_index(manifest, build_class_order(manifest), _split())
    rows = index.set_index("sample_id")

    assert rows.loc["s_train_a", "label_index"] == 0
    assert rows.loc["s_val_b", "label_index"] == 1
    assert rows.loc["s_train_a", "split"] == "train"
    assert rows.loc["s_val_b", "split"] == "validation"
    assert bool(rows.loc["s_train_a", "selected_for_run"])
    assert not bool(rows.loc["s_ineligible", "eligible_for_strict_training"])
    assert not bool(rows.loc["s_ineligible", "selected_for_run"])
    assert rows.loc["s_ineligible", "split"] == ""
    assert rows.loc["s_unselected", "split"] == ""
    assert (index["selected_for_run"] == index["split"].isin(["train", "validation"])).all()


def test_training_index_rejects_overlapping_users_and_duplicate_samples() -> None:
    from scripts.build_imu_training_index import build_class_order, build_training_index

    manifest = _manifest()
    contract = build_class_order(manifest)
    with pytest.raises(ValueError, match="disjoint"):
        build_training_index(
            manifest,
            contract,
            {"fold": 0, "train_users": ["u_train"], "val_users": ["u_train"]},
        )
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        build_training_index(pd.concat([manifest, manifest.iloc[[0]]]), contract, _split())


def test_behavior_hash_changes_when_train_and_validation_are_swapped() -> None:
    from scripts.build_imu_training_index import (
        build_class_order,
        build_training_index,
        hash_training_index,
    )

    manifest = _manifest()
    contract = build_class_order(manifest)
    original = build_training_index(manifest, contract, _split())
    swapped = build_training_index(
        manifest,
        contract,
        {"fold": 0, "train_users": ["u_val"], "val_users": ["u_train"]},
    )

    assert set(original["sample_id"]) == set(swapped["sample_id"])
    assert hash_training_index(original) != hash_training_index(swapped)


def test_metadata_binds_manifest_split_class_order_and_behavior(tmp_path: Path) -> None:
    from scripts.build_imu_training_index import (
        build_class_order,
        build_training_index,
        build_training_index_metadata,
        validate_training_index_metadata,
    )

    manifest = _manifest()
    contract = build_class_order(manifest)
    index = build_training_index(manifest, contract, _split())
    split_path = tmp_path / "fold.json"
    split_path.write_text(json.dumps(_split()), encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    manifest_path.write_text("fixture\n", encoding="utf-8")
    metadata = build_training_index_metadata(
        index,
        contract,
        split_definition=_split(),
        split_path=split_path,
        source_stage2_manifest_path=manifest_path,
        stage2_contract_sha256="a" * 64,
        repository_root=tmp_path,
    )

    assert metadata["split_definition_sha256"] == hashlib.sha256(split_path.read_bytes()).hexdigest()
    assert metadata["num_classes"] == 2
    validate_training_index_metadata(
        metadata,
        index,
        contract,
        expected_source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        expected_stage2_contract_sha256="a" * 64,
        expected_split_definition_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
    )

    for key, replacement in (
        ("source_stage2_manifest_sha256", "b" * 64),
        ("stage2_contract_sha256", "b" * 64),
        ("split_definition_sha256", "b" * 64),
        ("class_order_sha256", "b" * 64),
        ("training_index_sha256", "b" * 64),
    ):
        tampered = dict(metadata)
        tampered[key] = replacement
        with pytest.raises(ValueError, match=key):
            validate_training_index_metadata(
                tampered,
                index,
                contract,
                expected_source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                expected_stage2_contract_sha256="a" * 64,
                expected_split_definition_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
            )


def test_artifact_generation_rejects_a_missing_unselected_npz(tmp_path: Path) -> None:
    from scripts.build_imu_training_index import generate_training_index_artifacts
    from src.data.imu_stage2_io import build_stage2_schema

    stage2_root = tmp_path / "stage2"
    stage2_root.mkdir()
    manifest = _manifest()
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
    (stage2_root / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    split_path = tmp_path / "fold.json"
    split_path.write_text(json.dumps(_split()), encoding="utf-8")
    for relpath in manifest.loc[manifest["sample_id"] != "s_unselected", "stage2_npz_relpath"]:
        path = stage2_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")

    with pytest.raises(ValueError, match="missing"):
        generate_training_index_artifacts(
            manifest_path,
            tmp_path / "output",
            split_path,
            repository_root=tmp_path,
        )
