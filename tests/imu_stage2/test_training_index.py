from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
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
                "sensor_mask": "LL;RL;LA;RA;C",
                "usable_sensor_mask": "LL;RL;LA;RA;C",
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
                "sensor_mask": "LL;RL;LA;RA;C",
                "usable_sensor_mask": "LL;RL;LA;RA;C",
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
                "sensor_mask": "LL;RL;LA;RA",
                "usable_sensor_mask": "LL;RL;LA;RA",
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
                "sensor_mask": "LL;RL;LA;RA;C",
                "usable_sensor_mask": "LL;RL;LA;RA;C",
            },
        ]
    )


def _split() -> dict[str, object]:
    return {"fold": 0, "train_users": ["u_train"], "val_users": ["u_val"]}


def _generation_fixture(
    tmp_path: Path,
    *,
    manifest_name: str = "manifest.csv",
) -> tuple[Path, Path, Path]:
    from src.data.imu_stage2_io import build_stage2_schema

    stage2_root = tmp_path / "datasets" / "new_IMU_stage2"
    stage2_root.mkdir(parents=True)
    manifest = _manifest()
    manifest_path = stage2_root / manifest_name
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
    for relpath in manifest["stage2_npz_relpath"]:
        artifact = stage2_root / relpath
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"fixture")
    repository_root = tmp_path / "repo"
    split_path = repository_root / "metadata" / "splits" / "fold_0.json"
    split_path.parent.mkdir(parents=True)
    split_path.write_text(json.dumps(_split()), encoding="utf-8")
    return manifest_path, split_path, repository_root


def _assert_no_publication_residue(output_dir: Path) -> None:
    assert not output_dir.exists() or not any(output_dir.iterdir())
    assert not list(output_dir.parent.glob(f".{output_dir.name}.staging-*"))
    assert not list(output_dir.parent.glob(f".{output_dir.name}.backup-*"))


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
    assert rows.loc["s_ineligible", "sensor_mask"] == "[true,true,true,true,false]"
    assert rows.loc["s_ineligible", "usable_sensor_mask"] == "[true,true,true,true,false]"
    assert "missing_historical_sensor" in rows.loc["s_ineligible", "exclusion_reason"]
    assert "unusable_sensor" in rows.loc["s_ineligible", "exclusion_reason"]
    assert rows.loc["s_train_a", "sensor_mask"] == "[true,true,true,true,true]"
    assert rows.loc["s_train_a", "usable_sensor_mask"] == "[true,true,true,true,true]"
    assert rows.loc["s_unselected", "split"] == ""
    assert (index["selected_for_run"] == index["split"].isin(["train", "validation"])).all()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", (False, False, False, False, False)),
        ("LL", (True, False, False, False, False)),
        ("LL;RL", (True, True, False, False, False)),
        ("LA;RA;C", (False, False, True, True, True)),
        ("LL;RL;LA;RA;C", (True, True, True, True, True)),
    ],
)
def test_manifest_sensor_mask_parser_accepts_only_canonical_sensor_subsequences(
    value: str,
    expected: tuple[bool, ...],
) -> None:
    from scripts.build_imu_training_index import _parse_manifest_sensor_mask

    assert _parse_manifest_sensor_mask(value, "sensor_mask") == expected


@pytest.mark.parametrize(
    "value",
    [
        "UNKNOWN",
        "LL;UNKNOWN",
        "LL;LL",
        "RL;LL",
        "LL;;RL",
        ";LL",
        "LL;",
        "LL,RL",
        "[true,true,true,true,true]",
        "[True, True, True, True, True]",
    ],
)
@pytest.mark.parametrize("column", ["sensor_mask", "usable_sensor_mask"])
def test_manifest_sensor_mask_parser_rejects_noncanonical_encodings(
    value: str,
    column: str,
) -> None:
    from scripts.build_imu_training_index import _parse_manifest_sensor_mask

    with pytest.raises(ValueError, match=column):
        _parse_manifest_sensor_mask(value, column)


@pytest.mark.parametrize(
    "value",
    [
        "[true,false,true,false,true]",
        "[True, False, True, False, True]",
        [True, False, True, False, True],
        (True, False, True, False, True),
    ],
)
def test_training_mask_parser_accepts_only_actual_boolean_elements(
    value: object,
) -> None:
    from scripts.build_imu_training_index import _parse_mask

    assert _parse_mask(value, "sensor_mask") == (
        True,
        False,
        True,
        False,
        True,
    )


@pytest.mark.parametrize(
    "value",
    [
        "[1,1,1,1,1]",
        "[0,0,0,0,0]",
        "[1,0,1,0,1]",
        '["true","true","true","true","true"]',
        '["false","false","false","false","false"]',
        '["1","1","1","1","1"]',
        '["0","0","0","0","0"]',
        "[true,true,true,true,1]",
        [np.bool_(True)] * 5,
        [np.int64(1)] * 5,
    ],
)
@pytest.mark.parametrize("column", ["sensor_mask", "usable_sensor_mask"])
def test_training_mask_parser_rejects_non_boolean_elements(
    value: object,
    column: str,
) -> None:
    from scripts.build_imu_training_index import _parse_mask

    with pytest.raises(ValueError, match=column):
        _parse_mask(value, column)


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
        hash_training_index,
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
        expected_fold=0,
        expected_split_definition_path="fold.json",
        expected_source_manifest_path="manifest.csv",
        expected_split_definition=_split(),
    )
    unexpected = dict(metadata)
    unexpected["unexpected"] = "value"
    with pytest.raises(ValueError, match="keys"):
        validate_training_index_metadata(
            unexpected,
            index,
            contract,
            expected_source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            expected_stage2_contract_sha256="a" * 64,
            expected_split_definition_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
            expected_fold=0,
            expected_split_definition_path="fold.json",
            expected_source_manifest_path="manifest.csv",
            expected_split_definition=_split(),
        )

    for key, replacement in (
        ("training_index_version", "imu-training-index-v2"),
        ("fold", 1),
        ("split_definition_path", "other.json"),
        ("source_stage2_manifest_path", "other.csv"),
    ):
        tampered_contract = dict(metadata)
        tampered_contract[key] = replacement
        with pytest.raises(ValueError, match=key):
            validate_training_index_metadata(
                tampered_contract,
                index,
                contract,
                expected_source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                expected_stage2_contract_sha256="a" * 64,
                expected_split_definition_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
                expected_fold=0,
                expected_split_definition_path="fold.json",
                expected_source_manifest_path="manifest.csv",
                expected_split_definition=_split(),
            )

    for key, replacement in (
        ("source_stage2_manifest_sha256", "b" * 64),
        ("stage2_contract_sha256", "b" * 64),
        ("split_definition_sha256", "b" * 64),
        ("class_order_sha256", "b" * 64),
        ("training_index_sha256", "b" * 64),
        ("train_sample_id_sha256", "b" * 64),
        ("validation_sample_id_sha256", "b" * 64),
        ("selected_sample_id_sha256", "b" * 64),
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
                expected_fold=0,
                expected_split_definition_path="fold.json",
                expected_source_manifest_path="manifest.csv",
                expected_split_definition=_split(),
            )

    tampered_index = index.copy()
    tampered_index.loc[tampered_index["sample_id"] == "s_train_a", "class_name"] = "Wrong"
    tampered_metadata = dict(metadata)
    tampered_metadata["training_index_sha256"] = hash_training_index(tampered_index)
    with pytest.raises(ValueError, match="class identity"):
        validate_training_index_metadata(
            tampered_metadata,
            tampered_index,
            contract,
            expected_source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            expected_stage2_contract_sha256="a" * 64,
            expected_split_definition_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
            expected_fold=0,
            expected_split_definition_path="fold.json",
            expected_source_manifest_path="manifest.csv",
            expected_split_definition=_split(),
        )

    inconsistent = index.copy()
    inconsistent.loc[inconsistent["sample_id"] == "s_train_a", "selected_for_run"] = False
    inconsistent_metadata = dict(metadata)
    inconsistent_metadata["training_index_sha256"] = hash_training_index(inconsistent)
    with pytest.raises(ValueError, match="selected_for_run"):
        validate_training_index_metadata(
            inconsistent_metadata,
            inconsistent,
            contract,
            expected_source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            expected_stage2_contract_sha256="a" * 64,
            expected_split_definition_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),
            expected_fold=0,
            expected_split_definition_path="fold.json",
            expected_source_manifest_path="manifest.csv",
            expected_split_definition=_split(),
        )

    manifest_encoded = index.copy()
    manifest_encoded.loc[
        manifest_encoded["sample_id"] == "s_train_a", "sensor_mask"
    ] = "LL;RL;LA;RA;C"
    manifest_encoded_metadata = dict(metadata)
    manifest_encoded_metadata["training_index_sha256"] = hash_training_index(
        manifest_encoded
    )
    with pytest.raises(ValueError, match="sensor_mask"):
        validate_training_index_metadata(
            manifest_encoded_metadata,
            manifest_encoded,
            contract,
            expected_source_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
            expected_stage2_contract_sha256="a" * 64,
            expected_split_definition_sha256=hashlib.sha256(
                split_path.read_bytes()
            ).hexdigest(),
            expected_fold=0,
            expected_split_definition_path="fold.json",
            expected_source_manifest_path="manifest.csv",
            expected_split_definition=_split(),
        )

    for column, replacement in (
        ("sensor_mask", "[1,1,1,1,1]"),
        ("usable_sensor_mask", "[0,0,0,0,0]"),
        ("sensor_mask", '["true","true","true","true","true"]'),
    ):
        non_boolean = index.copy()
        non_boolean.loc[
            non_boolean["sample_id"] == "s_train_a", column
        ] = replacement
        non_boolean_metadata = dict(metadata)
        non_boolean_metadata["training_index_sha256"] = hash_training_index(
            non_boolean
        )
        with pytest.raises(ValueError, match=column):
            validate_training_index_metadata(
                non_boolean_metadata,
                non_boolean,
                contract,
                expected_source_manifest_sha256=hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                expected_stage2_contract_sha256="a" * 64,
                expected_split_definition_sha256=hashlib.sha256(
                    split_path.read_bytes()
                ).hexdigest(),
                expected_fold=0,
                expected_split_definition_path="fold.json",
                expected_source_manifest_path="manifest.csv",
                expected_split_definition=_split(),
            )


def test_generation_from_formal_manifest_publishes_strict_training_masks(
    tmp_path: Path,
) -> None:
    from scripts.build_imu_training_index import (
        _strict_json,
        generate_training_index_artifacts,
        load_class_order,
        validate_training_index_metadata,
    )
    from src.data.imu_stage2_contracts import sha256_file
    from src.data.imu_stage2_io import load_stage2_schema

    manifest_path, split_path, repository_root = _generation_fixture(tmp_path)
    output_dir = tmp_path / "index"

    generate_training_index_artifacts(
        manifest_path,
        output_dir,
        split_path,
        repository_root=repository_root,
    )

    assert sorted(path.name for path in output_dir.iterdir()) == [
        "class_order.json",
        "training_index.csv",
        "training_index.json",
    ]
    assert not list(output_dir.parent.glob(f".{output_dir.name}.staging-*"))
    assert not list(output_dir.parent.glob(f".{output_dir.name}.backup-*"))
    index = pd.read_csv(
        output_dir / "training_index.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    assert set(index["sensor_mask"]) == {
        "[true,true,true,true,true]",
        "[true,true,true,true,false]",
    }
    assert set(index["usable_sensor_mask"]) == {
        "[true,true,true,true,true]",
        "[true,true,true,true,false]",
    }
    assert not index["sensor_mask"].str.contains("LL", regex=False).any()
    assert not index["usable_sensor_mask"].str.contains("LL", regex=False).any()
    metadata = _strict_json(output_dir / "training_index.json")
    class_order = load_class_order(output_dir / "class_order.json")
    split_definition = _strict_json(split_path)
    schema = load_stage2_schema(manifest_path.parent / "schema.json")
    validate_training_index_metadata(
        metadata,
        index,
        class_order,
        expected_source_manifest_sha256=sha256_file(manifest_path),
        expected_stage2_contract_sha256=str(schema["contract_sha256"]),
        expected_split_definition_sha256=sha256_file(split_path),
        expected_fold=0,
        expected_split_definition_path="metadata/splits/fold_0.json",
        expected_source_manifest_path="manifest.csv",
        expected_split_definition=split_definition,
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


def test_cli_accepts_stage2_manifest_outside_repository(tmp_path: Path) -> None:
    from src.data.imu_stage2_io import build_stage2_schema

    repository_root = Path(__file__).resolve().parents[2]
    split_path = repository_root / "metadata" / "splits" / "fold_0.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    stage2_root = tmp_path / "external-datasets" / "new_IMU_stage2"
    stage2_root.mkdir(parents=True)
    manifest = pd.DataFrame(
        [
            {
                "sample_id": "external_sample",
                "class_id": "10",
                "class_name": "Ten",
                "user_id": str(split["train_users"][0]),
                "action_id": "a1",
                "stage2_npz_relpath": "10/user/a1/imu_stage2.npz",
                "status": "success",
                "imu_usable": "True",
                "sensor_mask": "LL;RL;LA;RA;C",
                "usable_sensor_mask": "LL;RL;LA;RA;C",
            }
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
    (stage2_root / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
    artifact = stage2_root / "10" / "user" / "a1" / "imu_stage2.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"fixture")
    output_dir = tmp_path / "index"

    assert not manifest_path.resolve().is_relative_to(repository_root.resolve())
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "build_imu_training_index.py"),
            "--stage2-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--split-file",
            str(split_path),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert sorted(path.name for path in output_dir.iterdir()) == [
        "class_order.json",
        "training_index.csv",
        "training_index.json",
    ]
    metadata = json.loads(
        (output_dir / "training_index.json").read_text(encoding="utf-8")
    )
    assert metadata["source_stage2_manifest_path"] == "manifest.csv"
    assert metadata["split_definition_path"] == "metadata/splits/fold_0.json"


@pytest.mark.parametrize("manifest_name", ["alternate.csv", "MANIFEST.csv", "Manifest.csv"])
def test_generation_rejects_noncanonical_stage2_manifest_before_publication(
    tmp_path: Path,
    manifest_name: str,
) -> None:
    from scripts.build_imu_training_index import generate_training_index_artifacts

    manifest_path, split_path, repository_root = _generation_fixture(
        tmp_path,
        manifest_name=manifest_name,
    )
    output_dir = tmp_path / "index"

    with pytest.raises(ValueError, match="named exactly manifest.csv"):
        generate_training_index_artifacts(
            manifest_path,
            output_dir,
            split_path,
            repository_root=repository_root,
        )

    _assert_no_publication_residue(output_dir)


def test_cli_rejects_noncanonical_stage2_manifest_before_publication(
    tmp_path: Path,
) -> None:
    manifest_path, _, _ = _generation_fixture(
        tmp_path,
        manifest_name="alternate.csv",
    )
    repository_root = Path(__file__).resolve().parents[2]
    split_path = repository_root / "metadata" / "splits" / "fold_0.json"
    output_dir = tmp_path / "index"

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[2] / "scripts" / "build_imu_training_index.py"),
            "--stage2-manifest",
            str(manifest_path),
            "--output-dir",
            str(output_dir),
            "--split-file",
            str(split_path),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "named exactly manifest.csv" in completed.stderr
    _assert_no_publication_residue(output_dir)


def test_transaction_removes_partial_artifacts_when_csv_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_imu_training_index

    manifest_path, split_path, repository_root = _generation_fixture(tmp_path)
    output_dir = tmp_path / "index"
    monkeypatch.setattr(
        build_imu_training_index,
        "_write_csv_atomic",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("injected csv failure")),
    )

    with pytest.raises(OSError, match="injected csv failure"):
        build_imu_training_index.generate_training_index_artifacts(
            manifest_path,
            output_dir,
            split_path,
            repository_root=repository_root,
        )

    _assert_no_publication_residue(output_dir)


def test_transaction_removes_partial_artifacts_when_metadata_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_imu_training_index

    manifest_path, split_path, repository_root = _generation_fixture(tmp_path)
    output_dir = tmp_path / "index"
    original = build_imu_training_index.write_json_atomic
    calls = 0

    def fail_second_json(*args: object, **kwargs: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected metadata failure")
        original(*args, **kwargs)

    monkeypatch.setattr(build_imu_training_index, "write_json_atomic", fail_second_json)
    with pytest.raises(OSError, match="injected metadata failure"):
        build_imu_training_index.generate_training_index_artifacts(
            manifest_path,
            output_dir,
            split_path,
            repository_root=repository_root,
        )

    _assert_no_publication_residue(output_dir)


def test_transaction_cleans_staging_when_cross_validation_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_imu_training_index

    manifest_path, split_path, repository_root = _generation_fixture(tmp_path)
    output_dir = tmp_path / "index"
    monkeypatch.setattr(
        build_imu_training_index,
        "_validate_staged_training_artifacts",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("injected staged validation failure")),
    )

    with pytest.raises(ValueError, match="injected staged validation failure"):
        build_imu_training_index.generate_training_index_artifacts(
            manifest_path,
            output_dir,
            split_path,
            repository_root=repository_root,
        )

    _assert_no_publication_residue(output_dir)


def test_transaction_restores_empty_output_when_install_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts import build_imu_training_index

    manifest_path, split_path, repository_root = _generation_fixture(tmp_path)
    output_dir = tmp_path / "index"
    output_dir.mkdir()
    original_replace = os.replace

    def fail_staging_install(source: object, destination: object) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name.startswith(f".{output_dir.name}.staging-") and destination_path == output_dir:
            raise OSError("injected install failure")
        original_replace(source, destination)

    monkeypatch.setattr(build_imu_training_index.os, "replace", fail_staging_install)
    with pytest.raises(OSError, match="injected install failure"):
        build_imu_training_index.generate_training_index_artifacts(
            manifest_path,
            output_dir,
            split_path,
            repository_root=repository_root,
        )

    _assert_no_publication_residue(output_dir)
