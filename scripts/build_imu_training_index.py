from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence
from uuid import uuid4

import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.data.imu_stage2_contracts import SENSOR_ORDER, canonical_json_bytes, sha256_file
from src.data.imu_stage2_io import load_stage2_schema, write_json_atomic


CLASS_ORDER_VERSION = "imu-class-order-v1"
TRAINING_INDEX_VERSION = "imu-training-index-v1"
TRAINING_METADATA_KEYS = frozenset(
    {
        "training_index_version",
        "class_order_version",
        "class_order_sha256",
        "num_classes",
        "fold",
        "split_definition_path",
        "split_definition_sha256",
        "source_stage2_manifest_path",
        "source_stage2_manifest_sha256",
        "stage2_contract_sha256",
        "training_index_sha256",
        "train_sample_id_sha256",
        "validation_sample_id_sha256",
        "selected_sample_id_sha256",
    }
)
TRAINING_INDEX_COLUMNS = (
    "sample_id",
    "class_id",
    "class_name",
    "user_id",
    "action_id",
    "label_index",
    "stage2_npz_relpath",
    "status",
    "imu_usable",
    "sensor_mask",
    "usable_sensor_mask",
    "eligible_for_strict_training",
    "selected_for_run",
    "split",
    "exclusion_reason",
)
HASH_COLUMNS = (
    "sample_id",
    "label_index",
    "split",
    "selected_for_run",
    "eligible_for_strict_training",
    "stage2_npz_relpath",
)
SUCCESS_STATUSES = frozenset({"success", "success_with_warnings"})
TRAINING_ARTIFACT_NAMES = frozenset(
    {"class_order.json", "training_index.csv", "training_index.json"}
)


@dataclass(frozen=True)
class ClassOrderContract:
    classes: tuple[dict[str, object], ...]
    class_order_sha256: str
    num_classes: int
    class_order_version: str = CLASS_ORDER_VERSION

    def contract_payload(self) -> dict[str, object]:
        return {
            "class_order_version": self.class_order_version,
            "num_classes": self.num_classes,
            "classes": [dict(record) for record in self.classes],
        }

    def to_payload(self) -> dict[str, object]:
        return {**self.contract_payload(), "class_order_sha256": self.class_order_sha256}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"Duplicate JSON key: {key}")
            payload[key] = value
        return payload

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    if not isinstance(payload, dict):
        raise ValueError("JSON contract must be an object")
    return payload


def _class_identity_rows(stage2_manifest: pd.DataFrame) -> list[tuple[int, str]]:
    required = {"class_id", "class_name"}
    if not required.issubset(stage2_manifest.columns):
        raise ValueError("Stage 2 manifest is missing class identity columns")
    identities: list[tuple[int, str]] = []
    for row in stage2_manifest.loc[:, ["class_id", "class_name"]].itertuples(index=False):
        try:
            class_id = int(str(row.class_id))
        except (TypeError, ValueError) as error:
            raise ValueError("class_id must be an integer") from error
        class_name = str(row.class_name).strip()
        if not class_name:
            raise ValueError("class_name must be non-empty")
        identities.append((class_id, class_name))
    id_to_name: dict[int, str] = {}
    name_to_id: dict[str, int] = {}
    for class_id, class_name in identities:
        if class_id in id_to_name and id_to_name[class_id] != class_name:
            raise ValueError("Class identity mapping must be one-to-one")
        if class_name in name_to_id and name_to_id[class_name] != class_id:
            raise ValueError("Class identity mapping must be one-to-one")
        id_to_name[class_id] = class_name
        name_to_id[class_name] = class_id
    return sorted(id_to_name.items())


def build_class_order(stage2_manifest: pd.DataFrame) -> ClassOrderContract:
    ordered = _class_identity_rows(stage2_manifest)
    if not ordered:
        raise ValueError("Stage 2 manifest has no class identities")
    classes = tuple(
        {"class_id": class_id, "class_name": class_name, "label_index": label_index}
        for label_index, (class_id, class_name) in enumerate(ordered)
    )
    payload = {
        "class_order_version": CLASS_ORDER_VERSION,
        "num_classes": len(classes),
        "classes": [dict(record) for record in classes],
    }
    return ClassOrderContract(
        classes=classes,
        class_order_sha256=_sha256_bytes(canonical_json_bytes(payload)),
        num_classes=len(classes),
    )


def load_class_order(path: Path) -> ClassOrderContract:
    payload = _strict_json(Path(path))
    if set(payload) != {
        "class_order_version",
        "num_classes",
        "classes",
        "class_order_sha256",
    }:
        raise ValueError("class_order.json keys do not match contract")
    if payload["class_order_version"] != CLASS_ORDER_VERSION:
        raise ValueError("class_order_version is incompatible")
    classes_value = payload["classes"]
    if not isinstance(classes_value, list) or not classes_value:
        raise ValueError("class_order classes must be a non-empty list")
    classes: list[dict[str, object]] = []
    for label_index, record in enumerate(classes_value):
        if not isinstance(record, dict) or set(record) != {
            "class_id",
            "class_name",
            "label_index",
        }:
            raise ValueError("class_order record keys are invalid")
        normalized = {
            "class_id": int(record["class_id"]),
            "class_name": str(record["class_name"]),
            "label_index": int(record["label_index"]),
        }
        if normalized["label_index"] != label_index:
            raise ValueError("class_order label_index must be consecutive")
        classes.append(normalized)
    contract = ClassOrderContract(
        classes=tuple(classes),
        class_order_sha256=str(payload["class_order_sha256"]).lower(),
        num_classes=int(payload["num_classes"]),
    )
    if contract.num_classes != len(contract.classes):
        raise ValueError("class_order num_classes mismatch")
    expected = _sha256_bytes(canonical_json_bytes(contract.contract_payload()))
    if contract.class_order_sha256 != expected:
        raise ValueError("class_order_sha256 mismatch")
    return contract


def _parse_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"{name} must be boolean")


def _parse_mask(value: object, name: str) -> tuple[bool, ...]:
    if isinstance(value, (list, tuple)):
        parsed = value
    else:
        try:
            parsed = json.loads(str(value))
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(str(value))
            except (SyntaxError, ValueError) as error:
                raise ValueError(f"{name} must be a five-element boolean list") from error
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 5:
        raise ValueError(f"{name} must be a five-element boolean list")
    if not all(type(item) is bool for item in parsed):
        raise ValueError(f"{name} must contain only boolean elements")
    return tuple(parsed)


def _parse_manifest_sensor_mask(
    value: object,
    name: str,
) -> tuple[bool, ...]:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a canonical sensor-name list")
    if value == "":
        return tuple(False for _ in SENSOR_ORDER)
    tokens = value.split(";")
    if any(token == "" for token in tokens):
        raise ValueError(f"{name} must be a canonical sensor-name list")
    positions = {sensor: index for index, sensor in enumerate(SENSOR_ORDER)}
    if any(token not in positions for token in tokens):
        raise ValueError(f"{name} must be a canonical sensor-name list")
    indices = [positions[token] for token in tokens]
    if len(set(tokens)) != len(tokens) or indices != sorted(indices):
        raise ValueError(f"{name} must be a canonical sensor-name list")
    present = set(tokens)
    return tuple(sensor in present for sensor in SENSOR_ORDER)


def _split_users(split_definition: Mapping[str, object]) -> tuple[set[str], set[str]]:
    train_value = split_definition.get("train_users")
    val_value = split_definition.get("val_users")
    if not isinstance(train_value, list) or not isinstance(val_value, list):
        raise ValueError("Split definition must contain train_users and val_users lists")
    train_users = {str(user) for user in train_value}
    validation_users = {str(user) for user in val_value}
    if train_users & validation_users:
        raise ValueError("Train and validation users must be disjoint")
    return train_users, validation_users


def build_training_index(
    stage2_manifest: pd.DataFrame,
    class_order: ClassOrderContract,
    split_definition: Mapping[str, object],
) -> pd.DataFrame:
    required = {
        "sample_id",
        "class_id",
        "class_name",
        "user_id",
        "action_id",
        "stage2_npz_relpath",
        "status",
        "imu_usable",
        "sensor_mask",
        "usable_sensor_mask",
    }
    missing = required - set(stage2_manifest.columns)
    if missing:
        raise ValueError(f"Stage 2 manifest is missing columns: {sorted(missing)}")
    if stage2_manifest["sample_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate sample_id in Stage 2 manifest")
    train_users, validation_users = _split_users(split_definition)
    labels = {
        (int(record["class_id"]), str(record["class_name"])): int(record["label_index"])
        for record in class_order.classes
    }
    rows: list[dict[str, object]] = []
    for source in stage2_manifest.to_dict(orient="records"):
        sample_id = str(source["sample_id"]).strip()
        if not sample_id:
            raise ValueError("sample_id must be non-empty")
        class_id = int(str(source["class_id"]))
        class_name = str(source["class_name"]).strip()
        identity = (class_id, class_name)
        if identity not in labels:
            raise ValueError("Stage 2 manifest class identity is absent from class order")
        label_index = labels[identity]
        if not 0 <= label_index < class_order.num_classes:
            raise ValueError("label_index is outside class order")
        user_id = str(source["user_id"])
        status = str(source["status"])
        imu_usable = _parse_bool(source["imu_usable"], "imu_usable")
        sensor_mask = _parse_manifest_sensor_mask(
            source["sensor_mask"], "sensor_mask"
        )
        usable_sensor_mask = _parse_manifest_sensor_mask(
            source["usable_sensor_mask"], "usable_sensor_mask"
        )
        stage2_npz_relpath = str(source["stage2_npz_relpath"]).strip()
        reasons: list[str] = []
        if not imu_usable:
            reasons.append("imu_unusable")
        if not all(sensor_mask):
            reasons.append("missing_historical_sensor")
        if not all(usable_sensor_mask):
            reasons.append("unusable_sensor")
        if status not in SUCCESS_STATUSES:
            reasons.append("ineligible_status")
        if not stage2_npz_relpath:
            reasons.append("missing_stage2_npz")
        eligible = not reasons
        split = ""
        if eligible and user_id in train_users:
            split = "train"
        elif eligible and user_id in validation_users:
            split = "validation"
        selected = split in {"train", "validation"}
        rows.append(
            {
                "sample_id": sample_id,
                "class_id": class_id,
                "class_name": class_name,
                "user_id": user_id,
                "action_id": str(source["action_id"]),
                "label_index": label_index,
                "stage2_npz_relpath": stage2_npz_relpath,
                "status": status,
                "imu_usable": imu_usable,
                "sensor_mask": json.dumps(list(sensor_mask), separators=(",", ":")),
                "usable_sensor_mask": json.dumps(
                    list(usable_sensor_mask), separators=(",", ":")
                ),
                "eligible_for_strict_training": eligible,
                "selected_for_run": selected,
                "split": split,
                "exclusion_reason": ";".join(reasons) if reasons else ("not_in_split" if not selected else ""),
            }
        )
    index = pd.DataFrame(rows, columns=TRAINING_INDEX_COLUMNS).sort_values(
        "sample_id", kind="stable", ignore_index=True
    )
    if not (
        index["selected_for_run"]
        == index["split"].isin(["train", "validation"])
    ).all():
        raise AssertionError("selected_for_run and split are inconsistent")
    train_selected = set(index.loc[index["split"] == "train", "user_id"])
    val_selected = set(index.loc[index["split"] == "validation", "user_id"])
    if train_selected & val_selected:
        raise ValueError("Selected train and validation users must be disjoint")
    return index


def hash_training_index(training_index: pd.DataFrame) -> str:
    missing = set(HASH_COLUMNS) - set(training_index.columns)
    if missing:
        raise ValueError(f"Training index hash columns are missing: {sorted(missing)}")
    rows = []
    for row in training_index.sort_values("sample_id", kind="stable").to_dict(orient="records"):
        rows.append({column: row[column] for column in HASH_COLUMNS})
    return _sha256_bytes(canonical_json_bytes({"rows": rows}))


def _hash_sample_ids(sample_ids: Sequence[str]) -> str:
    normalized = sorted(str(sample_id) for sample_id in sample_ids)
    payload = "".join(f"{sample_id}\n" for sample_id in normalized).encode("utf-8")
    return _sha256_bytes(payload)


def _repository_relative(path: Path, repository_root: Path) -> str:
    try:
        return Path(path).resolve(strict=True).relative_to(
            Path(repository_root).resolve(strict=True)
        ).as_posix()
    except ValueError as error:
        raise ValueError("Contract input path must be inside repository root") from error


def _canonical_stage2_manifest(path: Path) -> Path:
    resolved = Path(path).resolve(strict=True)
    if not resolved.is_file():
        raise ValueError("Stage 2 manifest must be a regular file")
    if resolved.name != "manifest.csv":
        raise ValueError("Stage 2 manifest must be named exactly manifest.csv")
    return resolved


def build_training_index_metadata(
    training_index: pd.DataFrame,
    class_order: ClassOrderContract,
    *,
    split_definition: Mapping[str, object],
    split_path: Path,
    source_stage2_manifest_path: Path,
    stage2_contract_sha256: str,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    source_stage2_manifest_path = _canonical_stage2_manifest(
        source_stage2_manifest_path
    )
    selected = training_index[training_index["selected_for_run"]]
    return {
        "training_index_version": TRAINING_INDEX_VERSION,
        "class_order_version": class_order.class_order_version,
        "class_order_sha256": class_order.class_order_sha256,
        "num_classes": class_order.num_classes,
        "fold": split_definition.get("fold"),
        "split_definition_path": _repository_relative(split_path, repository_root),
        "split_definition_sha256": sha256_file(Path(split_path)),
        "source_stage2_manifest_path": "manifest.csv",
        "source_stage2_manifest_sha256": sha256_file(Path(source_stage2_manifest_path)),
        "stage2_contract_sha256": str(stage2_contract_sha256).lower(),
        "training_index_sha256": hash_training_index(training_index),
        "train_sample_id_sha256": _hash_sample_ids(
            training_index.loc[training_index["split"] == "train", "sample_id"]
        ),
        "validation_sample_id_sha256": _hash_sample_ids(
            training_index.loc[training_index["split"] == "validation", "sample_id"]
        ),
        "selected_sample_id_sha256": _hash_sample_ids(selected["sample_id"]),
    }


def validate_training_index_metadata(
    metadata: Mapping[str, object],
    training_index: pd.DataFrame,
    class_order: ClassOrderContract,
    *,
    expected_source_manifest_sha256: str,
    expected_stage2_contract_sha256: str,
    expected_split_definition_sha256: str,
    expected_fold: object,
    expected_split_definition_path: str,
    expected_source_manifest_path: str,
    expected_split_definition: Mapping[str, object],
) -> None:
    if set(metadata) != TRAINING_METADATA_KEYS:
        raise ValueError("Training index metadata keys do not match contract")
    _validate_training_index_semantics(training_index, class_order)
    _validate_training_index_split_membership(
        training_index,
        expected_split_definition,
    )
    train_ids = training_index.loc[training_index["split"] == "train", "sample_id"]
    validation_ids = training_index.loc[
        training_index["split"] == "validation", "sample_id"
    ]
    selected_ids = training_index.loc[
        training_index["selected_for_run"].map(_parse_selected_bool), "sample_id"
    ]
    expected = {
        "training_index_version": TRAINING_INDEX_VERSION,
        "fold": expected_fold,
        "split_definition_path": _validate_metadata_relative_path(
            expected_split_definition_path,
            "split_definition_path",
        ),
        "source_stage2_manifest_path": _validate_metadata_relative_path(
            expected_source_manifest_path,
            "source_stage2_manifest_path",
        ),
        "source_stage2_manifest_sha256": expected_source_manifest_sha256.lower(),
        "stage2_contract_sha256": expected_stage2_contract_sha256.lower(),
        "split_definition_sha256": expected_split_definition_sha256.lower(),
        "class_order_sha256": class_order.class_order_sha256,
        "training_index_sha256": hash_training_index(training_index),
        "train_sample_id_sha256": _hash_sample_ids(train_ids),
        "validation_sample_id_sha256": _hash_sample_ids(validation_ids),
        "selected_sample_id_sha256": _hash_sample_ids(selected_ids),
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"{key} mismatch")
    if metadata.get("class_order_version") != class_order.class_order_version:
        raise ValueError("class_order_version mismatch")
    if metadata.get("num_classes") != class_order.num_classes:
        raise ValueError("num_classes mismatch")
    for label in training_index["label_index"]:
        if not 0 <= int(label) < class_order.num_classes:
            raise ValueError("label_index is outside class order")


def _validate_metadata_relative_path(value: str, name: str) -> str:
    path = Path(str(value))
    if path.is_absolute() or not path.parts or ".." in path.parts or "\\" in str(value):
        raise ValueError(f"{name} must be a safe relative POSIX path")
    return path.as_posix()


def _parse_selected_bool(value: object) -> bool:
    return _parse_bool(value, "selected_for_run")


def _validate_training_index_semantics(
    training_index: pd.DataFrame,
    class_order: ClassOrderContract,
) -> None:
    required = set(TRAINING_INDEX_COLUMNS)
    missing = required - set(training_index.columns)
    if missing:
        raise ValueError(f"Training index columns are missing: {sorted(missing)}")
    if training_index["sample_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate sample_id in training index")
    label_by_identity = {
        (int(record["class_id"]), str(record["class_name"])): int(record["label_index"])
        for record in class_order.classes
    }
    train_users: set[str] = set()
    validation_users: set[str] = set()
    for row in training_index.to_dict(orient="records"):
        identity = (int(row["class_id"]), str(row["class_name"]))
        if identity not in label_by_identity or int(row["label_index"]) != label_by_identity[identity]:
            raise ValueError("Training index class identity or label_index mismatch")
        imu_usable = _parse_bool(row["imu_usable"], "imu_usable")
        sensor_mask = _parse_mask(row["sensor_mask"], "sensor_mask")
        usable_sensor_mask = _parse_mask(
            row["usable_sensor_mask"], "usable_sensor_mask"
        )
        expected_eligible = bool(
            imu_usable
            and all(sensor_mask)
            and all(usable_sensor_mask)
            and str(row["status"]) in SUCCESS_STATUSES
            and str(row["stage2_npz_relpath"]).strip()
        )
        actual_eligible = _parse_bool(
            row["eligible_for_strict_training"],
            "eligible_for_strict_training",
        )
        if actual_eligible != expected_eligible:
            raise ValueError("eligible_for_strict_training contradicts row data")
        selected = _parse_selected_bool(row["selected_for_run"])
        split = str(row["split"])
        if selected != (split in {"train", "validation"}):
            raise ValueError("selected_for_run contradicts split")
        if selected and not actual_eligible:
            raise ValueError("selected_for_run requires strict eligibility")
        if split == "train":
            train_users.add(str(row["user_id"]))
        elif split == "validation":
            validation_users.add(str(row["user_id"]))
        elif split:
            raise ValueError("Training index split value is invalid")
    if train_users & validation_users:
        raise ValueError("Training index train and validation users are not disjoint")


def _validate_training_index_split_membership(
    training_index: pd.DataFrame,
    split_definition: Mapping[str, object],
) -> None:
    train_users, validation_users = _split_users(split_definition)
    for row in training_index.to_dict(orient="records"):
        eligible = _parse_bool(
            row["eligible_for_strict_training"],
            "eligible_for_strict_training",
        )
        user_id = str(row["user_id"])
        expected_split = ""
        if eligible and user_id in train_users:
            expected_split = "train"
        elif eligible and user_id in validation_users:
            expected_split = "validation"
        if str(row["split"]) != expected_split:
            raise ValueError("Training index split contradicts split definition")
        selected = _parse_selected_bool(row["selected_for_run"])
        if selected != bool(expected_split):
            raise ValueError("Training index selection contradicts split definition")


def _validate_manifest_artifacts(training_index: pd.DataFrame, stage2_root: Path) -> None:
    root = Path(stage2_root).resolve(strict=True)
    for relpath in training_index.loc[
        training_index["stage2_npz_relpath"].astype(str).str.len() > 0,
        "stage2_npz_relpath",
    ]:
        relative = Path(str(relpath))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Selected Stage 2 artifact path is unsafe")
        try:
            artifact = (root / relative).resolve(strict=True)
        except FileNotFoundError as error:
            raise ValueError("Stage 2 artifact declared by manifest is missing") from error
        try:
            artifact.relative_to(root)
        except ValueError as error:
            raise ValueError("Selected Stage 2 artifact escapes Stage 2 root") from error
        if not artifact.is_file():
            raise ValueError("Selected Stage 2 artifact is missing")


def _write_csv_atomic(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.parent / f".tmp-{path.name}-{os.getpid()}"
    try:
        frame.to_csv(temporary, index=False, encoding="utf-8-sig", lineterminator="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_staged_training_artifacts(
    staging_dir: Path,
    *,
    expected_class_order: ClassOrderContract,
    expected_training_index: pd.DataFrame,
    expected_metadata: Mapping[str, object],
    split_definition: Mapping[str, object],
    split_path: Path,
    source_stage2_manifest_path: Path,
    stage2_contract_sha256: str,
    repository_root: Path,
) -> None:
    names = {path.name for path in staging_dir.iterdir()}
    if names != TRAINING_ARTIFACT_NAMES:
        raise ValueError("Staged training artifact file set is invalid")
    class_order = load_class_order(staging_dir / "class_order.json")
    training_index = pd.read_csv(
        staging_dir / "training_index.csv",
        encoding="utf-8-sig",
        keep_default_na=False,
    )
    metadata = _strict_json(staging_dir / "training_index.json")
    if class_order.to_payload() != expected_class_order.to_payload():
        raise ValueError("Staged class order differs from generated contract")
    try:
        pd.testing.assert_frame_equal(
            training_index.reset_index(drop=True),
            expected_training_index.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError("Staged training index differs from generated index") from error
    if metadata != dict(expected_metadata):
        raise ValueError("Staged training metadata differs from generated metadata")
    validate_training_index_metadata(
        metadata,
        training_index,
        class_order,
        expected_source_manifest_sha256=sha256_file(source_stage2_manifest_path),
        expected_stage2_contract_sha256=stage2_contract_sha256,
        expected_split_definition_sha256=sha256_file(split_path),
        expected_fold=split_definition.get("fold"),
        expected_split_definition_path=_repository_relative(
            split_path,
            repository_root,
        ),
        expected_source_manifest_path="manifest.csv",
        expected_split_definition=split_definition,
    )


def _cleanup_generated_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def _publish_training_artifacts(
    output_dir: Path,
    *,
    class_order: ClassOrderContract,
    training_index: pd.DataFrame,
    metadata: Mapping[str, object],
    split_definition: Mapping[str, object],
    split_path: Path,
    source_stage2_manifest_path: Path,
    stage2_contract_sha256: str,
    repository_root: Path,
) -> None:
    output_dir = Path(output_dir)
    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Training index output directory must be missing or empty")
    if list(parent.glob(f".{output_dir.name}.staging-*")) or list(
        parent.glob(f".{output_dir.name}.backup-*")
    ):
        raise FileExistsError("Unknown training artifact staging or backup residue exists")

    token = uuid4().hex
    staging = parent / f".{output_dir.name}.staging-{token}"
    backup = parent / f".{output_dir.name}.backup-{token}"
    staging.mkdir(exist_ok=False)
    had_empty_output = output_dir.exists()
    backup_moved = False
    installed = False
    try:
        write_json_atomic(staging / "class_order.json", class_order.to_payload())
        _write_csv_atomic(staging / "training_index.csv", training_index)
        write_json_atomic(staging / "training_index.json", dict(metadata))
        _validate_staged_training_artifacts(
            staging,
            expected_class_order=class_order,
            expected_training_index=training_index,
            expected_metadata=metadata,
            split_definition=split_definition,
            split_path=split_path,
            source_stage2_manifest_path=source_stage2_manifest_path,
            stage2_contract_sha256=stage2_contract_sha256,
            repository_root=repository_root,
        )
        if had_empty_output:
            os.replace(output_dir, backup)
            backup_moved = True
        os.replace(staging, output_dir)
        installed = True
        if backup_moved:
            backup.rmdir()
            backup_moved = False
    except BaseException as error:
        try:
            if installed and backup_moved:
                os.replace(output_dir, staging)
                installed = False
            if backup_moved:
                os.replace(backup, output_dir)
                backup_moved = False
            _cleanup_generated_tree(staging)
        except BaseException as restore_error:
            raise RuntimeError(
                "Training artifact transaction recovery failed for "
                f"output={output_dir} staging={staging} backup={backup}"
            ) from restore_error
        raise


def generate_training_index_artifacts(
    stage2_manifest_path: Path,
    output_dir: Path,
    split_path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
) -> dict[str, object]:
    stage2_manifest_path = _canonical_stage2_manifest(stage2_manifest_path)
    split_path = Path(split_path).resolve(strict=True)
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Training index output directory must be missing or empty")
    manifest = pd.read_csv(stage2_manifest_path, encoding="utf-8-sig", keep_default_na=False)
    split_definition = _strict_json(split_path)
    schema = load_stage2_schema(stage2_manifest_path.parent / "schema.json")
    class_order = build_class_order(manifest)
    training_index = build_training_index(manifest, class_order, split_definition)
    _validate_manifest_artifacts(training_index, stage2_manifest_path.parent)
    metadata = build_training_index_metadata(
        training_index,
        class_order,
        split_definition=split_definition,
        split_path=split_path,
        source_stage2_manifest_path=stage2_manifest_path,
        stage2_contract_sha256=str(schema["contract_sha256"]),
        repository_root=repository_root,
    )
    _publish_training_artifacts(
        output_dir,
        class_order=class_order,
        training_index=training_index,
        metadata=metadata,
        split_definition=split_definition,
        split_path=split_path,
        source_stage2_manifest_path=stage2_manifest_path,
        stage2_contract_sha256=str(schema["contract_sha256"]),
        repository_root=repository_root,
    )
    return metadata


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build canonical IMU Stage 2 training index")
    parser.add_argument("--stage2-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=REPOSITORY_ROOT / "metadata" / "splits" / "fold_0.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generate_training_index_artifacts(
        args.stage2_manifest,
        args.output_dir,
        args.split_file,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
