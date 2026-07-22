from __future__ import annotations

import json
import csv
import os
import re
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
import torch
import yaml

from scripts.build_imu_training_index import ClassOrderContract, load_class_order
from scripts.compute_imu_normalization import validate_normalization_artifacts
from src.data.imu_stage1_bridge import process_raw_imu_source
from src.data.imu_stage2_contracts import (
    ImuActionSource,
    ImuPathNotDirectoryError,
    InferenceSample,
    MissingImuDirectoryError,
    NoRecognizableImuCsvError,
    NoUsableGridCellsError,
    NoValidStage1RecordsError,
    SequenceLengthSafetyError,
    Stage1ActionData,
    Stage1DataValidationError,
    TestSampleDescriptor,
    contract_sha256,
    sha256_file,
)
from src.data.imu_stage2_core import process_stage2_action
from src.data.imu_stage2_io import load_stage2_schema
from src.models.imu_stage2_tcn import (
    CHECKPOINT_HASH_FIELDS,
    build_checkpoint_metadata,
)


SAMPLE_ID_PATTERN = re.compile(r"^SM_test_\d{4}$")
SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")
BUNDLE_MANIFEST_VERSION = "imu-inference-bundle-v1"
BUNDLE_MANIFEST_NAME = "inference_bundle_manifest.json"
BUNDLE_ROLES = (
    "checkpoint",
    "model_config",
    "stage2_schema",
    "normalization_npz",
    "normalization_json",
    "class_order",
    "submission_contract",
    "inference_config",
)
DEGRADABLE_ERROR_TYPES = (
    MissingImuDirectoryError,
    ImuPathNotDirectoryError,
    NoRecognizableImuCsvError,
    NoValidStage1RecordsError,
    Stage1DataValidationError,
    NoUsableGridCellsError,
    SequenceLengthSafetyError,
)

SUBMISSION_CONTRACT_VERSION = "imu-submission-v1"
SUBMISSION_REPRESENTATIONS = frozenset({"class_id", "class_name", "label_index"})
INFERENCE_CONFIG_CONTRACT = {
    "config_version": "imu-stage2-inference-v1",
    "hard_safety_limit_t": 10_000,
    "inference_seed": 20260715,
    "deterministic_algorithms": True,
    "batch_feature_budget": 327680,
    "maximum_batch_size": 16,
    "model_output_type": "logits",
    "prediction_rule": "argmax",
    "imu_unavailable_policy": "packaged_null_embedding",
}


@dataclass(frozen=True)
class TestSampleDiscoveryResult:
    samples: tuple[TestSampleDescriptor, ...]
    ignored_entries: tuple[str, ...]
    sample_ignored_entries: tuple[str, ...]


@dataclass(frozen=True)
class InferencePreprocessDiagnostics:
    sample: InferenceSample
    source_status: str
    stage1_status: str
    stage2_status: str
    stage1_result: Stage1ActionData | None
    stage2_result: object | None
    degradation_error: Exception | None


@dataclass(frozen=True)
class InferenceBundle:
    root: Path
    paths: Mapping[str, Path]
    manifest: Mapping[str, object]
    stage2_schema: Mapping[str, object]
    normalization_metadata: Mapping[str, object]
    normalization_arrays: Mapping[str, np.ndarray]
    class_order: ClassOrderContract
    submission_contract: Mapping[str, object]
    model_config: Mapping[str, object]
    inference_config: Mapping[str, object]
    checkpoint_metadata: Mapping[str, object]


def derive_submission_contract(sample_submission_path: Path) -> dict[str, object]:
    path = Path(sample_submission_path).resolve(strict=True)
    if not path.is_file():
        raise ValueError("Sample submission must be a regular file")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if len(rows) < 2 or len(rows[0]) != 2:
        raise ValueError("Sample submission must have two columns and at least one row")
    columns = rows[0]
    if any(not column for column in columns) or len(set(columns)) != 2:
        raise ValueError("Sample submission columns must be distinct and non-empty")
    if any(len(row) != 2 for row in rows[1:]):
        raise ValueError("Sample submission row width does not match its header")
    sample_id_column, prediction_column = columns
    if prediction_column not in SUBMISSION_REPRESENTATIONS:
        raise ValueError("Sample submission prediction representation is unsupported")
    sample_ids = [row[0] for row in rows[1:]]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(
        sample_ids
    ):
        raise ValueError("Sample submission IDs must be unique and non-empty")
    contract: dict[str, object] = {
        "submission_contract_version": SUBMISSION_CONTRACT_VERSION,
        "columns": columns,
        "sample_id_column": sample_id_column,
        "prediction_column": prediction_column,
        "encoding": "utf-8",
        "header": True,
        "row_order": "sample_submission",
        "sample_ids": sample_ids,
        "prediction_representation": prediction_column,
    }
    return {
        "contract": contract,
        "submission_contract_sha256": contract_sha256(contract),
    }


def build_inference_bundle_manifest(
    bundle_root: Path,
    artifact_paths: Mapping[str, Path],
) -> dict[str, object]:
    root = Path(bundle_root).resolve(strict=True)
    if set(artifact_paths) != set(BUNDLE_ROLES):
        raise ValueError("Inference bundle artifact roles do not match contract")
    files: dict[str, object] = {}
    for role in BUNDLE_ROLES:
        path = Path(artifact_paths[role]).resolve(strict=True)
        try:
            relative = path.relative_to(root)
        except ValueError as error:
            raise ValueError("Managed artifact is outside the inference bundle") from error
        if not path.is_file() or any(part in {"", ".", ".."} for part in relative.parts):
            raise ValueError("Managed artifact must be a regular bundle file")
        files[role] = {"path": relative.as_posix(), "sha256": sha256_file(path)}
    return {"bundle_manifest_version": BUNDLE_MANIFEST_VERSION, "files": files}


def validate_logits(
    logits: torch.Tensor,
    *,
    batch_size: int,
    num_classes: int,
) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        raise ValueError("logits must be a torch.Tensor")
    if logits.shape != (batch_size, num_classes):
        raise ValueError("logits must have shape [B,num_classes]")
    if not torch.is_floating_point(logits):
        raise ValueError("logits must use a floating-point dtype")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    return logits


def validate_inference_config(config: Mapping[str, object]) -> dict[str, object]:
    if set(config) != set(INFERENCE_CONFIG_CONTRACT):
        raise ValueError("Inference configuration fields do not match contract")
    for key, expected in INFERENCE_CONFIG_CONTRACT.items():
        actual = config[key]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Inference configuration {key} is incompatible")
    return dict(config)


def decode_predictions(
    logits_or_indices: torch.Tensor,
    class_order: ClassOrderContract,
    submission_contract: Mapping[str, object],
) -> list[object]:
    tensor = torch.as_tensor(logits_or_indices)
    if tensor.ndim == 2:
        tensor = torch.argmax(tensor, dim=1)
    if tensor.ndim != 1:
        raise ValueError("Predictions must be logits or one-dimensional indices")
    indices = [int(value) for value in tensor.detach().cpu().tolist()]
    if any(index < 0 or index >= class_order.num_classes for index in indices):
        raise ValueError("Prediction index is outside class order")
    representation = submission_contract.get("prediction_representation")
    if representation not in SUBMISSION_REPRESENTATIONS:
        raise ValueError("Submission prediction representation is unsupported")
    records = list(class_order.classes)
    if representation == "label_index":
        return indices
    return [records[index][str(representation)] for index in indices]


def _validate_submission_contract_mapping(contract: Mapping[str, object]) -> None:
    required = {
        "submission_contract_version",
        "columns",
        "sample_id_column",
        "prediction_column",
        "encoding",
        "header",
        "row_order",
        "sample_ids",
        "prediction_representation",
    }
    if set(contract) != required:
        raise ValueError("Submission contract fields do not match contract")
    columns = contract["columns"]
    sample_ids = contract["sample_ids"]
    if (
        contract["submission_contract_version"] != SUBMISSION_CONTRACT_VERSION
        or not isinstance(columns, list)
        or len(columns) != 2
        or contract["sample_id_column"] != columns[0]
        or contract["prediction_column"] != columns[1]
        or contract["prediction_representation"] != columns[1]
        or contract["prediction_representation"] not in SUBMISSION_REPRESENTATIONS
        or contract["encoding"] != "utf-8"
        or contract["header"] is not True
        or contract["row_order"] != "sample_submission"
        or not isinstance(sample_ids, list)
        or not all(isinstance(value, str) and value for value in sample_ids)
        or len(set(sample_ids)) != len(sample_ids)
    ):
        raise ValueError("Submission contract is incompatible")


def validate_submission_file(path: Path, contract: Mapping[str, object]) -> None:
    _validate_submission_contract_mapping(contract)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    columns = list(contract["columns"])
    sample_ids = list(contract["sample_ids"])
    if not rows or rows[0] != columns or len(rows) != len(sample_ids) + 1:
        raise ValueError("Submission header or row count violates contract")
    actual_ids: list[str] = []
    representation = str(contract["prediction_representation"])
    for row in rows[1:]:
        if len(row) != 2:
            raise ValueError("Submission row width violates contract")
        actual_ids.append(row[0])
        if representation in {"class_id", "label_index"}:
            try:
                int(row[1])
            except ValueError as error:
                raise ValueError("Submission prediction must be an integer") from error
        elif not row[1]:
            raise ValueError("Submission prediction must be non-empty")
    if actual_ids != sample_ids:
        raise ValueError("Submission sample IDs or row order violate contract")


def write_submission_atomic(
    output_path: Path,
    rows: Sequence[tuple[str, object]],
    contract: Mapping[str, object],
    *,
    overwrite: bool = False,
) -> None:
    _validate_submission_contract_mapping(contract)
    output = Path(output_path)
    if output.exists() and not overwrite:
        raise FileExistsError(output)
    if not output.parent.exists() or not output.parent.is_dir():
        raise FileNotFoundError(output.parent)
    expected_ids = list(contract["sample_ids"])
    if [sample_id for sample_id, _ in rows] != expected_ids:
        raise ValueError("Submission rows do not match the contracted sample order")
    temporary = output.parent / f".{output.name}.tmp-{uuid.uuid4().hex}"
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(list(contract["columns"]))
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        validate_submission_file(temporary, contract)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def _natural_key(value: str) -> tuple[tuple[int, object], ...]:
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part.casefold())
        for part in re.split(r"(\d+)", value)
        if part
    )


def discover_test_samples(test_root: Path) -> TestSampleDiscoveryResult:
    root = Path(test_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    samples: list[TestSampleDescriptor] = []
    ignored: list[str] = []
    sample_ignored: list[str] = []
    seen_ids: set[str] = set()
    for entry in sorted(root.iterdir(), key=lambda path: _natural_key(path.name)):
        if entry.is_dir() and SAMPLE_ID_PATTERN.fullmatch(entry.name):
            if entry.name in seen_ids:
                raise ValueError(f"Duplicate test sample ID: {entry.name}")
            seen_ids.add(entry.name)
            samples.append(
                TestSampleDescriptor(
                    sample_id=entry.name,
                    sample_directory=entry,
                    source_relative_path=Path(entry.name),
                )
            )
            for child in sorted(entry.iterdir(), key=lambda path: _natural_key(path.name)):
                if child.name != "IMU":
                    sample_ignored.append(child.relative_to(root).as_posix())
                    continue
                if child.is_dir():
                    for imu_entry in sorted(
                        child.iterdir(), key=lambda path: _natural_key(path.name)
                    ):
                        if not (
                            imu_entry.is_file()
                            and imu_entry.suffix.casefold() == ".csv"
                        ):
                            sample_ignored.append(imu_entry.relative_to(root).as_posix())
        else:
            ignored.append(entry.name)
    samples.sort(key=lambda item: _natural_key(item.sample_id))
    return TestSampleDiscoveryResult(
        tuple(samples), tuple(ignored), tuple(sorted(sample_ignored, key=_natural_key))
    )


def adapt_raw_imu_source(descriptor: TestSampleDescriptor) -> ImuActionSource:
    imu_directory = descriptor.sample_directory / "IMU"
    if not imu_directory.exists():
        raise MissingImuDirectoryError(
            descriptor.sample_id,
            "IMU directory is missing",
        )
    if not imu_directory.is_dir():
        raise ImuPathNotDirectoryError(
            descriptor.sample_id,
            "IMU path is not a directory",
        )
    discovered = sorted(
        (
            path
            for path in imu_directory.iterdir()
            if path.is_file() and path.suffix.casefold() == ".csv"
        ),
        key=lambda path: _natural_key(path.name),
    )
    unique: list[Path] = []
    seen_paths: set[Path] = set()
    for path in discovered:
        identity = path.resolve(strict=True)
        if identity not in seen_paths:
            seen_paths.add(identity)
            unique.append(path)
    if not unique:
        raise NoRecognizableImuCsvError(
            descriptor.sample_id,
            "No recognizable direct IMU CSV files",
        )
    return ImuActionSource(
        sample_id=descriptor.sample_id,
        input_directory=imu_directory,
        input_csv_files=tuple(unique),
        source_relative_path=descriptor.source_relative_path / "IMU",
        action_id=descriptor.sample_id,
    )


def preprocess_inference_sample(
    descriptor: TestSampleDescriptor,
    *,
    hard_safety_limit_t: int = 10_000,
) -> InferenceSample:
    diagnostics = preprocess_inference_sample_with_diagnostics(
        descriptor,
        hard_safety_limit_t=hard_safety_limit_t,
    )
    return diagnostics.sample


def preprocess_inference_sample_with_diagnostics(
    descriptor: TestSampleDescriptor,
    *,
    hard_safety_limit_t: int = 10_000,
) -> InferencePreprocessDiagnostics:
    try:
        source = adapt_raw_imu_source(descriptor)
    except DEGRADABLE_ERROR_TYPES as error:
        return InferencePreprocessDiagnostics(
            sample=InferenceSample(descriptor.sample_id, None, False, False),
            source_status="unavailable",
            stage1_status="unavailable",
            stage2_status="unavailable",
            stage1_result=None,
            stage2_result=None,
            degradation_error=error,
        )
    try:
        stage1 = process_raw_imu_source(source)
    except DEGRADABLE_ERROR_TYPES as error:
        return InferencePreprocessDiagnostics(
            sample=InferenceSample(descriptor.sample_id, None, False, False),
            source_status="available",
            stage1_status="degraded",
            stage2_status="unavailable",
            stage1_result=None,
            stage2_result=None,
            degradation_error=error,
        )
    try:
        result = process_stage2_action(
            stage1,
            hard_safety_limit_t=hard_safety_limit_t,
        )
        result.validate()
        if result.sample_id != descriptor.sample_id:
            raise ValueError("Stage 2 sample ID disagrees with discovery descriptor")
        if not result.imu_usable:
            raise NoUsableGridCellsError(
                descriptor.sample_id,
                "Stage 2 has no usable grid cells",
            )
        sample = InferenceSample(
                sample_id=descriptor.sample_id,
                imu_result=result,
                imu_available=True,
                modality_mask=True,
        )
        return InferencePreprocessDiagnostics(
            sample=sample,
            source_status="available",
            stage1_status="success",
            stage2_status="success",
            stage1_result=stage1,
            stage2_result=result,
            degradation_error=None,
        )
    except DEGRADABLE_ERROR_TYPES as error:
        return InferencePreprocessDiagnostics(
            sample=InferenceSample(descriptor.sample_id, None, False, False),
            source_status="available",
            stage1_status="success",
            stage2_status="degraded",
            stage1_result=stage1,
            stage2_result=None,
            degradation_error=error,
        )


def collate_inference_samples(
    samples: Sequence[InferenceSample],
) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty inference batch")
    lengths: list[int] = []
    for sample in samples:
        if sample.imu_available != sample.modality_mask:
            raise ValueError("IMU availability and modality mask disagree")
        if sample.imu_available:
            if sample.imu_result is None:
                raise ValueError("Available IMU sample requires a Stage 2 result")
            sample.imu_result.validate()
            if sample.imu_result.sample_id != sample.sample_id:
                raise ValueError("Inference sample ID disagrees with Stage 2 result")
            lengths.append(int(sample.imu_result.values.shape[0]))
        else:
            if sample.imu_result is not None:
                raise ValueError("Unavailable IMU sample cannot carry a Stage 2 result")
            lengths.append(0)

    batch_size = len(samples)
    batch_t = max(1, max(lengths))
    values = torch.zeros((batch_size, batch_t, 5, 16), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, batch_t, 5), dtype=torch.bool)
    sequence_mask = torch.zeros((batch_size, batch_t), dtype=torch.bool)
    sensor_mask = torch.zeros((batch_size, 5), dtype=torch.bool)
    usable_sensor_mask = torch.zeros((batch_size, 5), dtype=torch.bool)
    timestamps_ms = torch.full((batch_size, batch_t), -1, dtype=torch.int64)
    modality_mask = torch.as_tensor(
        [sample.modality_mask for sample in samples], dtype=torch.bool
    )
    for batch_index, (sample, length) in enumerate(zip(samples, lengths, strict=True)):
        if length == 0:
            continue
        result = sample.imu_result
        assert result is not None
        finite_values = result.values.copy()
        finite_values[~result.valid_mask] = 0.0
        values[batch_index, :length] = torch.from_numpy(finite_values)
        valid_mask[batch_index, :length] = torch.from_numpy(result.valid_mask.copy())
        sequence_mask[batch_index, :length] = True
        sensor_mask[batch_index] = torch.from_numpy(result.sensor_mask.copy())
        usable_sensor_mask[batch_index] = torch.from_numpy(
            result.usable_sensor_mask.copy()
        )
        timestamps_ms[batch_index, :length] = torch.from_numpy(
            result.timestamps_ms.copy()
        )
    return {
        "values": values,
        "valid_mask": valid_mask,
        "sequence_mask": sequence_mask,
        "sensor_mask": sensor_mask,
        "usable_sensor_mask": usable_sensor_mask,
        "timestamps_ms": timestamps_ms,
        "lengths": torch.as_tensor(lengths, dtype=torch.int64),
        "sample_id": [sample.sample_id for sample in samples],
        "imu_modality_mask": modality_mask,
    }


def _load_json_strict(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path.name}")
    return payload


def _load_yaml_mapping(path: Path, name: str) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or not all(isinstance(key, str) for key in payload):
        raise ValueError(f"{name} must be a YAML mapping")
    return dict(payload)


def _normalize_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a 64-character SHA-256")
    return value.lower()


def _bundle_relative_path(root: Path, value: object, role: str) -> Path:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Bundle {role} path must be a relative POSIX path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Bundle {role} path must be a relative POSIX path")
    candidate = (root / Path(*relative.parts)).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise ValueError(f"Bundle {role} path must be a relative POSIX path") from error
    if not candidate.is_file():
        raise ValueError(f"Bundle {role} must be a regular file")
    return candidate


def _verified_bundle_paths(
    root: Path,
    manifest: Mapping[str, object],
) -> dict[str, Path]:
    if set(manifest) != {"bundle_manifest_version", "files"}:
        raise ValueError("Inference bundle manifest keys do not match contract")
    if manifest["bundle_manifest_version"] != BUNDLE_MANIFEST_VERSION:
        raise ValueError("Inference bundle manifest version is incompatible")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(BUNDLE_ROLES):
        raise ValueError("Inference bundle roles do not match contract")
    paths: dict[str, Path] = {}
    expected_hashes: dict[str, str] = {}
    seen_paths: set[Path] = set()
    for role in BUNDLE_ROLES:
        entry = files[role]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ValueError(f"Bundle {role} entry keys do not match contract")
        path = _bundle_relative_path(root, entry["path"], role)
        if path in seen_paths:
            raise ValueError("Inference bundle roles must use distinct files")
        seen_paths.add(path)
        paths[role] = path
        expected_hashes[role] = _normalize_sha256(
            entry["sha256"], f"Bundle {role} SHA-256"
        )
    for role in BUNDLE_ROLES:
        if sha256_file(paths[role]) != expected_hashes[role]:
            raise ValueError(f"Bundle {role} SHA-256 mismatch")
    return paths


def _load_checkpoint_metadata(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Checkpoint must contain a mapping")
    metadata = payload.get("checkpoint_metadata")
    if not isinstance(metadata, dict) or set(metadata) != {
        "checkpoint_metadata_version",
        "num_classes",
        *CHECKPOINT_HASH_FIELDS,
    }:
        raise ValueError("Checkpoint metadata keys do not match contract")
    if metadata.get("checkpoint_metadata_version") != "imu-checkpoint-v1":
        raise ValueError("Checkpoint metadata version is incompatible")
    return build_checkpoint_metadata(
        stage2_contract_sha256=metadata["stage2_contract_sha256"],
        training_index_sha256=metadata["training_index_sha256"],
        normalization_contract_sha256=metadata["normalization_contract_sha256"],
        normalization_file_sha256=metadata["normalization_file_sha256"],
        class_order_sha256=metadata["class_order_sha256"],
        submission_contract_sha256=metadata["submission_contract_sha256"],
        num_classes=metadata["num_classes"],
    )


def _validate_submission_contract(path: Path) -> dict[str, object]:
    payload = _load_json_strict(path)
    if set(payload) != {"contract", "submission_contract_sha256"}:
        raise ValueError("Submission contract keys do not match contract")
    contract = payload["contract"]
    if not isinstance(contract, dict):
        raise ValueError("Submission contract payload must be an object")
    expected = contract_sha256(contract)
    actual = _normalize_sha256(
        payload["submission_contract_sha256"],
        "submission_contract_sha256",
    )
    if actual != expected:
        raise ValueError("submission_contract_sha256 mismatch")
    payload["submission_contract_sha256"] = actual
    return payload


def load_inference_bundle(bundle_root: Path) -> InferenceBundle:
    root = Path(bundle_root).resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    manifest = _load_json_strict(root / BUNDLE_MANIFEST_NAME)
    paths = _verified_bundle_paths(root, manifest)

    stage2_schema = load_stage2_schema(paths["stage2_schema"])
    normalization_metadata = _load_json_strict(paths["normalization_json"])
    normalization_contract = normalization_metadata.get("contract")
    normalization_provenance = normalization_metadata.get("provenance")
    if not isinstance(normalization_contract, dict) or not isinstance(
        normalization_provenance, dict
    ):
        raise ValueError("Normalization metadata structure is invalid")
    normalization_arrays = validate_normalization_artifacts(
        paths["normalization_npz"],
        paths["normalization_json"],
        expected_stage2_contract_sha256=str(stage2_schema["contract_sha256"]),
        expected_training_index_sha256=str(
            normalization_contract.get("training_index_sha256")
        ),
        expected_train_sample_id_sha256=str(
            normalization_contract.get("train_sample_id_sha256")
        ),
        expected_fold=normalization_contract.get("fold"),
        expected_train_users=normalization_contract.get("train_users", []),
        expected_source_stage2_manifest_sha256=str(
            normalization_provenance.get("source_stage2_manifest_sha256")
        ),
    )
    class_order = load_class_order(paths["class_order"])
    submission_contract = _validate_submission_contract(
        paths["submission_contract"]
    )
    model_config = _load_yaml_mapping(paths["model_config"], "Model config")
    inference_config = _load_yaml_mapping(
        paths["inference_config"], "Inference config"
    )
    if inference_config.get("hard_safety_limit_t") != 10_000:
        raise ValueError("Inference hard_safety_limit_t is incompatible")
    checkpoint_metadata = _load_checkpoint_metadata(paths["checkpoint"])

    expected_bindings = {
        "stage2_contract_sha256": stage2_schema["contract_sha256"],
        "training_index_sha256": normalization_contract.get(
            "training_index_sha256"
        ),
        "normalization_contract_sha256": normalization_metadata.get(
            "normalization_contract_sha256"
        ),
        "normalization_file_sha256": normalization_metadata.get(
            "normalization_file_sha256"
        ),
        "class_order_sha256": class_order.class_order_sha256,
        "submission_contract_sha256": submission_contract[
            "submission_contract_sha256"
        ],
    }
    for name, expected in expected_bindings.items():
        if checkpoint_metadata.get(name) != expected:
            raise ValueError(f"Checkpoint {name} mismatch")
    if checkpoint_metadata.get("num_classes") != class_order.num_classes:
        raise ValueError("Checkpoint num_classes mismatch")
    return InferenceBundle(
        root=root,
        paths=paths,
        manifest=manifest,
        stage2_schema=stage2_schema,
        normalization_metadata=normalization_metadata,
        normalization_arrays=normalization_arrays,
        class_order=class_order,
        submission_contract=submission_contract,
        model_config=model_config,
        inference_config=inference_config,
        checkpoint_metadata=checkpoint_metadata,
    )
