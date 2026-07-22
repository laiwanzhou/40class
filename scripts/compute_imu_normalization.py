from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.build_imu_training_index import hash_training_index
from src.data.imu_stage2_contracts import (
    FEATURE_ORDER,
    SENSOR_ORDER,
    DataStatus,
    canonical_json_bytes,
    sha256_file,
)
from src.data.imu_stage2_io import load_and_validate_npz, load_stage2_schema, write_json_atomic


NORMALIZATION_VERSION = "imu-normalization-v1"
NEAR_CONSTANT_THRESHOLD = 1e-6
NPZ_KEYS = (
    "count",
    "mean",
    "raw_std",
    "applied_scale",
    "near_constant_mask",
    "minimum",
    "maximum",
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _hash_sample_ids(sample_ids: Sequence[str]) -> str:
    payload = "".join(f"{sample_id}\n" for sample_id in sorted(map(str, sample_ids)))
    return _sha256_bytes(payload.encode("utf-8"))


class StreamingMoments:
    def __init__(self) -> None:
        self.count = np.zeros((5, 16), dtype=np.int64)
        self.mean = np.zeros((5, 16), dtype=np.float64)
        self.m2 = np.zeros((5, 16), dtype=np.float64)
        self.minimum = np.full((5, 16), np.inf, dtype=np.float64)
        self.maximum = np.full((5, 16), -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray, valid_mask: np.ndarray) -> None:
        values = np.asarray(values)
        valid_mask = np.asarray(valid_mask)
        if values.dtype != np.float64 or values.ndim != 3 or values.shape[1:] != (5, 16):
            raise ValueError("values must be float64 with shape [T,5,16]")
        if valid_mask.dtype != np.bool_ or valid_mask.shape != values.shape[:2]:
            raise ValueError("valid_mask must be bool with shape [T,5]")
        if not np.isfinite(values[valid_mask]).all():
            raise ValueError("Valid normalization cells must be finite")
        for sensor_index in range(5):
            batch = values[valid_mask[:, sensor_index], sensor_index, :]
            if not len(batch):
                continue
            batch_count = len(batch)
            batch_mean = batch.mean(axis=0, dtype=np.float64)
            centered = batch - batch_mean
            batch_m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
            old_count = self.count[sensor_index].astype(np.float64)
            total = old_count + batch_count
            delta = batch_mean - self.mean[sensor_index]
            self.mean[sensor_index] += delta * (batch_count / total)
            self.m2[sensor_index] += (
                batch_m2 + delta * delta * old_count * batch_count / total
            )
            self.count[sensor_index] += batch_count
            self.minimum[sensor_index] = np.minimum(
                self.minimum[sensor_index], batch.min(axis=0)
            )
            self.maximum[sensor_index] = np.maximum(
                self.maximum[sensor_index], batch.max(axis=0)
            )

    def finalize(self) -> dict[str, np.ndarray]:
        if np.any(self.count == 0):
            raise ValueError("Normalization has a zero count sensor-feature")
        variance = self.m2 / self.count
        tolerance = np.finfo(np.float64).eps * np.maximum(1.0, self.mean * self.mean) * 16
        if np.any(variance < -tolerance):
            raise ValueError("Normalization variance is meaningfully negative")
        variance = np.maximum(variance, 0.0)
        raw_std = np.sqrt(variance)
        near_constant = raw_std < NEAR_CONSTANT_THRESHOLD
        applied_scale = np.where(near_constant, 1.0, raw_std)
        arrays = {
            "count": self.count.copy(),
            "mean": self.mean.copy(),
            "raw_std": raw_std,
            "applied_scale": applied_scale,
            "near_constant_mask": near_constant,
            "minimum": self.minimum.copy(),
            "maximum": self.maximum.copy(),
        }
        if not all(np.isfinite(array).all() for key, array in arrays.items() if key != "near_constant_mask"):
            raise ValueError("Normalization statistics must be finite")
        if np.any(arrays["minimum"] > arrays["maximum"]):
            raise ValueError("Normalization minimum exceeds maximum")
        for sensor_count in self.count:
            if not np.all(sensor_count == sensor_count[0]):
                raise ValueError("Per-sensor feature counts must be equal")
        return arrays


def compute_normalization(
    training_index: pd.DataFrame,
    stage2_root: Path,
) -> dict[str, np.ndarray]:
    required = {
        "sample_id",
        "stage2_npz_relpath",
        "status",
        "selected_for_run",
        "split",
    }
    if not required.issubset(training_index.columns):
        raise ValueError("Training index is missing normalization columns")
    selected_train = training_index[
        training_index["selected_for_run"].map(_as_bool)
        & (training_index["split"].astype(str) == "train")
    ]
    if selected_train.empty:
        raise ValueError("Normalization requires selected training samples")
    root = Path(stage2_root).resolve(strict=True)
    moments = StreamingMoments()
    for row in selected_train.to_dict(orient="records"):
        relative = Path(str(row["stage2_npz_relpath"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Training Stage 2 NPZ path is unsafe")
        path = (root / relative).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("Training Stage 2 NPZ escapes Stage 2 root") from error
        result = load_and_validate_npz(
            path,
            sample_id=str(row["sample_id"]),
            status=DataStatus(str(row["status"])),
            qc={},
        )
        moments.update(result.values.astype(np.float64), result.valid_mask)
    return moments.finalize()


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError("Boolean contract value is invalid")


def _cast_for_storage(statistics: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    if set(statistics) != set(NPZ_KEYS):
        raise ValueError("Normalization statistic keys do not match contract")
    return {
        "count": np.asarray(statistics["count"], dtype=np.int64),
        "mean": np.asarray(statistics["mean"], dtype=np.float32),
        "raw_std": np.asarray(statistics["raw_std"], dtype=np.float32),
        "applied_scale": np.asarray(statistics["applied_scale"], dtype=np.float32),
        "near_constant_mask": np.asarray(statistics["near_constant_mask"], dtype=bool),
        "minimum": np.asarray(statistics["minimum"], dtype=np.float32),
        "maximum": np.asarray(statistics["maximum"], dtype=np.float32),
    }


def _normalization_contract(
    *,
    stage2_contract_sha256: str,
    training_index_sha256: str,
    train_sample_id_sha256: str,
    fold: object,
    train_users: Sequence[str],
) -> dict[str, object]:
    return {
        "normalization_version": NORMALIZATION_VERSION,
        "stage2_contract_sha256": str(stage2_contract_sha256).lower(),
        "training_index_sha256": str(training_index_sha256).lower(),
        "train_sample_id_sha256": str(train_sample_id_sha256).lower(),
        "fold": fold,
        "train_users": sorted(map(str, train_users)),
        "sensor_order": list(SENSOR_ORDER),
        "feature_order": list(FEATURE_ORDER),
        "shape": [5, 16],
        "ddof": 0,
        "near_constant_threshold": NEAR_CONSTANT_THRESHOLD,
        "count_dtype": "int64",
        "statistics_dtype": "float32",
        "near_constant_mask_dtype": "bool",
    }


def _write_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.parent / f".tmp-{path.name}-{os.getpid()}.npz"
    try:
        np.savez(temporary, **arrays)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_normalization_artifacts(
    statistics: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    stage2_contract_sha256: str,
    training_index_sha256: str,
    train_sample_id_sha256: str,
    fold: object,
    train_users: Sequence[str],
    source_stage2_manifest_sha256: str,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Normalization output directory must be missing or empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _cast_for_storage(statistics)
    _validate_arrays(arrays)
    contract = _normalization_contract(
        stage2_contract_sha256=stage2_contract_sha256,
        training_index_sha256=training_index_sha256,
        train_sample_id_sha256=train_sample_id_sha256,
        fold=fold,
        train_users=train_users,
    )
    npz_path = output_dir / "imu_normalization.npz"
    _write_npz_atomic(npz_path, arrays)
    near_constant_features = [
        f"{sensor}/{feature}"
        for sensor_index, sensor in enumerate(SENSOR_ORDER)
        for feature_index, feature in enumerate(FEATURE_ORDER)
        if bool(arrays["near_constant_mask"][sensor_index, feature_index])
    ]
    metadata = {
        "contract": contract,
        "provenance": {
            "generator_script": "scripts/compute_imu_normalization.py",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_stage2_manifest_sha256": str(source_stage2_manifest_sha256).lower(),
        },
        "normalization_contract_sha256": _sha256_bytes(canonical_json_bytes(contract)),
        "normalization_file_sha256": sha256_file(npz_path),
        "near_constant_features": near_constant_features,
    }
    write_json_atomic(output_dir / "imu_normalization.json", metadata)
    return metadata


def _strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value is forbidden: {value}")

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, parse_constant=reject_constant)
    if not isinstance(payload, dict):
        raise ValueError("Normalization metadata must be an object")
    return payload


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        expected = {f"{key}.npy" for key in NPZ_KEYS}
        if {member.filename for member in members} != expected or len(members) != len(expected):
            raise ValueError("Normalization NPZ members do not match contract")
        if any(member.compress_type != zipfile.ZIP_STORED for member in members):
            raise ValueError("Normalization NPZ must be uncompressed")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(NPZ_KEYS):
            raise ValueError("Normalization NPZ keys do not match contract")
        return {key: archive[key].copy() for key in NPZ_KEYS}


def _validate_arrays(arrays: Mapping[str, np.ndarray]) -> None:
    expected_dtypes = {
        "count": np.dtype(np.int64),
        "mean": np.dtype(np.float32),
        "raw_std": np.dtype(np.float32),
        "applied_scale": np.dtype(np.float32),
        "near_constant_mask": np.dtype(bool),
        "minimum": np.dtype(np.float32),
        "maximum": np.dtype(np.float32),
    }
    if set(arrays) != set(NPZ_KEYS):
        raise ValueError("Normalization array keys do not match contract")
    for key, dtype in expected_dtypes.items():
        array = arrays[key]
        if array.shape != (5, 16) or array.dtype != dtype:
            raise ValueError(f"Normalization {key} shape or dtype mismatch")
    if np.any(arrays["count"] <= 0):
        raise ValueError("Normalization count must be positive")
    for row in arrays["count"]:
        if not np.all(row == row[0]):
            raise ValueError("Normalization per-sensor feature counts differ")
    for key in ("mean", "raw_std", "applied_scale", "minimum", "maximum"):
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Normalization {key} must be finite")
    if np.any(arrays["raw_std"] < 0) or np.any(arrays["applied_scale"] <= 0):
        raise ValueError("Normalization scales are invalid")
    expected_near = arrays["raw_std"] < NEAR_CONSTANT_THRESHOLD
    if not np.array_equal(arrays["near_constant_mask"], expected_near):
        raise ValueError("Normalization near_constant_mask mismatch")
    expected_scale = np.where(expected_near, 1.0, arrays["raw_std"])
    if not np.array_equal(arrays["applied_scale"], expected_scale.astype(np.float32)):
        raise ValueError("Normalization applied_scale mismatch")
    if np.any(arrays["minimum"] > arrays["maximum"]):
        raise ValueError("Normalization minimum exceeds maximum")


def validate_normalization_artifacts(
    npz_path: Path,
    metadata_path: Path,
    *,
    expected_stage2_contract_sha256: str,
    expected_training_index_sha256: str,
    expected_train_sample_id_sha256: str,
    expected_fold: object,
) -> dict[str, np.ndarray]:
    metadata = _strict_json(Path(metadata_path))
    required = {
        "contract",
        "provenance",
        "normalization_contract_sha256",
        "normalization_file_sha256",
        "near_constant_features",
    }
    if set(metadata) != required or not isinstance(metadata["contract"], dict):
        raise ValueError("Normalization metadata keys do not match contract")
    contract = metadata["contract"]
    contract_hash = _sha256_bytes(canonical_json_bytes(contract))
    if metadata["normalization_contract_sha256"] != contract_hash:
        raise ValueError("normalization_contract_sha256 mismatch")
    expected_bindings = {
        "stage2_contract_sha256": expected_stage2_contract_sha256.lower(),
        "training_index_sha256": expected_training_index_sha256.lower(),
        "train_sample_id_sha256": expected_train_sample_id_sha256.lower(),
        "fold": expected_fold,
        "sensor_order": list(SENSOR_ORDER),
        "feature_order": list(FEATURE_ORDER),
    }
    for key, expected in expected_bindings.items():
        if contract.get(key) != expected:
            raise ValueError(f"Normalization contract {key} mismatch")
    if metadata["normalization_file_sha256"] != sha256_file(Path(npz_path)):
        raise ValueError("normalization_file_sha256 mismatch")
    arrays = _load_npz(Path(npz_path))
    _validate_arrays(arrays)
    return arrays


def generate_normalization_artifacts(
    training_index_path: Path,
    training_index_metadata_path: Path,
    stage2_root: Path,
    stage2_schema_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    training_index = pd.read_csv(
        training_index_path, encoding="utf-8-sig", keep_default_na=False
    )
    metadata = _strict_json(Path(training_index_metadata_path))
    schema = load_stage2_schema(Path(stage2_schema_path))
    if metadata.get("training_index_sha256") != hash_training_index(training_index):
        raise ValueError("training_index_sha256 mismatch")
    if metadata.get("stage2_contract_sha256") != schema["contract_sha256"]:
        raise ValueError("stage2_contract_sha256 mismatch")
    selected_train = training_index[
        training_index["selected_for_run"].map(_as_bool)
        & (training_index["split"] == "train")
    ]
    train_sample_hash = _hash_sample_ids(selected_train["sample_id"])
    if metadata.get("train_sample_id_sha256") != train_sample_hash:
        raise ValueError("train_sample_id_sha256 mismatch")
    statistics = compute_normalization(training_index, stage2_root)
    return write_normalization_artifacts(
        statistics,
        output_dir,
        stage2_contract_sha256=str(schema["contract_sha256"]),
        training_index_sha256=str(metadata["training_index_sha256"]),
        train_sample_id_sha256=train_sample_hash,
        fold=metadata.get("fold"),
        train_users=selected_train["user_id"].astype(str).unique().tolist(),
        source_stage2_manifest_sha256=str(metadata["source_stage2_manifest_sha256"]),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute fold-only IMU normalization")
    parser.add_argument("--training-index", type=Path, required=True)
    parser.add_argument("--training-index-metadata", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--stage2-schema", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    generate_normalization_artifacts(
        args.training_index,
        args.training_index_metadata,
        args.stage2_root,
        args.stage2_schema,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
