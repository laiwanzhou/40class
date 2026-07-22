from __future__ import annotations

import hashlib
import json
import random
import zipfile
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler

from scripts.build_imu_training_index import hash_training_index
from scripts.compute_imu_normalization import validate_normalization_artifacts
from src.data.imu_stage2_contracts import (
    DataStatus,
    SequenceLengthSafetyError,
    sha256_file,
)
from src.data.imu_stage2_io import load_and_validate_npz, load_stage2_schema


def _load_metadata(value: Mapping[str, object] | Path) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        payload: dict[str, object] = {}
        for key, item in pairs:
            if key in payload:
                raise ValueError(f"Duplicate JSON key: {key}")
            payload[key] = item
        return payload

    with Path(value).open("r", encoding="utf-8") as handle:
        payload = json.load(handle, object_pairs_hook=reject_duplicates)
    if not isinstance(payload, dict):
        raise ValueError("Training index metadata must be an object")
    return payload


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError("Boolean training-index value is invalid")


def _read_npy_header(handle: object) -> tuple[tuple[int, ...], np.dtype]:
    version = np.lib.format.read_magic(handle)
    if version == (1, 0):
        shape, _, dtype = np.lib.format.read_array_header_1_0(handle)
    elif version in {(2, 0), (3, 0)}:
        shape, _, dtype = np.lib.format.read_array_header_2_0(handle)
    else:
        raise ValueError("Unsupported NPY format version")
    return tuple(int(dimension) for dimension in shape), np.dtype(dtype)


def _validate_npz_headers(
    path: Path,
    *,
    sample_id: str,
    hard_safety_limit_t: int,
) -> int:
    with zipfile.ZipFile(path) as archive:
        expected_members = {
            "values.npy",
            "sensor_mask.npy",
            "valid_mask.npy",
            "timestamps_ms.npy",
        }
        members = archive.infolist()
        if len(members) != 4 or {member.filename for member in members} != expected_members:
            raise ValueError("Stage 2 NPZ members do not match contract")
        if any(member.compress_type != zipfile.ZIP_STORED for member in members):
            raise ValueError("Stage 2 NPZ must be uncompressed")
        headers: dict[str, tuple[tuple[int, ...], np.dtype]] = {}
        for member_name in expected_members:
            with archive.open(member_name) as handle:
                headers[member_name] = _read_npy_header(handle)
    values_shape, values_dtype = headers["values.npy"]
    if len(values_shape) != 3 or values_shape[1:] != (5, 16) or values_dtype != np.float32:
        raise ValueError("Persisted Stage 2 values header is invalid")
    length = values_shape[0]
    if length < 1:
        raise ValueError("Persisted Stage 2 length must be positive")
    if length > hard_safety_limit_t:
        raise SequenceLengthSafetyError(
            sample_id,
            f"Persisted length {length} exceeds {hard_safety_limit_t}",
        )
    expected_headers = {
        "sensor_mask.npy": ((5,), np.dtype(bool)),
        "valid_mask.npy": ((length, 5), np.dtype(bool)),
        "timestamps_ms.npy": ((length,), np.dtype(np.int64)),
    }
    for member_name, expected in expected_headers.items():
        if headers[member_name] != expected:
            raise ValueError(f"Persisted Stage 2 {member_name[:-4]} header is invalid")
    return length


class IMUStage2Dataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        training_index: pd.DataFrame | Path,
        *,
        stage2_root: Path,
        stage2_schema: Path,
        normalization_npz: Path,
        normalization_json: Path,
        training_index_metadata: Mapping[str, object] | Path,
        hard_safety_limit_t: int,
        split: str | None = None,
    ) -> None:
        self.stage2_root = Path(stage2_root).resolve(strict=True)
        stage2_schema = Path(stage2_schema).resolve(strict=True)
        expected_schema = (self.stage2_root / "schema.json").resolve(strict=True)
        if stage2_schema != expected_schema:
            raise ValueError("stage2_schema must be stage2_root/schema.json")
        schema = load_stage2_schema(stage2_schema)
        contract_limit = int(schema["contract"]["hard_safety_limit_t"])
        if hard_safety_limit_t != contract_limit:
            raise ValueError("hard_safety_limit_t does not match Stage 2 contract")
        self.hard_safety_limit_t = hard_safety_limit_t
        metadata = _load_metadata(training_index_metadata)
        manifest_path = (self.stage2_root / "manifest.csv").resolve(strict=True)
        if metadata.get("source_stage2_manifest_path") != "manifest.csv":
            raise ValueError("source_stage2_manifest_path mismatch")
        actual_manifest_sha256 = sha256_file(manifest_path)
        if metadata.get("source_stage2_manifest_sha256") != actual_manifest_sha256:
            raise ValueError("source_stage2_manifest_sha256 mismatch")
        if isinstance(training_index, pd.DataFrame):
            frame = training_index.copy()
        else:
            frame = pd.read_csv(
                training_index, encoding="utf-8-sig", keep_default_na=False
            )
        required = {
            "sample_id",
            "stage2_npz_relpath",
            "status",
            "selected_for_run",
            "split",
        }
        if not required.issubset(frame.columns):
            raise ValueError("Training index is missing dataset columns")
        actual_index_sha256 = hash_training_index(frame)
        if metadata.get("training_index_sha256") != actual_index_sha256:
            raise ValueError("training_index_sha256 mismatch")
        selected_train = frame[
            frame["selected_for_run"].map(_as_bool)
            & (frame["split"].astype(str) == "train")
        ]
        train_sample_payload = "".join(
            f"{sample_id}\n" for sample_id in sorted(selected_train["sample_id"].astype(str))
        ).encode("utf-8")
        actual_train_sample_sha256 = hashlib.sha256(train_sample_payload).hexdigest()
        if metadata.get("train_sample_id_sha256") != actual_train_sample_sha256:
            raise ValueError("train_sample_id_sha256 mismatch")
        if "user_id" not in frame.columns:
            raise ValueError("Training index is missing user_id")
        self.normalization = validate_normalization_artifacts(
            Path(normalization_npz),
            Path(normalization_json),
            expected_stage2_contract_sha256=str(schema["contract_sha256"]),
            expected_training_index_sha256=actual_index_sha256,
            expected_train_sample_id_sha256=actual_train_sample_sha256,
            expected_fold=metadata.get("fold"),
            expected_train_users=selected_train["user_id"].astype(str).unique().tolist(),
            expected_source_stage2_manifest_sha256=actual_manifest_sha256,
        )
        frame = frame[frame["selected_for_run"].map(_as_bool)]
        if split is not None:
            if split not in {"train", "validation"}:
                raise ValueError("split must be train or validation")
            frame = frame[frame["split"].astype(str) == split]
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("Dataset sample_id values must be unique")
        self._rows = frame.to_dict(orient="records")
        self._length_cache: dict[int, tuple[tuple[int, int, int, int], int]] = {}

    def __len__(self) -> int:
        return len(self._rows)

    def _artifact_path(self, row: Mapping[str, object]) -> Path:
        relative = Path(str(row["stage2_npz_relpath"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Stage 2 dataset path is unsafe")
        path = (self.stage2_root / relative).resolve(strict=True)
        try:
            path.relative_to(self.stage2_root)
        except ValueError as error:
            raise ValueError("Stage 2 dataset path escapes root") from error
        return path

    def sequence_length(self, index: int) -> int:
        row = self._rows[index]
        path = self._artifact_path(row)
        signature = self._artifact_signature(path)
        cached = self._length_cache.get(index)
        if cached is not None and cached[0] == signature:
            return cached[1]
        length = _validate_npz_headers(
            path,
            sample_id=str(row["sample_id"]),
            hard_safety_limit_t=self.hard_safety_limit_t,
        )
        if self._artifact_signature(path) != signature:
            raise ValueError("Stage 2 NPZ changed during header validation")
        self._length_cache[index] = (signature, length)
        return length

    @staticmethod
    def _artifact_signature(path: Path) -> tuple[int, int, int, int]:
        stat = path.stat()
        return (
            int(stat.st_dev),
            int(stat.st_ino),
            int(stat.st_size),
            int(stat.st_mtime_ns),
        )

    @property
    def lengths(self) -> list[int]:
        return [self.sequence_length(index) for index in range(len(self))]

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self._rows[index]
        expected_length = self.sequence_length(index)
        result = load_and_validate_npz(
            self._artifact_path(row),
            sample_id=str(row["sample_id"]),
            status=DataStatus(str(row["status"])),
            qc={},
        )
        if len(result.values) != expected_length:
            raise ValueError("Stage 2 NPZ length changed during load")
        valid = result.valid_mask
        standardized = np.zeros_like(result.values, dtype=np.float32)
        centered = (
            result.values.astype(np.float64)
            - self.normalization["mean"].astype(np.float64)[None, :, :]
        ) / self.normalization["applied_scale"].astype(np.float64)[None, :, :]
        standardized[valid] = centered[valid].astype(np.float32)
        if not np.isfinite(standardized).all() or not np.all(standardized[~valid] == 0):
            raise ValueError("Standardized Stage 2 values violate finite-zero contract")
        sample: dict[str, object] = {
            "values": torch.from_numpy(standardized),
            "valid_mask": torch.from_numpy(valid.copy()),
            "sensor_mask": torch.from_numpy(result.sensor_mask.copy()),
            "usable_sensor_mask": torch.from_numpy(result.usable_sensor_mask.copy()),
            "timestamps_ms": torch.from_numpy(result.timestamps_ms.copy()),
            "length": expected_length,
            "sample_id": str(row["sample_id"]),
            "imu_modality_mask": True,
        }
        if "label_index" in row and str(row["label_index"]) != "":
            sample["label"] = int(row["label_index"])
        return sample


def collate_imu_stage2(samples: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if not samples:
        raise ValueError("Cannot collate an empty IMU batch")
    lengths = torch.as_tensor([int(sample["length"]) for sample in samples], dtype=torch.int64)
    if torch.any(lengths < 1):
        raise ValueError("Training Stage 2 samples must have positive length")
    batch_size = len(samples)
    batch_t = int(lengths.max().item())
    values = torch.zeros((batch_size, batch_t, 5, 16), dtype=torch.float32)
    valid_mask = torch.zeros((batch_size, batch_t, 5), dtype=torch.bool)
    sequence_mask = torch.zeros((batch_size, batch_t), dtype=torch.bool)
    timestamps_ms = torch.full((batch_size, batch_t), -1, dtype=torch.int64)
    sensor_mask = torch.zeros((batch_size, 5), dtype=torch.bool)
    usable_sensor_mask = torch.zeros((batch_size, 5), dtype=torch.bool)
    modality_mask = torch.zeros(batch_size, dtype=torch.bool)
    sample_ids: list[str] = []
    labels: list[int] = []
    has_labels = all("label" in sample for sample in samples)
    if any("label" in sample for sample in samples) and not has_labels:
        raise ValueError("Either all or no collated samples must have labels")
    for batch_index, sample in enumerate(samples):
        length = int(sample["length"])
        sample_values = torch.as_tensor(sample["values"], dtype=torch.float32)
        sample_valid = torch.as_tensor(sample["valid_mask"], dtype=torch.bool)
        if sample_values.shape != (length, 5, 16) or sample_valid.shape != (length, 5):
            raise ValueError("Sample tensor shape does not match length")
        values[batch_index, :length] = sample_values
        valid_mask[batch_index, :length] = sample_valid
        sequence_mask[batch_index, :length] = True
        timestamps_ms[batch_index, :length] = torch.as_tensor(
            sample["timestamps_ms"], dtype=torch.int64
        )
        sensor_mask[batch_index] = torch.as_tensor(sample["sensor_mask"], dtype=torch.bool)
        usable_sensor_mask[batch_index] = torch.as_tensor(
            sample["usable_sensor_mask"], dtype=torch.bool
        )
        modality_mask[batch_index] = bool(sample.get("imu_modality_mask", True))
        sample_ids.append(str(sample["sample_id"]))
        if has_labels:
            labels.append(int(sample["label"]))
    if not torch.equal(usable_sensor_mask, valid_mask.any(dim=1)):
        raise ValueError("usable_sensor_mask disagrees with valid_mask")
    batch: dict[str, object] = {
        "values": values,
        "valid_mask": valid_mask,
        "sequence_mask": sequence_mask,
        "sensor_mask": sensor_mask,
        "usable_sensor_mask": usable_sensor_mask,
        "timestamps_ms": timestamps_ms,
        "lengths": lengths,
        "sample_id": sample_ids,
        "imu_modality_mask": modality_mask,
    }
    if has_labels:
        batch["labels"] = torch.as_tensor(labels, dtype=torch.int64)
    return batch


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        lengths: Sequence[int],
        bucket_boundaries: Sequence[int],
        batch_feature_budget: int,
        maximum_batch_size: int,
        minimum_batch_size: int,
        shuffle_seed: int,
        drop_last: bool,
    ) -> None:
        self.lengths = [int(length) for length in lengths]
        self.bucket_boundaries = [int(boundary) for boundary in bucket_boundaries]
        if any(length < 1 for length in self.lengths):
            raise ValueError("Sampler lengths must be positive")
        if self.bucket_boundaries != sorted(set(self.bucket_boundaries)):
            raise ValueError("bucket_boundaries must be unique and increasing")
        if batch_feature_budget <= 0 or maximum_batch_size < 1 or minimum_batch_size < 1:
            raise ValueError("Sampler budgets and batch sizes must be positive")
        if minimum_batch_size > maximum_batch_size:
            raise ValueError("minimum_batch_size exceeds maximum_batch_size")
        self.batch_feature_budget = int(batch_feature_budget)
        self.maximum_batch_size = int(maximum_batch_size)
        self.minimum_batch_size = int(minimum_batch_size)
        self.shuffle_seed = int(shuffle_seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0
        self.omitted_indices: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _bucket(self, length: int) -> int:
        for index, boundary in enumerate(self.bucket_boundaries):
            if length <= boundary:
                return index
        return len(self.bucket_boundaries)

    def _build_batches(self) -> list[list[int]]:
        rng = random.Random(self.shuffle_seed + self.epoch)
        buckets: dict[int, list[int]] = defaultdict(list)
        for index, length in enumerate(self.lengths):
            buckets[self._bucket(length)].append(index)
        batches: list[list[int]] = []
        omitted: list[int] = []
        for bucket_id in sorted(buckets):
            indices = buckets[bucket_id]
            rng.shuffle(indices)
            current: list[int] = []
            current_max = 0
            for index in indices:
                candidate_max = max(current_max, self.lengths[index])
                candidate_size = len(current) + 1
                candidate_cost = candidate_size * candidate_max * 5 * 16
                if current and (
                    candidate_size > self.maximum_batch_size
                    or candidate_cost > self.batch_feature_budget
                ):
                    if not self.drop_last or len(current) >= self.minimum_batch_size:
                        batches.append(current)
                    else:
                        omitted.extend(current)
                    current = []
                    current_max = 0
                current.append(index)
                current_max = max(current_max, self.lengths[index])
                if len(current) == 1 and current_max * 5 * 16 > self.batch_feature_budget:
                    batches.append(current)
                    current = []
                    current_max = 0
            if current:
                if not self.drop_last or len(current) >= self.minimum_batch_size:
                    batches.append(current)
                else:
                    omitted.extend(current)
        rng.shuffle(batches)
        self.omitted_indices = sorted(omitted)
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._build_batches()

    def __len__(self) -> int:
        return len(self._build_batches())
