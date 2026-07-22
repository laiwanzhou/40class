from __future__ import annotations

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

from scripts.compute_imu_normalization import validate_normalization_artifacts
from src.data.imu_stage2_contracts import DataStatus, SequenceLengthSafetyError
from src.data.imu_stage2_io import load_and_validate_npz, load_stage2_schema


def _load_metadata(value: Mapping[str, object] | Path) -> dict[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    with Path(value).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
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


def _values_shape_from_npz(path: Path) -> tuple[int, ...]:
    with zipfile.ZipFile(path) as archive:
        try:
            with archive.open("values.npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, _, _ = np.lib.format.read_array_header_1_0(handle)
                elif version in {(2, 0), (3, 0)}:
                    shape, _, _ = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise ValueError("Unsupported values.npy format version")
        except KeyError as error:
            raise ValueError("Stage 2 NPZ is missing values.npy") from error
    return tuple(int(dimension) for dimension in shape)


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
        schema = load_stage2_schema(Path(stage2_schema))
        contract_limit = int(schema["contract"]["hard_safety_limit_t"])
        if hard_safety_limit_t != contract_limit:
            raise ValueError("hard_safety_limit_t does not match Stage 2 contract")
        self.hard_safety_limit_t = hard_safety_limit_t
        self.stage2_root = Path(stage2_root).resolve(strict=True)
        metadata = _load_metadata(training_index_metadata)
        self.normalization = validate_normalization_artifacts(
            Path(normalization_npz),
            Path(normalization_json),
            expected_stage2_contract_sha256=str(schema["contract_sha256"]),
            expected_training_index_sha256=str(metadata["training_index_sha256"]),
            expected_train_sample_id_sha256=str(metadata["train_sample_id_sha256"]),
            expected_fold=metadata.get("fold"),
        )
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
        frame = frame[frame["selected_for_run"].map(_as_bool)]
        if split is not None:
            if split not in {"train", "validation"}:
                raise ValueError("split must be train or validation")
            frame = frame[frame["split"].astype(str) == split]
        if frame["sample_id"].astype(str).duplicated().any():
            raise ValueError("Dataset sample_id values must be unique")
        self._rows = frame.to_dict(orient="records")

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
        shape = _values_shape_from_npz(self._artifact_path(row))
        if len(shape) != 3 or shape[1:] != (5, 16):
            raise ValueError("Persisted Stage 2 values shape is invalid")
        length = shape[0]
        if length > self.hard_safety_limit_t:
            raise SequenceLengthSafetyError(
                str(row["sample_id"]),
                f"Persisted length {length} exceeds {self.hard_safety_limit_t}",
            )
        return length

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
