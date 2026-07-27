from __future__ import annotations

import hashlib
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.build_imu_training_index import load_class_order
from src.data.imu_stage2_contracts import sha256_file
from src.data.imu_stage2_dataset import (
    IMUStage2Dataset,
    LengthBucketBatchSampler,
    collate_imu_stage2,
)
from src.data.imu_stage2_io import load_stage2_schema
from src.models.imu_stage2_tcn import (
    build_imu_stage2_model,
    build_training_checkpoint_metadata,
)


TRAINING_CONFIG_VERSION = "imu-stage2-training-v1"
EXPECTED_CONFIG: dict[str, object] = {
    "config_version": TRAINING_CONFIG_VERSION,
    "fold": 0,
    "seed": 20260724,
    "maximum_epochs": 40,
    "early_stopping_patience": 8,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "label_smoothing": 0.05,
    "gradient_clip_norm": 1.0,
    "optimizer": "AdamW",
    "scheduler": "cosine",
    "dtype": "float32",
    "num_workers": 0,
    "embedding_dim": 128,
    "tcn_channels": [64, 128],
    "dropout": 0.2,
    "imu_modality_dropout": 0.0,
    "num_classes": 40,
    "hard_safety_limit_t": 10_000,
    "bucket_boundaries": [24, 48, 64, 96, 128, 192, 256],
    "batch_feature_budget": 327_680,
    "maximum_batch_size": 16,
    "minimum_batch_size": 1,
    "drop_last": False,
}


def _strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON value is forbidden: {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(
            handle,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    if not isinstance(payload, dict):
        raise ValueError("JSON payload must be an object")
    return payload


def load_training_config(path: Path) -> dict[str, object]:
    payload = _strict_json(Path(path))
    if set(payload) != set(EXPECTED_CONFIG):
        unknown = sorted(set(payload) - set(EXPECTED_CONFIG))
        missing = sorted(set(EXPECTED_CONFIG) - set(payload))
        name = unknown[0] if unknown else missing[0]
        raise ValueError(f"Training config field set mismatch: {name}")
    for name, expected in EXPECTED_CONFIG.items():
        actual = payload[name]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"Training config {name} must equal {expected!r}")
    return payload


def validate_training_artifact_bindings(
    training_metadata: Mapping[str, object],
    *,
    stage2_contract_sha256: str,
    class_order_sha256: str,
    num_classes: int,
    expected_fold: int,
    normalization_fold: object,
    split_definition_path: str,
) -> None:
    expected = {
        "stage2_contract_sha256": stage2_contract_sha256,
        "class_order_sha256": class_order_sha256,
        "num_classes": num_classes,
        "fold": expected_fold,
    }
    for field, value in expected.items():
        if training_metadata.get(field) != value:
            raise ValueError(f"Training artifact {field} mismatch")
    if normalization_fold != expected_fold:
        raise ValueError("normalization contract fold mismatch")
    expected_split = f"metadata/splits/fold_{expected_fold}.json"
    if split_definition_path != expected_split:
        raise ValueError("Training artifact split_definition_path mismatch")


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_metrics(
    *,
    labels: np.ndarray,
    predictions: np.ndarray,
    num_classes: int,
) -> dict[str, object]:
    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    if (
        labels.ndim != 1
        or predictions.shape != labels.shape
        or labels.dtype.kind not in "iu"
        or predictions.dtype.kind not in "iu"
        or isinstance(num_classes, bool)
        or not isinstance(num_classes, int)
        or num_classes < 1
    ):
        raise ValueError("Metric inputs are invalid")
    if labels.size and (
        labels.min() < 0
        or predictions.min() < 0
        or labels.max() >= num_classes
        or predictions.max() >= num_classes
    ):
        raise ValueError("Metric label is outside class range")
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (labels, predictions), 1)
    per_class: list[dict[str, object]] = []
    precision_values: list[float] = []
    recall_values: list[float] = []
    f1_values: list[float] = []
    for index in range(num_classes):
        true_positive = int(confusion[index, index])
        predicted = int(confusion[:, index].sum())
        support = int(confusion[index].sum())
        precision = true_positive / predicted if predicted else 0.0
        recall = true_positive / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    accuracy = float(np.trace(confusion) / labels.size) if labels.size else 0.0
    return {
        "accuracy": accuracy,
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def _move_batch(batch: Mapping[str, object], device: torch.device) -> dict[str, object]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def training_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )


def _assert_finite_tensor(value: torch.Tensor, name: str) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name} detected")


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    label_smoothing: float,
    gradient_clip_norm: float,
    fail_fast_first_batch: bool,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    total_correct = 0
    maximum_gradient_norm = 0.0
    batches = 0
    for batch_index, original_batch in enumerate(loader):
        batch = _move_batch(original_batch, device)
        values = torch.as_tensor(batch["values"])
        if not torch.isfinite(values).all():
            raise FloatingPointError("Non-finite input detected")
        labels = torch.as_tensor(batch["labels"], dtype=torch.int64, device=device)
        optimizer.zero_grad(set_to_none=True)
        result = model(batch)
        logits = result["logits"]
        _assert_finite_tensor(logits, "logits")
        loss = training_loss(logits, labels, label_smoothing=label_smoothing)
        _assert_finite_tensor(loss, "loss")
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        if not gradients or any(not torch.isfinite(gradient).all() for gradient in gradients):
            raise FloatingPointError("Non-finite gradient detected")
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=gradient_clip_norm,
            error_if_nonfinite=True,
        )
        if not torch.isfinite(torch.as_tensor(gradient_norm)):
            raise FloatingPointError("Non-finite gradient norm detected")
        if fail_fast_first_batch and batch_index == 0:
            _assert_finite_tensor(logits, "logits")
            _assert_finite_tensor(loss, "loss")
        optimizer.step()
        count = int(labels.numel())
        total_loss += float(loss.detach().cpu()) * count
        total_samples += count
        total_correct += int((torch.argmax(logits.detach(), dim=1) == labels).sum().item())
        maximum_gradient_norm = max(
            maximum_gradient_norm,
            float(torch.as_tensor(gradient_norm).detach().cpu()),
        )
        batches += 1
    if batches == 0 or total_samples == 0:
        raise ValueError("Training loader produced no samples")
    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
        "gradient_norm": maximum_gradient_norm,
        "batches": float(batches),
        "samples": float(total_samples),
    }


def evaluate(
    model: nn.Module,
    loader: Iterable[Mapping[str, object]],
    *,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    was_training = model.training
    model.eval()
    logits_parts: list[np.ndarray] = []
    embedding_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    sample_ids: list[str] = []
    total_loss = 0.0
    total_samples = 0
    with torch.inference_mode():
        for original_batch in loader:
            batch = _move_batch(original_batch, device)
            labels = torch.as_tensor(batch["labels"], dtype=torch.int64, device=device)
            result = model(batch)
            logits = result["logits"]
            embeddings = result["embedding"]
            _assert_finite_tensor(logits, "validation logits")
            _assert_finite_tensor(embeddings, "validation embedding")
            loss = torch.nn.functional.cross_entropy(logits, labels)
            _assert_finite_tensor(loss, "validation loss")
            count = int(labels.numel())
            total_loss += float(loss.cpu()) * count
            total_samples += count
            logits_parts.append(logits.cpu().to(torch.float32).numpy())
            embedding_parts.append(embeddings.cpu().to(torch.float32).numpy())
            label_parts.append(labels.cpu().numpy().astype(np.int64, copy=False))
            sample_ids.extend(map(str, batch["sample_id"]))
    model.train(was_training)
    if total_samples == 0:
        raise ValueError("Validation loader produced no samples")
    logits_array = np.concatenate(logits_parts, axis=0)
    embeddings_array = np.concatenate(embedding_parts, axis=0)
    labels_array = np.concatenate(label_parts, axis=0)
    predictions = np.argmax(logits_array, axis=1).astype(np.int64)
    metrics = classification_metrics(
        labels=labels_array,
        predictions=predictions,
        num_classes=num_classes,
    )
    return {
        **metrics,
        "loss": total_loss / total_samples,
        "logits": logits_array,
        "embeddings": embeddings_array,
        "labels": labels_array,
        "sample_ids": sample_ids,
    }


def is_better_validation(
    candidate: Mapping[str, object],
    incumbent: Mapping[str, object] | None,
) -> bool:
    if incumbent is None:
        return True
    candidate_key = (
        float(candidate["macro_f1"]),
        float(candidate["accuracy"]),
        -float(candidate["loss"]),
        -int(candidate["epoch"]),
    )
    incumbent_key = (
        float(incumbent["macro_f1"]),
        float(incumbent["accuracy"]),
        -float(incumbent["loss"]),
        -int(incumbent["epoch"]),
    )
    return candidate_key > incumbent_key


@dataclass
class EarlyStopping:
    patience: int
    best: dict[str, object] | None = None
    stale_epochs: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.patience, bool) or self.patience < 1:
            raise ValueError("patience must be a positive integer")

    def observe(self, metrics: Mapping[str, object]) -> bool:
        if is_better_validation(metrics, self.best):
            self.best = dict(metrics)
            self.stale_epochs = 0
            return False
        self.stale_epochs += 1
        return self.stale_epochs >= self.patience


def _validate_training_metadata(metadata: Mapping[str, object]) -> dict[str, object]:
    expected_keys = {
        "checkpoint_metadata_version",
        "stage2_contract_sha256",
        "training_index_sha256",
        "normalization_contract_sha256",
        "normalization_file_sha256",
        "class_order_sha256",
        "num_classes",
    }
    if set(metadata) != expected_keys:
        raise ValueError("Training checkpoint metadata keys mismatch")
    if metadata.get("checkpoint_metadata_version") != "imu-training-checkpoint-v1":
        raise ValueError("Training checkpoint metadata version mismatch")
    return build_training_checkpoint_metadata(
        stage2_contract_sha256=metadata["stage2_contract_sha256"],  # type: ignore[arg-type]
        training_index_sha256=metadata["training_index_sha256"],  # type: ignore[arg-type]
        normalization_contract_sha256=metadata["normalization_contract_sha256"],  # type: ignore[arg-type]
        normalization_file_sha256=metadata["normalization_file_sha256"],  # type: ignore[arg-type]
        class_order_sha256=metadata["class_order_sha256"],  # type: ignore[arg-type]
        num_classes=metadata["num_classes"],  # type: ignore[arg-type]
    )


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    metrics: Mapping[str, object],
    metadata: Mapping[str, object],
    config: Mapping[str, object],
    class_order: Sequence[Mapping[str, object]] | None = None,
    provenance: Mapping[str, object] | None = None,
    include_optimizer: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_metadata = _validate_training_metadata(metadata)
    payload = {
        "checkpoint_version": "imu-stage2-training-state-v1",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "metrics": dict(metrics),
        "checkpoint_metadata": normalized_metadata,
        "training_config": dict(config),
    }
    if class_order is not None:
        payload["class_order"] = [dict(record) for record in class_order]
    if provenance is not None:
        payload["provenance"] = dict(provenance)
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()
        payload["scheduler_state_dict"] = scheduler.state_dict()
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}"
    try:
        torch.save(payload, temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    expected_metadata: Mapping[str, object],
) -> dict[str, object]:
    expected = _validate_training_metadata(expected_metadata)
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("checkpoint_version") != "imu-stage2-training-state-v1":
        raise ValueError("Checkpoint payload is incompatible")
    if payload.get("checkpoint_metadata") != expected:
        raise ValueError("Checkpoint metadata mismatch")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint model state is invalid")
    model.load_state_dict(state, strict=True)
    return payload


def _normalize_class_order(
    class_order: Sequence[Mapping[str, object]],
    *,
    num_classes: int,
) -> list[dict[str, object]]:
    if len(class_order) != num_classes:
        raise ValueError("class_order length does not match logits")
    normalized: list[dict[str, object]] = []
    class_ids: set[int] = set()
    class_names: set[str] = set()
    for label_index, record in enumerate(class_order):
        if set(record) != {"class_id", "class_name", "label_index"}:
            raise ValueError("class_order record keys are invalid")
        class_id = int(record["class_id"])
        class_name = str(record["class_name"])
        if int(record["label_index"]) != label_index or not class_name:
            raise ValueError("class_order labels must be consecutive and named")
        if class_id in class_ids or class_name in class_names:
            raise ValueError("class_order identities must be unique")
        class_ids.add(class_id)
        class_names.add(class_name)
        normalized.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "label_index": label_index,
            }
        )
    return normalized


def _class_order_array(records: Sequence[Mapping[str, object]]) -> np.ndarray:
    width = max(len(str(record["class_name"])) for record in records)
    dtype = np.dtype(
        [
            ("label_index", np.int64),
            ("class_id", np.int64),
            ("class_name", f"U{width}"),
        ]
    )
    result = np.empty(len(records), dtype=dtype)
    for index, record in enumerate(records):
        result[index] = (
            int(record["label_index"]),
            int(record["class_id"]),
            str(record["class_name"]),
        )
    return result


def write_validation_outputs(
    path: Path,
    *,
    sample_ids: Sequence[str],
    labels: np.ndarray,
    logits: np.ndarray,
    embeddings: np.ndarray,
    class_order: Sequence[Mapping[str, object]],
) -> None:
    path = Path(path)
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    embeddings = np.asarray(embeddings)
    count = len(sample_ids)
    if labels.dtype != np.int64 or labels.shape != (count,):
        raise ValueError("Validation labels are invalid")
    if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[0] != count:
        raise ValueError("Validation logits are invalid")
    if embeddings.dtype != np.float32 or embeddings.shape != (count, 128):
        raise ValueError("Validation embeddings must have shape [N,128]")
    if not np.isfinite(logits).all() or not np.isfinite(embeddings).all():
        raise ValueError("Validation outputs must be finite")
    normalized_order = _normalize_class_order(
        class_order,
        num_classes=int(logits.shape[1]),
    )
    sample_array = np.asarray(list(map(str, sample_ids)), dtype=np.str_)
    if any(not sample_id for sample_id in sample_array.tolist()):
        raise ValueError("Validation sample_id values must be non-empty")
    if len(set(sample_array.tolist())) != count:
        raise ValueError("Validation sample_id values must be unique")
    order = np.argsort(sample_array, kind="stable")
    sample_array = sample_array[order]
    labels = labels[order]
    logits = logits[order]
    embeddings = embeddings[order]
    predictions = np.argmax(logits, axis=1).astype(np.int64)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}.npz"
    try:
        np.savez(
            temporary,
            sample_ids=sample_array,
            labels=labels,
            predictions=predictions,
            logits=logits,
            embeddings=embeddings,
            class_order=_class_order_array(normalized_order),
        )
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        validate_validation_outputs(path, expected_class_order=normalized_order)
    finally:
        if temporary.exists():
            temporary.unlink()


def validate_validation_outputs(
    path: Path,
    *,
    expected_class_order: Sequence[Mapping[str, object]],
) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        expected_keys = {
            "sample_ids",
            "labels",
            "predictions",
            "logits",
            "embeddings",
            "class_order",
        }
        if set(archive.files) != expected_keys:
            raise ValueError("Validation output NPZ keys mismatch")
        arrays = {key: archive[key].copy() for key in expected_keys}
    sample_ids = arrays["sample_ids"]
    labels = arrays["labels"]
    predictions = arrays["predictions"]
    logits = arrays["logits"]
    embeddings = arrays["embeddings"]
    class_order_array = arrays["class_order"]
    count = len(sample_ids)
    if sample_ids.ndim != 1 or sample_ids.dtype.kind != "U":
        raise ValueError("Validation sample_ids dtype or shape mismatch")
    sample_list = sample_ids.tolist()
    if any(not value for value in sample_list) or len(set(sample_list)) != count:
        raise ValueError("Validation sample_id values must be non-empty and unique")
    if sample_list != sorted(sample_list):
        raise ValueError("Validation sample_id values must be stably sorted")
    if labels.dtype != np.int64 or labels.shape != (count,):
        raise ValueError("Validation labels dtype or shape mismatch")
    if predictions.dtype != np.int64 or predictions.shape != (count,):
        raise ValueError("Validation predictions dtype or shape mismatch")
    num_classes = len(expected_class_order)
    if logits.dtype != np.float32 or logits.shape != (count, num_classes):
        raise ValueError("Validation logits dtype or shape mismatch")
    if embeddings.dtype != np.float32 or embeddings.shape != (count, 128):
        raise ValueError("Validation embeddings dtype or shape mismatch")
    if not np.isfinite(logits).all() or not np.isfinite(embeddings).all():
        raise ValueError("Validation outputs must be finite")
    if not np.array_equal(predictions, np.argmax(logits, axis=1).astype(np.int64)):
        raise ValueError("Validation predictions disagree with argmax(logits)")
    normalized_order = _normalize_class_order(
        expected_class_order,
        num_classes=num_classes,
    )
    if class_order_array.dtype.names != ("label_index", "class_id", "class_name"):
        raise ValueError("Validation class_order dtype mismatch")
    reopened_order = [
        {
            "label_index": int(record["label_index"]),
            "class_id": int(record["class_id"]),
            "class_name": str(record["class_name"]),
        }
        for record in class_order_array
    ]
    if reopened_order != normalized_order:
        raise ValueError("Validation class_order mismatch")
    return arrays


@contextmanager
def staged_output_directory(output_dir: Path) -> Iterator[Path]:
    output_dir = Path(output_dir)
    parent = output_dir.parent
    if output_dir.exists():
        raise FileExistsError("Training output already exists; overwrite and resume are forbidden")
    residues = list(parent.glob(f".{output_dir.name}.staging-*"))
    if residues:
        raise FileExistsError("Unknown training output staging residue exists")
    staging = parent / f".{output_dir.name}.staging-{uuid4().hex}"
    parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(exist_ok=False)
    try:
        yield staging
        os.replace(staging, output_dir)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Non-finite metric cannot be serialized")
    return value


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: Iterable[Mapping[str, object]],
) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fieldnames})
        handle.flush()
        os.fsync(handle.fileno())


def _write_run_manifest(output_dir: Path) -> None:
    files = []
    for path in sorted(Path(output_dir).iterdir(), key=lambda item: item.name):
        if path.name == "run_manifest.json" or not path.is_file():
            continue
        files.append(
            {
                "relative_path": path.name,
                "size": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    _write_json(
        Path(output_dir) / "run_manifest.json",
        {
            "manifest_version": "imu-stage2-training-run-manifest-v1",
            "files": files,
        },
    )


def _validate_run_manifest(output_dir: Path, expected_names: set[str]) -> None:
    output_dir = Path(output_dir)
    actual_names = {path.name for path in output_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise ValueError("Training run output file set mismatch")
    manifest = _strict_json(output_dir / "run_manifest.json")
    if set(manifest) != {"manifest_version", "files"} or manifest.get(
        "manifest_version"
    ) != "imu-stage2-training-run-manifest-v1":
        raise ValueError("Training run manifest contract mismatch")
    records = manifest.get("files")
    if not isinstance(records, list):
        raise ValueError("Training run manifest files must be a list")
    expected_manifest_names = expected_names - {"run_manifest.json"}
    names: list[str] = []
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "relative_path",
            "size",
            "sha256",
        }:
            raise ValueError("Training run manifest record is invalid")
        name = str(record["relative_path"])
        if Path(name).name != name or name == "run_manifest.json":
            raise ValueError("Training run manifest path is invalid")
        artifact = output_dir / name
        if not artifact.is_file():
            raise ValueError("Training run manifest artifact is missing")
        if record["size"] != artifact.stat().st_size:
            raise ValueError("Training run manifest size mismatch")
        if record["sha256"] != sha256_file(artifact):
            raise ValueError("Training run manifest SHA-256 mismatch")
        names.append(name)
    if names != sorted(expected_manifest_names):
        raise ValueError("Training run manifest file list mismatch")


def fit_model(
    *,
    model: nn.Module,
    train_loader: Iterable[Mapping[str, object]],
    validation_loader: Iterable[Mapping[str, object]],
    output_dir: Path,
    config: Mapping[str, object],
    metadata: Mapping[str, object],
    class_order: Sequence[Mapping[str, object]],
    provenance: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
    normalized_order = _normalize_class_order(
        class_order,
        num_classes=int(metadata["num_classes"]),
    )
    started_at = datetime.now(timezone.utc)
    model.to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    maximum_epochs = int(config["maximum_epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=maximum_epochs,
    )
    stopper = EarlyStopping(int(config["early_stopping_patience"]))
    history: list[dict[str, object]] = []
    best_epoch = 0
    with staged_output_directory(Path(output_dir)) as staging:
        for epoch in range(1, maximum_epochs + 1):
            epoch_started = time.perf_counter()
            batch_sampler = getattr(train_loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch - 1)
            train_metrics = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device=device,
                label_smoothing=float(config["label_smoothing"]),
                gradient_clip_norm=float(config["gradient_clip_norm"]),
                fail_fast_first_batch=epoch == 1,
            )
            validation = evaluate(
                model,
                validation_loader,
                device=device,
                num_classes=int(metadata["num_classes"]),
            )
            comparable = {
                "macro_f1": validation["macro_f1"],
                "accuracy": validation["accuracy"],
                "loss": validation["loss"],
                "epoch": epoch,
            }
            improved = is_better_validation(comparable, stopper.best)
            epoch_record = {
                "epoch": epoch,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "train": train_metrics,
                "validation": {
                    key: value
                    for key, value in validation.items()
                    if key not in {"logits", "embeddings", "labels", "sample_ids"}
                },
                "epoch_duration_seconds": time.perf_counter() - epoch_started,
                "is_best": improved,
            }
            history.append(epoch_record)
            should_stop = stopper.observe(comparable)
            scheduler.step()
            save_checkpoint(
                staging / "last_model.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=comparable,
                metadata=metadata,
                config=config,
                class_order=normalized_order,
                provenance=provenance,
            )
            if improved:
                best_epoch = epoch
                save_checkpoint(
                    staging / "best_model.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=comparable,
                    metadata=metadata,
                    config=config,
                    class_order=normalized_order,
                    provenance=provenance,
                    include_optimizer=False,
                )
            if should_stop:
                break
        reloaded_model = build_imu_stage2_model(
            config,
            num_classes=int(metadata["num_classes"]),
        ).to(device=device, dtype=torch.float32)
        best_payload = load_checkpoint(
            staging / "best_model.pt",
            model=reloaded_model,
            expected_metadata=metadata,
        )
        if best_payload.get("class_order") != normalized_order:
            raise ValueError("Best checkpoint class_order mismatch")
        if "optimizer_state_dict" in best_payload or "scheduler_state_dict" in best_payload:
            raise ValueError("Best inference checkpoint must not contain optimizer state")
        last_payload = torch.load(
            staging / "last_model.pt", map_location="cpu", weights_only=False
        )
        if "optimizer_state_dict" not in last_payload:
            raise ValueError("Last checkpoint must contain optimizer state")
        best_validation = evaluate(
            reloaded_model,
            validation_loader,
            device=device,
            num_classes=int(metadata["num_classes"]),
        )
        write_validation_outputs(
            staging / "validation_outputs.npz",
            sample_ids=best_validation["sample_ids"],
            labels=best_validation["labels"],
            logits=best_validation["logits"],
            embeddings=best_validation["embeddings"],
            class_order=normalized_order,
        )
        validation_arrays = validate_validation_outputs(
            staging / "validation_outputs.npz",
            expected_class_order=normalized_order,
        )
        predictions = validation_arrays["predictions"]
        labels = validation_arrays["labels"]
        sample_ids = validation_arrays["sample_ids"].tolist()
        _write_csv(
            staging / "validation_predictions.csv",
            fieldnames=(
                "sample_id",
                "true_label_index",
                "predicted_label_index",
                "correct",
            ),
            rows=(
                {
                    "sample_id": sample_id,
                    "true_label_index": int(label),
                    "predicted_label_index": int(prediction),
                    "correct": bool(label == prediction),
                }
                for sample_id, label, prediction in zip(
                    sample_ids, labels, predictions
                )
            ),
        )
        confusion = np.asarray(best_validation["confusion_matrix"], dtype=np.int64)
        confusion_fields = ["true_label_index", "true_class_name"] + [
            f"predicted_{index}" for index in range(len(normalized_order))
        ]
        _write_csv(
            staging / "confusion_matrix.csv",
            fieldnames=confusion_fields,
            rows=(
                {
                    "true_label_index": index,
                    "true_class_name": normalized_order[index]["class_name"],
                    **{
                        f"predicted_{prediction}": int(confusion[index, prediction])
                        for prediction in range(len(normalized_order))
                    },
                }
                for index in range(len(normalized_order))
            ),
        )
        _write_csv(
            staging / "per_class_metrics.csv",
            fieldnames=(
                "label_index",
                "class_id",
                "class_name",
                "precision",
                "recall",
                "f1",
                "support",
            ),
            rows=(
                {
                    **record,
                    **best_validation["per_class"][index],
                }
                for index, record in enumerate(normalized_order)
            ),
        )
        _write_csv(
            staging / "metrics.csv",
            fieldnames=(
                "epoch",
                "learning_rate",
                "train_loss",
                "train_accuracy",
                "validation_loss",
                "validation_accuracy",
                "validation_macro_precision",
                "validation_macro_recall",
                "validation_macro_f1",
                "gradient_norm",
                "epoch_duration_seconds",
                "is_best",
            ),
            rows=(
                {
                    "epoch": record["epoch"],
                    "learning_rate": record["learning_rate"],
                    "train_loss": record["train"]["loss"],
                    "train_accuracy": record["train"]["accuracy"],
                    "validation_loss": record["validation"]["loss"],
                    "validation_accuracy": record["validation"]["accuracy"],
                    "validation_macro_precision": record["validation"]["macro_precision"],
                    "validation_macro_recall": record["validation"]["macro_recall"],
                    "validation_macro_f1": record["validation"]["macro_f1"],
                    "gradient_norm": record["train"]["gradient_norm"],
                    "epoch_duration_seconds": record["epoch_duration_seconds"],
                    "is_best": record["is_best"],
                }
                for record in history
            ),
        )
        _write_json(staging / "resolved_config.json", dict(config))
        finished_at = datetime.now(timezone.utc)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        model_bytes = (staging / "best_model.pt").stat().st_size
        if model_bytes >= 95 * 1024 * 1024:
            raise ValueError("Best model checkpoint exceeds 95 MiB")
        best_metrics = {
            key: value
            for key, value in best_validation.items()
            if key not in {"logits", "embeddings", "labels", "sample_ids"}
        }
        _write_json(
            staging / "training_summary.json",
            {
                "training_summary_version": "imu-stage2-training-summary-v1",
                "best_epoch": best_epoch,
                "last_epoch": len(history),
                "stopped_early": len(history) < maximum_epochs,
                "best_metrics": best_metrics,
                "sample_counts": {
                    "train": int(history[-1]["train"]["samples"]),
                    "validation": len(sample_ids),
                },
                "class_count": len(normalized_order),
                "parameter_count": parameter_count,
                "model_byte_size": model_bytes,
                "device": str(device),
                "python_version": sys.version.split()[0],
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "gpu_name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else None
                ),
                "started_at_utc": started_at.isoformat(),
                "finished_at_utc": finished_at.isoformat(),
                "input_hashes": provenance.get("input_hashes", {}),
                "training_code_git_commit": provenance.get(
                    "training_code_git_commit"
                ),
                "data_provenance_git_commit": provenance.get(
                    "data_provenance_git_commit"
                ),
            },
        )
        expected_names = {
            "best_model.pt",
            "last_model.pt",
            "metrics.csv",
            "validation_predictions.csv",
            "validation_outputs.npz",
            "confusion_matrix.csv",
            "per_class_metrics.csv",
            "resolved_config.json",
            "training_summary.json",
            "run_manifest.json",
        }
        _write_run_manifest(staging)
        _validate_run_manifest(staging, expected_names)
        validate_validation_outputs(
            staging / "validation_outputs.npz",
            expected_class_order=normalized_order,
        )
        _strict_json(staging / "resolved_config.json")
        _strict_json(staging / "training_summary.json")
    return {
        "status": "success",
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "output_dir": str(Path(output_dir)),
    }


@dataclass(frozen=True)
class TrainingArtifacts:
    config: dict[str, object]
    train_dataset: IMUStage2Dataset
    validation_dataset: IMUStage2Dataset
    train_sampler: LengthBucketBatchSampler
    validation_sampler: LengthBucketBatchSampler
    train_loader: DataLoader[dict[str, object]]
    validation_loader: DataLoader[dict[str, object]]
    model: nn.Module
    metadata: dict[str, object]
    class_order: list[dict[str, object]]
    provenance: dict[str, object]


def _build_artifacts(
    *,
    config_path: Path,
    stage2_root: Path,
    training_index_dir: Path,
    normalization_dir: Path,
) -> TrainingArtifacts:
    config = load_training_config(config_path)
    set_deterministic_seed(int(config["seed"]))
    stage2_root = Path(stage2_root).resolve(strict=True)
    training_index_dir = Path(training_index_dir).resolve(strict=True)
    normalization_dir = Path(normalization_dir).resolve(strict=True)
    training_index_path = training_index_dir / "training_index.csv"
    training_metadata_path = training_index_dir / "training_index.json"
    class_order_path = training_index_dir / "class_order.json"
    normalization_npz = normalization_dir / "imu_normalization.npz"
    normalization_json = normalization_dir / "imu_normalization.json"
    schema_path = stage2_root / "schema.json"
    frame = pd.read_csv(training_index_path, encoding="utf-8-sig", keep_default_na=False)
    training_metadata = _strict_json(training_metadata_path)
    normalization_metadata = _strict_json(normalization_json)
    class_order = load_class_order(class_order_path)
    schema = load_stage2_schema(schema_path)
    if class_order.num_classes != config["num_classes"]:
        raise ValueError("Configured num_classes disagrees with class order")
    contract = normalization_metadata.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Normalization contract is missing")
    validate_training_artifact_bindings(
        training_metadata,
        stage2_contract_sha256=str(schema["contract_sha256"]),
        class_order_sha256=class_order.class_order_sha256,
        num_classes=class_order.num_classes,
        expected_fold=int(config["fold"]),
        normalization_fold=contract.get("fold"),
        split_definition_path=str(training_metadata.get("split_definition_path")),
    )
    metadata = build_training_checkpoint_metadata(
        stage2_contract_sha256=str(schema["contract_sha256"]),
        training_index_sha256=str(training_metadata["training_index_sha256"]),
        normalization_contract_sha256=str(
            normalization_metadata["normalization_contract_sha256"]
        ),
        normalization_file_sha256=sha256_file(normalization_npz),
        class_order_sha256=class_order.class_order_sha256,
        num_classes=class_order.num_classes,
    )
    common = {
        "training_index": frame,
        "stage2_root": stage2_root,
        "stage2_schema": schema_path,
        "normalization_npz": normalization_npz,
        "normalization_json": normalization_json,
        "training_index_metadata": training_metadata,
        "hard_safety_limit_t": int(config["hard_safety_limit_t"]),
    }
    train_dataset = IMUStage2Dataset(**common, split="train")
    validation_dataset = IMUStage2Dataset(**common, split="validation")
    sampler_common = {
        "bucket_boundaries": config["bucket_boundaries"],
        "batch_feature_budget": int(config["batch_feature_budget"]),
        "maximum_batch_size": int(config["maximum_batch_size"]),
        "minimum_batch_size": int(config["minimum_batch_size"]),
        "drop_last": bool(config["drop_last"]),
    }
    train_sampler = LengthBucketBatchSampler(
        lengths=train_dataset.lengths,
        shuffle_seed=int(config["seed"]),
        **sampler_common,
    )
    validation_sampler = LengthBucketBatchSampler(
        lengths=validation_dataset.lengths,
        shuffle_seed=int(config["seed"]),
        **sampler_common,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_imu_stage2,
        num_workers=int(config["num_workers"]),
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_sampler=validation_sampler,
        collate_fn=collate_imu_stage2,
        num_workers=int(config["num_workers"]),
    )
    model = build_imu_stage2_model(config, num_classes=class_order.num_classes)
    schema_provenance = schema.get("provenance")
    if not isinstance(schema_provenance, dict):
        raise ValueError("Stage 2 schema provenance is missing")
    training_code_git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip()
    provenance = {
        "training_code_git_commit": training_code_git_commit,
        "data_provenance_git_commit": schema_provenance.get("git_commit"),
        "input_hashes": {
            key: metadata[key]
            for key in (
                "stage2_contract_sha256",
                "training_index_sha256",
                "normalization_contract_sha256",
                "normalization_file_sha256",
                "class_order_sha256",
            )
        },
    }
    return TrainingArtifacts(
        config=config,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        train_sampler=train_sampler,
        validation_sampler=validation_sampler,
        train_loader=train_loader,
        validation_loader=validation_loader,
        model=model,
        metadata=metadata,
        class_order=[dict(record) for record in class_order.classes],
        provenance=provenance,
    )


def _length_summary(lengths: Sequence[int]) -> dict[str, object]:
    values = np.asarray(lengths, dtype=np.int64)
    return {
        "minimum": int(values.min()),
        "maximum": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def _bucket_counts(lengths: Sequence[int], boundaries: Sequence[int]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for length in lengths:
        upper = next((boundary for boundary in boundaries if length <= boundary), None)
        label = f"<= {upper}" if upper is not None else f"> {boundaries[-1]}"
        counts[label] += 1
    return dict(sorted(counts.items()))


def preflight_training(
    *,
    config_path: Path,
    stage2_root: Path,
    training_index_dir: Path,
    normalization_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    artifacts = _build_artifacts(
        config_path=config_path,
        stage2_root=stage2_root,
        training_index_dir=training_index_dir,
        normalization_dir=normalization_dir,
    )
    artifacts.model.to(device=device, dtype=torch.float32).eval()
    first_validation = next(iter(artifacts.validation_loader))
    with torch.inference_mode():
        result = artifacts.model(_move_batch(first_validation, device))
    logits = result["logits"]
    embeddings = result["embedding"]
    _assert_finite_tensor(logits, "validation logits")
    _assert_finite_tensor(embeddings, "validation embedding")
    train_batches = len(artifacts.train_sampler)
    validation_batches = len(artifacts.validation_sampler)
    parameter_count = sum(parameter.numel() for parameter in artifacts.model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in artifacts.model.parameters()
    )
    all_lengths = artifacts.train_dataset.lengths + artifacts.validation_dataset.lengths
    return {
        "status": "preflight_ok",
        "train_samples": len(artifacts.train_dataset),
        "validation_samples": len(artifacts.validation_dataset),
        "selected_samples": len(artifacts.train_dataset) + len(artifacts.validation_dataset),
        "sequence_lengths": _length_summary(all_lengths),
        "bucket_counts": _bucket_counts(all_lengths, artifacts.config["bucket_boundaries"]),
        "estimated_train_batches": train_batches,
        "estimated_validation_batches": validation_batches,
        "sampler_omitted_train": len(artifacts.train_sampler.omitted_indices),
        "sampler_omitted_validation": len(artifacts.validation_sampler.omitted_indices),
        "parameter_count": parameter_count,
        "parameter_bytes_float32": parameter_bytes,
        "validation_batch_size": int(logits.shape[0]),
        "validation_logits_shape": list(logits.shape),
        "validation_embedding_shape": list(embeddings.shape),
        "validation_logits_finite": bool(torch.isfinite(logits).all()),
        "validation_embeddings_finite": bool(torch.isfinite(embeddings).all()),
        "checkpoint_metadata": artifacts.metadata,
    }


def train_from_artifacts(
    *,
    config_path: Path,
    stage2_root: Path,
    training_index_dir: Path,
    normalization_dir: Path,
    output_dir: Path,
    device: torch.device,
) -> dict[str, object]:
    artifacts = _build_artifacts(
        config_path=config_path,
        stage2_root=stage2_root,
        training_index_dir=training_index_dir,
        normalization_dir=normalization_dir,
    )
    return fit_model(
        model=artifacts.model,
        train_loader=artifacts.train_loader,
        validation_loader=artifacts.validation_loader,
        output_dir=output_dir,
        config=artifacts.config,
        metadata=artifacts.metadata,
        class_order=artifacts.class_order,
        provenance=artifacts.provenance,
        device=device,
    )
