from __future__ import annotations

import hashlib
import json
import math
import os
import random
import shutil
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
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
) -> None:
    expected = {
        "stage2_contract_sha256": stage2_contract_sha256,
        "class_order_sha256": class_order_sha256,
        "num_classes": num_classes,
    }
    for field, value in expected.items():
        if training_metadata.get(field) != value:
            raise ValueError(f"Training artifact {field} mismatch")


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
        f1_values.append(f1)
    accuracy = float(np.trace(confusion) / labels.size) if labels.size else 0.0
    return {
        "accuracy": accuracy,
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
    last_gradient_norm = 0.0
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
        last_gradient_norm = float(torch.as_tensor(gradient_norm).detach().cpu())
        batches += 1
    if batches == 0 or total_samples == 0:
        raise ValueError("Training loader produced no samples")
    return {
        "loss": total_loss / total_samples,
        "gradient_norm": last_gradient_norm,
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
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized_metadata = _validate_training_metadata(metadata)
    payload = {
        "checkpoint_version": "imu-stage2-training-state-v1",
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "metrics": dict(metrics),
        "checkpoint_metadata": normalized_metadata,
        "training_config": dict(config),
    }
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


def write_validation_outputs(
    path: Path,
    *,
    sample_ids: Sequence[str],
    labels: np.ndarray,
    logits: np.ndarray,
    embeddings: np.ndarray,
) -> None:
    path = Path(path)
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    embeddings = np.asarray(embeddings)
    if labels.dtype != np.int64 or labels.shape != (len(sample_ids),):
        raise ValueError("Validation labels are invalid")
    if logits.dtype != np.float32 or logits.ndim != 2 or logits.shape[0] != len(sample_ids):
        raise ValueError("Validation logits are invalid")
    if embeddings.dtype != np.float32 or embeddings.shape != (len(sample_ids), 128):
        raise ValueError("Validation embeddings must have shape [N,128]")
    if not np.isfinite(logits).all() or not np.isfinite(embeddings).all():
        raise ValueError("Validation outputs must be finite")
    sample_array = np.asarray(list(map(str, sample_ids)), dtype=np.str_)
    temporary = path.parent / f".{path.name}.tmp-{uuid4().hex}.npz"
    try:
        np.savez(
            temporary,
            sample_ids=sample_array,
            labels=labels,
            logits=logits,
            embeddings=embeddings,
        )
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def fit_model(
    *,
    model: nn.Module,
    train_loader: Iterable[Mapping[str, object]],
    validation_loader: Iterable[Mapping[str, object]],
    output_dir: Path,
    config: Mapping[str, object],
    metadata: Mapping[str, object],
    device: torch.device,
) -> dict[str, object]:
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
            }
            history.append(epoch_record)
            save_checkpoint(
                staging / "last.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                metrics=comparable,
                metadata=metadata,
                config=config,
            )
            if improved:
                best_epoch = epoch
                save_checkpoint(
                    staging / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    metrics=comparable,
                    metadata=metadata,
                    config=config,
                )
            should_stop = stopper.observe(comparable)
            scheduler.step()
            if should_stop:
                break
        load_checkpoint(staging / "best.pt", model=model, expected_metadata=metadata)
        best_validation = evaluate(
            model,
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
        )
        metrics_payload = {
            "training_run_version": "imu-stage2-training-run-v1",
            "best_epoch": best_epoch,
            "epochs_completed": len(history),
            "best_validation": {
                key: value
                for key, value in best_validation.items()
                if key not in {"logits", "embeddings", "labels", "sample_ids"}
            },
            "history": history,
        }
        _write_json(staging / "metrics.json", metrics_payload)
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
    validate_training_artifact_bindings(
        training_metadata,
        stage2_contract_sha256=str(schema["contract_sha256"]),
        class_order_sha256=class_order.class_order_sha256,
        num_classes=class_order.num_classes,
    )
    contract = normalization_metadata.get("contract")
    if not isinstance(contract, dict):
        raise ValueError("Normalization contract is missing")
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
        device=device,
    )
