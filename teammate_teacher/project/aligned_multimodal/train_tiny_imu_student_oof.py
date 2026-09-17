from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from imu_data import DEVICES, read_index
from imu_model import DeviceAwareIMUStudent
from run_imu_stat_baseline import (
    drop_devices,
    feature_vector,
    random_present_devices,
)


PROJECT_DIR = Path(__file__).resolve().parent
FEATURES_PER_DEVICE = 48
MASKS_PER_DEVICE = 2
NUM_CLASSES = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fixed-epoch sub-1-MiB IMU student with fold-local RF distillation"
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_DIR / "cache" / "imu_32"
    )
    parser.add_argument(
        "--fold-summary",
        type=Path,
        default=PROJECT_DIR / "data" / "subject_folds" / "folds_summary.json",
    )
    parser.add_argument(
        "--reference-teacher-oof",
        type=Path,
        default=(
            PROJECT_DIR
            / "runs"
            / "p3_imu_stat"
            / "random_forest_device_dropout_oof.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p20_tiny_imu_student_oof",
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--hard-loss-weight", type=float, default=0.60)
    parser.add_argument("--distillation-temperature", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=20260723)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def classification_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    }


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def fit_normalizer(
    features: np.ndarray,
    masks: np.ndarray,
    train: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    device_features = features[train].reshape(-1, len(DEVICES), FEATURES_PER_DEVICE)
    device_masks = masks[train].reshape(-1, len(DEVICES), MASKS_PER_DEVICE)
    mean = np.zeros((len(DEVICES), FEATURES_PER_DEVICE), dtype=np.float32)
    std = np.ones((len(DEVICES), FEATURES_PER_DEVICE), dtype=np.float32)
    for device in range(len(DEVICES)):
        present = device_masks[:, device, 0] > 0
        if not np.any(present):
            continue
        values = device_features[present, device]
        mean[device] = values.mean(axis=0)
        std[device] = np.maximum(values.std(axis=0), 1e-5)
    return mean, std


def student_inputs(
    features: np.ndarray,
    masks: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    device_features = features.reshape(-1, len(DEVICES), FEATURES_PER_DEVICE)
    device_masks = masks.reshape(-1, len(DEVICES), MASKS_PER_DEVICE)
    normalized = (device_features - mean[None]) / std[None]
    normalized *= (device_masks[..., :1] > 0).astype(np.float32)
    return np.concatenate([normalized, device_masks], axis=2).astype(
        np.float32, copy=False
    )


def probability_logits(
    model: RandomForestClassifier, source: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = model.predict_proba(source)
    dense = np.full((len(source), NUM_CLASSES), 1e-12, dtype=np.float32)
    dense[:, model.classes_.astype(np.int64)] = probabilities.astype(np.float32)
    dense /= dense.sum(axis=1, keepdims=True)
    return dense, np.log(np.clip(dense, 1e-12, 1.0))


@torch.inference_mode()
def infer_student(
    model: DeviceAwareIMUStudent,
    source: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    output: list[np.ndarray] = []
    for start in range(0, len(source), batch_size):
        batch = torch.from_numpy(source[start : start + batch_size]).to(device)
        output.append(model(batch).float().cpu().numpy())
    return np.concatenate(output)


def train_student(
    model: DeviceAwareIMUStudent,
    train_source: np.ndarray,
    train_labels: np.ndarray,
    teacher_log_probabilities: np.ndarray,
    val_source: np.ndarray,
    val_labels: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    source = torch.from_numpy(train_source).to(device)
    labels = torch.from_numpy(train_labels).long().to(device)
    teacher_log = torch.from_numpy(teacher_log_probabilities).to(device)
    counts = torch.bincount(labels, minlength=NUM_CLASSES).float().clamp_min(1.0)
    sample_weights = (1.0 / counts[labels]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(args.epochs)
    )
    temperature = float(args.distillation_temperature)
    hard_weight = float(args.hard_loss_weight)
    history: list[dict[str, float | int]] = []

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        sampled = torch.multinomial(
            sample_weights, len(sample_weights), replacement=True
        )
        loss_sum = 0.0
        hard_sum = 0.0
        kd_sum = 0.0
        correct = 0
        count = 0
        for start in range(0, len(sampled), int(args.batch_size)):
            indices = sampled[start : start + int(args.batch_size)]
            optimizer.zero_grad(set_to_none=True)
            logits = model(source[indices])
            hard_loss = F.cross_entropy(
                logits, labels[indices], label_smoothing=0.05
            )
            teacher_soft = torch.softmax(teacher_log[indices] / temperature, dim=1)
            kd_loss = (
                F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    teacher_soft,
                    reduction="batchmean",
                )
                * temperature
                * temperature
            )
            loss = hard_weight * hard_loss + (1.0 - hard_weight) * kd_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            batch_count = len(indices)
            loss_sum += float(loss.detach()) * batch_count
            hard_sum += float(hard_loss.detach()) * batch_count
            kd_sum += float(kd_loss.detach()) * batch_count
            correct += int((logits.argmax(1) == labels[indices]).sum())
            count += batch_count
        scheduler.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == int(args.epochs):
            val_logits = infer_student(model, val_source, device)
            val_predictions = val_logits.argmax(1)
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": loss_sum / max(count, 1),
                "train_hard_loss": hard_sum / max(count, 1),
                "train_kd_loss": kd_sum / max(count, 1),
                "sampled_train_accuracy": correct / max(count, 1),
                "val_accuracy_monitor_only": float(
                    np.mean(val_predictions == val_labels)
                ),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    return infer_student(model, val_source, device), history


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = args.cache_dir.resolve()
    rows = [
        row
        for row in read_index(cache / "index.csv")
        if row.split == "train" and row.usable
    ]
    # These arrays are small enough to load eagerly. Row-wise access through
    # memmap caused severe page-fault/I/O stalls on Windows while extracting
    # the 250 statistical features, even though model training itself was fast.
    values = np.load(cache / "imu_float32.npy", allow_pickle=False)
    time_mask = np.load(cache / "time_mask_uint8.npy", allow_pickle=False)
    device_mask = np.load(cache / "device_mask_uint8.npy", allow_pickle=False)
    extracted_features: list[np.ndarray] = []
    extracted_masks: list[np.ndarray] = []
    for row in rows:
        feature, mask = feature_vector(
            values[row.cache_index],
            time_mask[row.cache_index],
            device_mask[row.cache_index],
        )
        extracted_features.append(feature)
        extracted_masks.append(mask)
    features = np.stack(extracted_features)
    masks = np.stack(extracted_masks)
    flat_source = np.concatenate([features, masks], axis=1)
    labels = np.asarray([row.class_id for row in rows], dtype=np.int64)
    users = np.asarray([row.user_id for row in rows])
    sample_ids = np.asarray([row.sample_id for row in rows])
    fold_summary = json.loads(
        args.fold_summary.resolve().read_text(encoding="utf-8")
    )
    held_folds = np.full(len(rows), -1, dtype=np.int64)
    for fold_info in fold_summary["folds"]:
        held_folds[np.isin(users, fold_info["val_users"])] = int(fold_info["fold"])
    if np.any(held_folds < 0):
        raise ValueError("Some usable IMU rows do not belong to a subject fold")

    reference = np.load(args.reference_teacher_oof.resolve(), allow_pickle=False)
    reference_lookup = {
        sample_id: index
        for index, sample_id in enumerate(
            reference["sample_ids"].astype(str).tolist()
        )
    }
    student_oof = np.zeros((len(rows), NUM_CLASSES), dtype=np.float32)
    teacher_oof = np.zeros((len(rows), NUM_CLASSES), dtype=np.float32)
    fold_summaries: list[dict[str, object]] = []
    total_started = time.time()

    for fold_info in fold_summary["folds"]:
        fold = int(fold_info["fold"])
        fold_started = time.time()
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train = np.isin(users, fold_info["train_users"])
        val = np.isin(users, fold_info["val_users"])
        if set(users[train].tolist()) & set(users[val].tolist()):
            raise RuntimeError("Train and validation subjects overlap")

        rng = np.random.default_rng(int(args.seed) + fold)
        dropped_features, dropped_masks = drop_devices(
            features[train],
            masks[train],
            random_present_devices(masks[train], rng),
        )
        teacher_fit_source = np.concatenate(
            [
                flat_source[train],
                np.concatenate([dropped_features, dropped_masks], axis=1),
            ],
            axis=0,
        )
        fit_labels = np.concatenate([labels[train], labels[train]])
        teacher = RandomForestClassifier(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=int(args.seed),
        )
        teacher_fit_started = time.time()
        teacher.fit(teacher_fit_source, fit_labels)
        teacher_fit_seconds = time.time() - teacher_fit_started
        teacher_fit_probabilities, teacher_fit_logits = probability_logits(
            teacher, teacher_fit_source
        )
        del teacher_fit_probabilities
        _, fold_teacher_logits = probability_logits(teacher, flat_source[val])
        teacher_oof[val] = fold_teacher_logits

        reference_indices = np.asarray(
            [reference_lookup[sample_id] for sample_id in sample_ids[val].tolist()],
            dtype=np.int64,
        )
        reference_probabilities = np.exp(
            reference["logits"][reference_indices].astype(np.float64)
        )
        reproduced_probabilities = np.exp(fold_teacher_logits.astype(np.float64))
        teacher_reference_max_abs_delta = float(
            np.max(np.abs(reference_probabilities - reproduced_probabilities))
        )

        mean, std = fit_normalizer(features, masks, train)
        original_student_source = student_inputs(
            features[train], masks[train], mean, std
        )
        dropped_student_source = student_inputs(
            dropped_features, dropped_masks, mean, std
        )
        fit_student_source = np.concatenate(
            [original_student_source, dropped_student_source], axis=0
        )
        val_student_source = student_inputs(features[val], masks[val], mean, std)

        seed_everything(int(args.seed) + fold)
        model = DeviceAwareIMUStudent(dropout=float(args.dropout)).to(device)
        student_started = time.time()
        fold_student_logits, history = train_student(
            model,
            fit_student_source,
            fit_labels,
            teacher_fit_logits,
            val_student_source,
            labels[val],
            args,
            device,
        )
        student_seconds = time.time() - student_started
        student_oof[val] = fold_student_logits
        fold_predictions = fold_student_logits.argmax(1)
        teacher_predictions = fold_teacher_logits.argmax(1)

        config = {
            **{
                key: (
                    str(value.resolve())
                    if isinstance(value, Path)
                    else value
                )
                for key, value in vars(args).items()
            },
            "fold": fold,
            "train_users": sorted(set(users[train].tolist())),
            "val_users": sorted(set(users[val].tolist())),
            "device": str(device),
            "selection_protocol": (
                "fixed final epoch; validation trajectory is monitor-only"
            ),
        }
        checkpoint = {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "normalizer_mean": mean,
            "normalizer_std": std,
            "config": config,
            "epoch": int(args.epochs),
            "fixed_final_epoch": True,
        }
        checkpoint_path = fold_dir / "final.pt"
        torch.save(checkpoint, checkpoint_path)
        np.savez_compressed(
            fold_dir / "val_logits.npz",
            sample_ids=sample_ids[val],
            labels=labels[val],
            held_fold=np.full(int(val.sum()), fold, dtype=np.int64),
            student_logits=fold_student_logits,
            teacher_logits=fold_teacher_logits,
        )
        summary = {
            "fold": fold,
            "train_samples": int(train.sum()),
            "train_samples_after_device_dropout": int(len(fit_labels)),
            "val_samples": int(val.sum()),
            "parameters": parameter_count(model),
            "fp32_parameter_size_mib": parameter_count(model) * 4 / 1024**2,
            "checkpoint_size_mib": checkpoint_path.stat().st_size / 1024**2,
            "student_metrics": classification_metrics(
                labels[val], fold_predictions
            ),
            "teacher_metrics": classification_metrics(
                labels[val], teacher_predictions
            ),
            "teacher_reference_max_abs_probability_delta": (
                teacher_reference_max_abs_delta
            ),
            "teacher_fit_seconds": round(teacher_fit_seconds, 3),
            "student_train_seconds": round(student_seconds, 3),
            "elapsed_seconds": round(time.time() - fold_started, 3),
            "selection_protocol": config["selection_protocol"],
            "history": history,
        }
        (fold_dir / "metrics.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        fold_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        del teacher, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if np.any(~np.isfinite(student_oof)) or np.any(~np.isfinite(teacher_oof)):
        raise RuntimeError("Non-finite OOF logits")
    order = np.argsort(sample_ids.astype(str))
    np.savez_compressed(
        output_dir / "oof_logits.npz",
        sample_ids=sample_ids[order],
        labels=labels[order],
        held_fold=held_folds[order],
        student_logits=student_oof[order],
        teacher_logits=teacher_oof[order],
    )
    overall = {
        "status": "complete",
        "experiment": "tiny_device_aware_imu_student",
        "device": str(device),
        "samples": int(len(labels)),
        "parameters": int(fold_summaries[0]["parameters"]),
        "max_checkpoint_size_mib": float(
            max(float(row["checkpoint_size_mib"]) for row in fold_summaries)
        ),
        "student_oof_metrics": classification_metrics(
            labels, student_oof.argmax(1)
        ),
        "teacher_oof_metrics": classification_metrics(
            labels, teacher_oof.argmax(1)
        ),
        "folds": fold_summaries,
        "elapsed_seconds": round(time.time() - total_started, 3),
        "selection_protocol": (
            "three fixed subject-disjoint folds; fold-local normalizer and RF "
            "teacher; fixed final epoch student"
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
