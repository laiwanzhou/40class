from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from imu_data import read_index
from imu_model import DeviceAwareIMUStudent
from probe_p27r3_incremental_information import metric_bundle
from run_imu_stat_baseline import (
    drop_devices,
    feature_vector,
    random_present_devices,
)
from train_tiny_imu_student_oof import (
    fit_normalizer,
    parameter_count,
    probability_logits,
    seed_everything,
    student_inputs,
    train_student,
)


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the fixed P20 tiny IMU student on P27 outer-train "
            "subject-disjoint development folds."
        )
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=PROJECT_DIR / "data" / "p27_strong_inner",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=PROJECT_DIR / "cache" / "imu_32"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p27_strong_inner" / "tiny_imu_student",
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = args.cache_dir.resolve()
    rows = [
        row
        for row in read_index(cache / "index.csv")
        if row.split == "train" and row.usable
    ]
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
    sample_ids = np.asarray([row.sample_id for row in rows])
    device_counts = device_mask[
        np.asarray([row.cache_index for row in rows], dtype=np.int64)
    ].sum(axis=1).astype(np.float32)
    location = {str(sample_id): index for index, sample_id in enumerate(sample_ids)}
    fold_summaries: list[dict[str, object]] = []
    started = time.time()

    for fold in range(3):
        manifest = read_csv(args.manifest_dir.resolve() / f"fold_{fold}.csv")
        train_ids = {
            row["sample_id"] for row in manifest if row["split"] == "train"
        }
        held_ids = {row["sample_id"] for row in manifest if row["split"] == "val"}
        train = np.asarray(
            [location[sample_id] for sample_id in sorted(train_ids & set(location))],
            dtype=np.int64,
        )
        held = np.asarray(
            [location[sample_id] for sample_id in sorted(held_ids & set(location))],
            dtype=np.int64,
        )
        if set(sample_ids[train].tolist()) & set(sample_ids[held].tolist()):
            raise RuntimeError("train/held sample overlap")
        rng = np.random.default_rng(int(args.seed) + fold)
        dropped_features, dropped_masks = drop_devices(
            features[train],
            masks[train],
            random_present_devices(masks[train], rng),
        )
        teacher_source = np.concatenate(
            [
                flat_source[train],
                np.concatenate([dropped_features, dropped_masks], axis=1),
            ]
        )
        fit_labels = np.concatenate([labels[train], labels[train]])
        teacher = RandomForestClassifier(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=int(args.seed) + fold,
        )
        teacher.fit(teacher_source, fit_labels)
        _, teacher_train_logits = probability_logits(teacher, teacher_source)
        _, teacher_held_logits = probability_logits(teacher, flat_source[held])
        mean, std = fit_normalizer(features, masks, train)
        student_train_source = np.concatenate(
            [
                student_inputs(features[train], masks[train], mean, std),
                student_inputs(dropped_features, dropped_masks, mean, std),
            ]
        )
        student_held_source = student_inputs(
            features[held], masks[held], mean, std
        )
        seed_everything(int(args.seed) + fold)
        model = DeviceAwareIMUStudent(dropout=float(args.dropout)).to(device)
        student_held_logits, history = train_student(
            model,
            student_train_source,
            fit_labels,
            teacher_train_logits,
            student_held_source,
            labels[held],
            args,
            device,
        )
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in model.state_dict().items()
            },
            "normalizer_mean": mean,
            "normalizer_std": std,
            "config": {
                **{
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in vars(args).items()
                },
                "fold": fold,
                "fixed_final_epoch": True,
            },
            "epoch": int(args.epochs),
        }
        torch.save(checkpoint, fold_dir / "final.pt")
        np.savez_compressed(
            output / f"fold_{fold}_logits.npz",
            protocol=np.asarray("p27-p20-tiny-imu-inner-v1"),
            sample_ids=sample_ids[held],
            labels=labels[held],
            imu_logits=student_held_logits.astype(np.float32),
            teacher_logits=teacher_held_logits.astype(np.float32),
            imu_device_counts=device_counts[held],
            outer_held_predictions_generated=np.asarray(False),
        )
        summary = {
            "fold": fold,
            "train_samples": int(len(train)),
            "held_samples": int(len(held)),
            "parameters": int(parameter_count(model)),
            "student_metrics": metric_bundle(
                labels[held], student_held_logits.argmax(axis=1)
            ),
            "teacher_metrics": metric_bundle(
                labels[held], teacher_held_logits.argmax(axis=1)
            ),
            "history": history,
            "outer_held_predictions_generated": False,
        }
        (fold_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        fold_summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        del teacher, model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    aggregate = {
        "protocol": "p27-p20-tiny-imu-inner-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "folds": fold_summaries,
        "elapsed_seconds": time.time() - started,
    }
    (output / "summary.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
