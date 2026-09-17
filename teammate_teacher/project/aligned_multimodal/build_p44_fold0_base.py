from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

from aligned_data import AlignedMultimodalDataset
from imu_data import read_index
from residual_logit_fusion import build_model
from run_imu_stat_baseline import drop_devices, feature_vector, random_present_devices


PROJECT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the P44 fold0-pilot P12 Skeleton+Depth+RF-IMU base."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_DIR / "data" / "p27_strong_inner" / "fold_0.csv",
    )
    parser.add_argument(
        "--skeleton-checkpoint",
        type=Path,
        default=PROJECT_DIR
        / "runs"
        / "p44_p12_base_inner_oof"
        / "skeleton"
        / "fold_0"
        / "best_accuracy.pt",
    )
    parser.add_argument(
        "--depth-checkpoint",
        type=Path,
        default=PROJECT_DIR
        / "runs"
        / "p44_p12_base_inner_oof"
        / "depth"
        / "fold_0"
        / "best_accuracy.pt",
    )
    parser.add_argument(
        "--imu-cache", type=Path, default=PROJECT_DIR / "cache" / "imu_32"
    )
    parser.add_argument(
        "--outer-folds",
        type=Path,
        default=PROJECT_DIR / "data" / "subject_folds" / "folds_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "runs" / "p44_p12_base_inner_oof" / "fold0_pilot",
    )
    parser.add_argument("--seed", type=int, default=44001)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_dict(labels: np.ndarray, logits: np.ndarray) -> dict[str, float]:
    predictions = logits.argmax(1)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
    }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def dense_log_probabilities(
    model: RandomForestClassifier, source: np.ndarray
) -> np.ndarray:
    probabilities = model.predict_proba(source)
    logits = np.full((len(source), 40), np.log(1e-12), dtype=np.float32)
    logits[:, model.classes_.astype(np.int64)] = np.log(
        np.clip(probabilities, 1e-12, 1.0)
    )
    return logits


def infer_visual_experts(
    manifest_path: Path,
    skeleton_checkpoint_path: Path,
    depth_checkpoint_path: Path,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    paths = {
        "skeleton": skeleton_checkpoint_path.resolve(),
        "depth": depth_checkpoint_path.resolve(),
    }
    checkpoints = {
        name: torch.load(path, map_location="cpu", weights_only=False)
        for name, path in paths.items()
    }
    configs = {name: checkpoints[name]["config"] for name in paths}
    if list(configs["skeleton"]["modalities"]) != ["skeleton"]:
        raise ValueError("Skeleton checkpoint is not skeleton-only")
    if list(configs["depth"]["modalities"]) != ["depth"]:
        raise ValueError("Depth checkpoint is not depth-only")
    for key in ("num_frames", "image_height", "image_width"):
        if int(configs["skeleton"][key]) != int(configs["depth"][key]):
            raise ValueError(f"Checkpoint {key} mismatch")

    dataset = AlignedMultimodalDataset(
        manifest_path=manifest_path,
        split="val",
        modalities=["depth", "skeleton"],
        num_frames=int(configs["depth"]["num_frames"]),
        image_height=int(configs["depth"]["image_height"]),
        image_width=int(configs["depth"]["image_width"]),
        augment=False,
        cache_dir=configs["depth"].get("cache_dir"),
        skeleton_strategy=configs["skeleton"].get("skeleton_strategy", "first"),
        depth_representation=configs["depth"].get("depth_representation", "jet_rgb"),
        visual_normalization=configs["depth"].get("visual_normalization", "legacy"),
        skeleton_representation=configs["skeleton"].get(
            "skeleton_representation", "frame_joint"
        ),
        skeleton_raw_cache_dir=configs["skeleton"].get("skeleton_raw_cache_dir"),
        temporal_sampling="uniform",
    )
    loader = DataLoader(
        dataset,
        batch_size=int(configs["depth"]["batch_size"]),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {name: build_model(checkpoint, device) for name, checkpoint in checkpoints.items()}
    use_amp = bool(configs["depth"].get("use_amp", True) and device.type == "cuda")
    sample_ids: list[str] = []
    labels: list[int] = []
    logits: dict[str, list[torch.Tensor]] = {"skeleton": [], "depth": []}
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                logits["skeleton"].append(
                    models["skeleton"](
                        {"skeleton": batch["skeleton"].to(device, non_blocking=True)}
                    )
                    .float()
                    .cpu()
                )
                logits["depth"].append(
                    models["depth"](
                        {"depth": batch["depth"].to(device, non_blocking=True)}
                    )
                    .float()
                    .cpu()
                )
            sample_ids.extend(batch["sample_id"])
            labels.extend(batch["label"].tolist())
    info = {
        "device": str(device),
        "seconds": time.perf_counter() - started,
        "checkpoints": {
            name: {
                "path": str(path),
                "sha256": sha256(path),
                "epoch": int(checkpoints[name]["epoch"]),
            }
            for name, path in paths.items()
        },
    }
    return (
        np.asarray(sample_ids),
        np.asarray(labels, dtype=np.int64),
        torch.cat(logits["skeleton"]).numpy(),
        torch.cat(logits["depth"]).numpy(),
        info,
    )


def write_rows(
    path: Path,
    sample_ids: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    class_names: dict[int, str],
    skeleton_logits: np.ndarray,
    depth_logits: np.ndarray,
    imu_logits: np.ndarray,
    imu_present: np.ndarray,
    imu_device_counts: np.ndarray,
    sd_logits: np.ndarray,
    base_logits: np.ndarray,
) -> None:
    probabilities = softmax(base_logits)
    top_order = np.argsort(-probabilities, axis=1)[:, :5]
    top_values = np.take_along_axis(probabilities, top_order, axis=1)
    entropy = -(probabilities * np.log(np.clip(probabilities, 1e-12, 1.0))).sum(1)
    skeleton_predictions = skeleton_logits.argmax(1)
    depth_predictions = depth_logits.argmax(1)
    imu_predictions = imu_logits.argmax(1)
    base_predictions = base_logits.argmax(1)
    fields = [
        "sample_id",
        "inner_fold",
        "user_id",
        "class_id",
        "class_name",
        "skeleton_prediction",
        "depth_prediction",
        "imu_prediction",
        "imu_present",
        "imu_device_count",
        "sd_prediction",
        "base_prediction",
        "base_correct",
        "base_confidence",
        "base_margin",
        "base_entropy",
        "top5_class_ids",
        "top5_probabilities",
        "expert_agreement_count",
        "all_present_experts_agree",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, sample_id in enumerate(sample_ids.astype(str)):
            expert_predictions = [
                int(skeleton_predictions[index]),
                int(depth_predictions[index]),
            ]
            if imu_present[index]:
                expert_predictions.append(int(imu_predictions[index]))
            agreement = max(Counter(expert_predictions).values())
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "inner_fold": 0,
                    "user_id": str(subjects[index]),
                    "class_id": int(labels[index]),
                    "class_name": class_names[int(labels[index])],
                    "skeleton_prediction": int(skeleton_predictions[index]),
                    "depth_prediction": int(depth_predictions[index]),
                    "imu_prediction": (
                        int(imu_predictions[index]) if imu_present[index] else -1
                    ),
                    "imu_present": int(imu_present[index]),
                    "imu_device_count": int(imu_device_counts[index]),
                    "sd_prediction": int(sd_logits[index].argmax()),
                    "base_prediction": int(base_predictions[index]),
                    "base_correct": int(base_predictions[index] == labels[index]),
                    "base_confidence": float(top_values[index, 0]),
                    "base_margin": float(top_values[index, 0] - top_values[index, 1]),
                    "base_entropy": float(entropy[index]),
                    "top5_class_ids": json.dumps(top_order[index].tolist()),
                    "top5_probabilities": json.dumps(
                        [round(float(value), 8) for value in top_values[index]]
                    ),
                    "expert_agreement_count": int(agreement),
                    "all_present_experts_agree": int(
                        agreement == len(expert_predictions)
                    ),
                }
            )


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.resolve()
    manifest_rows = read_csv(manifest_path)
    train_rows = [row for row in manifest_rows if row["split"] == "train"]
    val_rows = [row for row in manifest_rows if row["split"] == "val"]
    train_subjects = sorted({row["user_id"] for row in train_rows})
    val_subjects = sorted({row["user_id"] for row in val_rows})
    if set(train_subjects) & set(val_subjects):
        raise RuntimeError("Inner train/val subjects overlap")
    outer = json.loads(args.outer_folds.resolve().read_text(encoding="utf-8"))
    outer_fold0 = next(row for row in outer["folds"] if int(row["fold"]) == 0)
    outer_train_subjects = set(outer_fold0["train_users"])
    outer_held_subjects = set(outer_fold0["val_users"])
    if not (set(train_subjects) | set(val_subjects)) <= outer_train_subjects:
        raise RuntimeError("Inner manifest contains a subject outside outer-train")
    if (set(train_subjects) | set(val_subjects)) & outer_held_subjects:
        raise RuntimeError("Outer-held subject leakage detected")

    (
        sample_ids,
        labels,
        skeleton_logits,
        depth_logits,
        inference_info,
    ) = infer_visual_experts(
        manifest_path,
        args.skeleton_checkpoint,
        args.depth_checkpoint,
        int(args.num_workers),
    )
    meta_by_id = {row["sample_id"]: row for row in val_rows}
    if set(sample_ids.astype(str)) != set(meta_by_id):
        raise RuntimeError("Inference sample IDs do not match fold0 validation rows")
    subjects = np.asarray([meta_by_id[str(sample_id)]["user_id"] for sample_id in sample_ids])
    class_names = {
        int(row["class_id"]): row["class_name"] for row in manifest_rows
    }

    imu_cache = args.imu_cache.resolve()
    imu_rows = [row for row in read_index(imu_cache / "index.csv") if row.split == "train"]
    values = np.load(imu_cache / "imu_float32.npy", mmap_mode="r", allow_pickle=False)
    time_mask = np.load(imu_cache / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    device_mask = np.load(imu_cache / "device_mask_uint8.npy", mmap_mode="r", allow_pickle=False)
    feature_by_id: dict[str, np.ndarray] = {}
    mask_by_id: dict[str, np.ndarray] = {}
    imu_meta_by_id = {row.sample_id: row for row in imu_rows}
    for row in imu_rows:
        feature, mask = feature_vector(
            values[row.cache_index], time_mask[row.cache_index], device_mask[row.cache_index]
        )
        feature_by_id[row.sample_id] = feature
        mask_by_id[row.sample_id] = mask
    fit_rows = [
        row
        for row in imu_rows
        if row.usable and row.user_id in set(train_subjects)
    ]
    fit_features = np.stack([feature_by_id[row.sample_id] for row in fit_rows])
    fit_masks = np.stack([mask_by_id[row.sample_id] for row in fit_rows])
    fit_labels = np.asarray([row.class_id for row in fit_rows], dtype=np.int64)
    rng = np.random.default_rng(int(args.seed))
    dropped_features, dropped_masks = drop_devices(
        fit_features, fit_masks, random_present_devices(fit_masks, rng)
    )
    rf_source = np.concatenate(
        [
            np.concatenate([fit_features, fit_masks], axis=1),
            np.concatenate([dropped_features, dropped_masks], axis=1),
        ],
        axis=0,
    )
    forest = RandomForestClassifier(
        n_estimators=400,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=int(args.seed),
    )
    rf_started = time.perf_counter()
    forest.fit(rf_source, np.concatenate([fit_labels, fit_labels]))
    rf_seconds = time.perf_counter() - rf_started
    rf_path = output / "fold0_imu_rf.joblib"
    joblib.dump(forest, rf_path, compress=3)

    imu_logits = np.zeros((len(sample_ids), 40), dtype=np.float32)
    imu_present = np.zeros(len(sample_ids), dtype=bool)
    imu_device_counts = np.zeros(len(sample_ids), dtype=np.int64)
    present_indices = [
        index
        for index, sample_id in enumerate(sample_ids.astype(str))
        if sample_id in imu_meta_by_id and imu_meta_by_id[sample_id].usable
    ]
    if present_indices:
        present_source = np.stack(
            [
                np.concatenate(
                    [feature_by_id[str(sample_ids[index])], mask_by_id[str(sample_ids[index])]]
                )
                for index in present_indices
            ]
        )
        imu_logits[present_indices] = dense_log_probabilities(forest, present_source)
        imu_present[present_indices] = True
        imu_device_counts[present_indices] = [
            imu_meta_by_id[str(sample_ids[index])].device_count for index in present_indices
        ]

    # Frozen P12 fusion form. The medians were fixed before this fold0 pilot;
    # no fold0 validation label is used to tune a temperature or weight.
    sd_temperature = 0.9167410586298098
    imu_temperature = 0.6312607866282122
    base_imu_weight = 0.4
    sd_logits = 0.6 * skeleton_logits + 0.4 * depth_logits
    per_sample_weight = (
        base_imu_weight * np.clip(imu_device_counts / 5.0, 0.0, 1.0)
    ).astype(np.float32)
    base_logits = sd_logits / sd_temperature
    base_logits = (
        (1.0 - per_sample_weight[:, None]) * base_logits
        + per_sample_weight[:, None] * imu_logits / imu_temperature
    )

    np.savez_compressed(
        output / "fold0_base_logits.npz",
        protocol=np.asarray("p44-fold0-pilot-p12-base-v1"),
        sample_ids=sample_ids,
        labels=labels,
        subjects=subjects,
        inner_folds=np.zeros(len(labels), dtype=np.int64),
        skeleton_logits=skeleton_logits.astype(np.float32),
        depth_logits=depth_logits.astype(np.float32),
        imu_logits=imu_logits.astype(np.float32),
        imu_present=imu_present.astype(np.uint8),
        imu_device_counts=imu_device_counts,
        sd_logits=sd_logits.astype(np.float32),
        base_logits=base_logits.astype(np.float32),
        per_sample_imu_weight=per_sample_weight,
        outer_held_predictions_generated=np.asarray(False),
    )
    write_rows(
        output / "fold0_base_rows.csv",
        sample_ids,
        labels,
        subjects,
        class_names,
        skeleton_logits,
        depth_logits,
        imu_logits,
        imu_present,
        imu_device_counts,
        sd_logits,
        base_logits,
    )
    summary = {
        "protocol": "p44-fold0-pilot-p12-base-v1",
        "status": "pilot_only_not_formal_three-fold-inner-oof",
        "counts": {
            "validation_samples": int(len(labels)),
            "validation_subjects": val_subjects,
            "training_subjects": train_subjects,
            "imu_present": int(imu_present.sum()),
            "imu_missing": int((~imu_present).sum()),
        },
        "purity_audit": {
            "inner_train_val_subject_overlap": [],
            "outer_fold": 0,
            "outer_held_subjects": sorted(outer_held_subjects),
            "outer_held_predictions_generated": False,
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
        },
        "fixed_fusion": {
            "sd_weights": {"skeleton": 0.6, "depth": 0.4},
            "sd_temperature": sd_temperature,
            "imu_temperature": imu_temperature,
            "base_imu_weight": base_imu_weight,
            "device_missing_rule": "weight = 0.4 * device_count / 5",
            "selection": "historical audited P12 median; no fold0 label tuning",
        },
        "metrics": {
            "skeleton": metric_dict(labels, skeleton_logits),
            "depth": metric_dict(labels, depth_logits),
            "sd_fixed_060_040": metric_dict(labels, sd_logits),
            "base_sd_rfimu": metric_dict(labels, base_logits),
        },
        "visual_inference": inference_info,
        "imu_rf": {
            "training_usable_samples": int(len(fit_rows)),
            "fit_seconds": rf_seconds,
            "path": str(rf_path),
            "sha256": sha256(rf_path),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
