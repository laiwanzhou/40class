from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader

from aligned_data import AlignedMultimodalDataset
from imu_data import read_index
from probe_p27r3_incremental_information import metric_bundle, write_csv
from residual_logit_fusion import build_model
from run_imu_stat_baseline import (
    drop_devices,
    feature_vector,
    random_present_devices,
)


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_DIR = PROJECT_DIR / "data" / "p27_strong_inner"
DEFAULT_SKELETON_DIR = (
    PROJECT_DIR / "runs" / "p27_strong_inner" / "skeleton"
)
DEFAULT_IMU_CACHE = PROJECT_DIR / "cache" / "imu_32"
DEFAULT_OUTPUT = PROJECT_DIR / "runs" / "p27_strong_inner" / "skeleton_imu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate legal nested Skeleton + RF-IMU inner baseline"
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--skeleton-dir", type=Path, default=DEFAULT_SKELETON_DIR)
    parser.add_argument("--imu-cache", type=Path, default=DEFAULT_IMU_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=27091)
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


def infer_skeleton(
    checkpoint_path: Path,
    manifest_path: Path,
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config = checkpoint["config"]
    if list(config["modalities"]) != ["skeleton"]:
        raise ValueError("Expected a skeleton-only checkpoint")
    dataset = AlignedMultimodalDataset(
        manifest_path=manifest_path,
        split="val",
        modalities=["skeleton"],
        num_frames=int(config["num_frames"]),
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        augment=False,
        cache_dir=config.get("cache_dir"),
        skeleton_strategy=config.get("skeleton_strategy", "first"),
        skeleton_representation=config.get(
            "skeleton_representation", "frame_joint"
        ),
        skeleton_raw_cache_dir=config.get("skeleton_raw_cache_dir"),
    )
    manifest_rows = read_csv(manifest_path)
    subject_by_id = {
        row["sample_id"]: row["user_id"]
        for row in manifest_rows
        if row["split"] == "val"
    }
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    model = build_model(checkpoint, device)
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    sample_ids: list[str] = []
    labels: list[int] = []
    logits: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(
                    {"skeleton": batch["skeleton"].to(device, non_blocking=True)}
                )
            logits.append(output.float().cpu())
            sample_ids.extend(batch["sample_id"])
            labels.extend(batch["label"].tolist())
    missing_subjects = [
        sample_id for sample_id in sample_ids if sample_id not in subject_by_id
    ]
    if missing_subjects:
        raise RuntimeError(
            f"{len(missing_subjects)} validation sample_ids have no manifest subject"
        )
    subjects = [subject_by_id[sample_id] for sample_id in sample_ids]
    info = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "fp16_parameter_mib": float(
            sum(value.numel() for value in model.parameters()) * 2 / 1024**2
        ),
        "inference_seconds": time.perf_counter() - started,
    }
    return (
        np.asarray(sample_ids),
        np.asarray(labels, dtype=np.int64),
        np.asarray(subjects),
        torch.cat(logits).numpy(),
        info,
    )


def dense_log_probabilities(
    model: RandomForestClassifier, source: np.ndarray
) -> np.ndarray:
    probabilities = model.predict_proba(source)
    logits = np.full((len(source), 40), np.log(1e-12), dtype=np.float32)
    logits[:, model.classes_.astype(np.int64)] = np.log(
        np.clip(probabilities, 1e-12, 1.0)
    )
    return logits


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    imu_cache = args.imu_cache.resolve()
    imu_rows = [
        row
        for row in read_index(imu_cache / "index.csv")
        if row.split == "train"
    ]
    values = np.load(
        imu_cache / "imu_float32.npy", mmap_mode="r", allow_pickle=False
    )
    time_mask = np.load(
        imu_cache / "time_mask_uint8.npy", mmap_mode="r", allow_pickle=False
    )
    device_mask = np.load(
        imu_cache / "device_mask_uint8.npy", mmap_mode="r", allow_pickle=False
    )
    feature_by_id: dict[str, np.ndarray] = {}
    mask_by_id: dict[str, np.ndarray] = {}
    imu_meta_by_id = {row.sample_id: row for row in imu_rows}
    for row in imu_rows:
        feature, mask = feature_vector(
            values[row.cache_index],
            time_mask[row.cache_index],
            device_mask[row.cache_index],
        )
        feature_by_id[row.sample_id] = feature
        mask_by_id[row.sample_id] = mask

    # Fixed before this nested run: medians of the audited P12 RF protocol.
    skeleton_temperature = float(
        np.median([0.9167410586298098, 0.8908255030742397, 0.9401831574570029])
    )
    imu_temperature = float(
        np.median([0.6363677656203999, 0.5918757644479288, 0.6312607866282122])
    )
    imu_weight = 0.4
    fold_rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    fold_summaries: dict[str, Any] = {}
    for fold in range(3):
        manifest_path = args.manifest_dir.resolve() / f"fold_{fold}.csv"
        manifest_rows = read_csv(manifest_path)
        train_subjects = sorted(
            {row["user_id"] for row in manifest_rows if row["split"] == "train"}
        )
        val_subjects = sorted(
            {row["user_id"] for row in manifest_rows if row["split"] == "val"}
        )
        (
            sample_ids,
            labels,
            subjects,
            skeleton_logits,
            skeleton_info,
        ) = infer_skeleton(
            args.skeleton_dir.resolve() / f"fold_{fold}" / "best_accuracy.pt",
            manifest_path,
            device,
            int(args.num_workers),
        )

        fit_rows = [
            row
            for row in imu_rows
            if row.usable and row.user_id in set(train_subjects)
        ]
        fit_features = np.stack(
            [feature_by_id[row.sample_id] for row in fit_rows]
        )
        fit_masks = np.stack([mask_by_id[row.sample_id] for row in fit_rows])
        fit_labels = np.asarray(
            [row.class_id for row in fit_rows], dtype=np.int64
        )
        rng = np.random.default_rng(int(args.seed) + fold)
        dropped_features, dropped_masks = drop_devices(
            fit_features,
            fit_masks,
            random_present_devices(fit_masks, rng),
        )
        source = np.concatenate(
            [
                np.concatenate([fit_features, fit_masks], axis=1),
                np.concatenate([dropped_features, dropped_masks], axis=1),
            ],
            axis=0,
        )
        source_labels = np.concatenate([fit_labels, fit_labels])
        forest = RandomForestClassifier(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=int(args.seed) + fold,
        )
        fit_started = time.perf_counter()
        forest.fit(source, source_labels)
        fit_seconds = time.perf_counter() - fit_started
        model_path = output / f"fold_{fold}_imu_rf.joblib"
        joblib.dump(forest, model_path, compress=3)

        imu_logits = np.zeros((len(sample_ids), 40), dtype=np.float32)
        imu_present = np.zeros(len(sample_ids), dtype=bool)
        imu_device_counts = np.zeros(len(sample_ids), dtype=np.int64)
        present_indices = [
            index
            for index, sample_id in enumerate(sample_ids.astype(str))
            if sample_id in imu_meta_by_id
            and imu_meta_by_id[sample_id].usable
        ]
        if present_indices:
            present_source = np.stack(
                [
                    np.concatenate(
                        [
                            feature_by_id[str(sample_ids[index])],
                            mask_by_id[str(sample_ids[index])],
                        ]
                    )
                    for index in present_indices
                ]
            )
            imu_logits[present_indices] = dense_log_probabilities(
                forest, present_source
            )
            imu_present[present_indices] = True
            imu_device_counts[present_indices] = [
                imu_meta_by_id[str(sample_ids[index])].device_count
                for index in present_indices
            ]

        per_sample_weight = (
            imu_weight * np.clip(imu_device_counts / 5.0, 0.0, 1.0)
        ).astype(np.float32)
        fused_logits = skeleton_logits / skeleton_temperature
        fused_logits = (
            (1.0 - per_sample_weight[:, None]) * fused_logits
            + per_sample_weight[:, None] * imu_logits / imu_temperature
        )
        methods = {
            "skeleton": skeleton_logits.argmax(1),
            "imu_present_only": np.where(
                imu_present, imu_logits.argmax(1), skeleton_logits.argmax(1)
            ),
            "skeleton_imu_fixed": fused_logits.argmax(1),
        }
        for method, predictions in methods.items():
            metrics = metric_bundle(labels, predictions)
            fold_rows.append(
                {
                    "inner_fold": fold,
                    "method": method,
                    **{
                        f"{subset}_{key}": value
                        for subset, values_dict in metrics.items()
                        for key, value in values_dict.items()
                    },
                }
            )
            for subject in val_subjects:
                selected = subjects == subject
                subject_metrics = metric_bundle(
                    labels[selected], predictions[selected]
                )["overall"]
                subject_rows.append(
                    {
                        "inner_fold": fold,
                        "method": method,
                        "subject": subject,
                        **subject_metrics,
                    }
                )
        np.savez_compressed(
            output / f"fold_{fold}_logits.npz",
            protocol=np.asarray("p27-strong-skeleton-rfimu-inner-v1"),
            sample_ids=sample_ids,
            labels=labels,
            subjects=subjects,
            skeleton_logits=skeleton_logits.astype(np.float32),
            imu_logits=imu_logits,
            fused_logits=fused_logits.astype(np.float32),
            imu_present=imu_present,
            imu_device_counts=imu_device_counts,
            outer_held_predictions_generated=np.asarray(False),
        )
        fold_summaries[str(fold)] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "train_subjects": train_subjects,
            "val_subjects": val_subjects,
            "samples": int(len(labels)),
            "imu_train_usable": int(len(fit_rows)),
            "imu_val_usable": int(imu_present.sum()),
            "skeleton": skeleton_info,
            "imu_rf_fit_seconds": fit_seconds,
            "imu_rf_path": str(model_path),
            "imu_rf_bytes": model_path.stat().st_size,
            "imu_rf_sha256": sha256(model_path),
            "outer_held_predictions_generated": False,
        }
        print(
            f"fold={fold} skeleton={np.mean(methods['skeleton'] == labels):.4f} "
            f"skeleton_imu={np.mean(methods['skeleton_imu_fixed'] == labels):.4f}",
            flush=True,
        )

    write_csv(output / "fold_metrics.csv", fold_rows)
    write_csv(output / "per_subject.csv", subject_rows)
    mean_metrics: dict[str, dict[str, float]] = {}
    for method in sorted({str(row["method"]) for row in fold_rows}):
        selected = [row for row in fold_rows if row["method"] == method]
        mean_metrics[method] = {
            key: float(np.mean([float(row[key]) for row in selected]))
            for key in selected[0]
            if key not in {"inner_fold", "method"}
        }
    summary = {
        "protocol": "p27-strong-skeleton-rfimu-inner-v1",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "fixed_fusion": {
            "source": "median of audited P12 stat_random_forest_device_dropout protocols; fixed before current inner evaluation",
            "skeleton_temperature": skeleton_temperature,
            "imu_temperature": imu_temperature,
            "base_imu_weight": imu_weight,
            "missing_rule": "weight multiplied by device_count/5; zero devices exactly fall back to Skeleton",
        },
        "mean_metrics": mean_metrics,
        "folds": fold_summaries,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
