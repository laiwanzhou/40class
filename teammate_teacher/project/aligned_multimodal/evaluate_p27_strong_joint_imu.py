from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from aligned_data import AlignedMultimodalDataset
from probe_p27r3_incremental_information import metric_bundle, write_csv
from residual_logit_fusion import build_model


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST_DIR = PROJECT_DIR / "data" / "p27_strong_inner"
DEFAULT_CHECKPOINT_DIR = (
    PROJECT_DIR / "runs" / "p27_strong_inner" / "depth_skeleton"
)
DEFAULT_IMU_DIR = (
    PROJECT_DIR / "runs" / "p27_strong_inner" / "skeleton_imu"
)
DEFAULT_OUTPUT = (
    PROJECT_DIR / "runs" / "p27_strong_inner" / "depth_skeleton_imu"
)

# P12 fold-pure S+D / stat-RF calibration. It is fixed external evidence, not
# selected on the current P27 inner validation labels.
P12_PROTOCOLS = {
    0: (0.9167410586298098, 0.6363677656203999, 0.4),
    1: (0.8908255030742397, 0.5918757644479288, 0.4),
    2: (0.9401831574570029, 0.6312607866282122, 0.4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fold-pure joint visual/Skeleton model with RF-IMU"
    )
    parser.add_argument("--manifest-dir", type=Path, default=DEFAULT_MANIFEST_DIR)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument(
        "--checkpoint-template",
        default="fold_{fold}/best_accuracy.pt",
        help=(
            "Path below --checkpoint-dir. Supports {fold}; this keeps exploratory "
            "run layouts auditable without copying checkpoints."
        ),
    )
    parser.add_argument("--imu-dir", type=Path, default=DEFAULT_IMU_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, nargs="+", default=[0])
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


def flatten_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        f"{subset}_{key}": value
        for subset, values in metrics.items()
        for key, value in values.items()
    }


def infer_checkpoint(
    checkpoint_path: Path,
    manifest_path: Path,
    device: torch.device,
    num_workers: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    modalities = list(config["modalities"])
    if "skeleton" not in modalities or not ({"depth", "ir"} & set(modalities)):
        raise ValueError(
            "Expected Skeleton plus at least one visual modality, "
            f"found {modalities}"
        )
    dataset = AlignedMultimodalDataset(
        manifest_path=manifest_path,
        split="val",
        modalities=modalities,
        num_frames=int(config["num_frames"]),
        image_height=int(config["image_height"]),
        image_width=int(config["image_width"]),
        augment=False,
        cache_dir=config.get("cache_dir"),
        skeleton_strategy=config.get("skeleton_strategy", "first"),
        depth_representation=config.get("depth_representation", "jet_rgb"),
        visual_normalization=config.get("visual_normalization", "legacy"),
        skeleton_representation=config.get(
            "skeleton_representation", "frame_joint"
        ),
        skeleton_raw_cache_dir=config.get("skeleton_raw_cache_dir"),
        ir_motion_mode=config.get("ir_motion_mode", "none"),
        ir_roi_csv=config.get("ir_roi_csv"),
        ir_roi_context=float(config.get("ir_roi_context", 0.3)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(config.get("batch_size", 32)),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    subject_by_id = {
        row["sample_id"]: row["user_id"]
        for row in read_csv(manifest_path)
        if row["split"] == "val"
    }
    model = build_model(checkpoint, device)
    use_amp = bool(config.get("use_amp", True) and device.type == "cuda")
    sample_ids: list[str] = []
    labels: list[int] = []
    logits: list[torch.Tensor] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                modality: batch[modality].to(device, non_blocking=True)
                for modality in modalities
            }
            if bool(config.get("use_ir_motion", False)):
                inputs["ir_motion"] = batch["ir_motion"].to(
                    device, non_blocking=True
                )
            if bool(config.get("use_ir_local", False)):
                inputs["ir_local"] = batch["ir_local"].to(
                    device, non_blocking=True
                )
                inputs["ir_local_quality"] = batch["ir_local_quality"].to(
                    device, non_blocking=True
                )
            with torch.amp.autocast(
                device_type=device.type, dtype=torch.float16, enabled=use_amp
            ):
                output = model(inputs)
            if not isinstance(output, torch.Tensor):
                raise TypeError("Expected tensor logits")
            logits.append(output.float().cpu())
            sample_ids.extend(batch["sample_id"])
            labels.extend(batch["label"].tolist())
    missing = [sample_id for sample_id in sample_ids if sample_id not in subject_by_id]
    if missing:
        raise RuntimeError(f"{len(missing)} samples have no manifest subject")
    info = {
        "modalities": modalities,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "parameters": int(sum(value.numel() for value in model.parameters())),
        "fp16_parameter_mib": float(
            sum(value.numel() for value in model.parameters()) * 2 / 1024**2
        ),
        "inference_seconds": float(time.perf_counter() - started),
    }
    return (
        np.asarray(sample_ids),
        np.asarray(labels, dtype=np.int64),
        np.asarray([subject_by_id[sample_id] for sample_id in sample_ids]),
        torch.cat(logits).numpy(),
        info,
    )


def align_imu(
    archive: np.lib.npyio.NpzFile,
    sample_ids: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if bool(archive["outer_held_predictions_generated"]):
        raise RuntimeError("IMU archive unexpectedly contains outer-held predictions")
    position = {
        str(sample_id): index
        for index, sample_id in enumerate(archive["sample_ids"].astype(str))
    }
    indices = np.asarray([position[str(sample_id)] for sample_id in sample_ids])
    if not np.array_equal(archive["labels"][indices], labels):
        raise RuntimeError("Joint and IMU labels do not align")
    return (
        archive["imu_logits"][indices].astype(np.float32),
        archive["imu_device_counts"][indices].astype(np.int64),
        archive["imu_present"][indices].astype(bool),
    )


def main() -> None:
    args = parse_args()
    folds = sorted(set(int(fold) for fold in args.folds))
    if any(fold not in P12_PROTOCOLS for fold in folds):
        raise ValueError(f"Unsupported folds: {folds}")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, Any]] = []
    subject_rows: list[dict[str, Any]] = []
    fold_info: dict[str, Any] = {}
    for fold in folds:
        manifest_path = args.manifest_dir.resolve() / f"fold_{fold}.csv"
        checkpoint_path = (
            args.checkpoint_dir.resolve()
            / str(args.checkpoint_template).format(fold=fold)
        )
        sample_ids, labels, subjects, joint_logits, model_info = infer_checkpoint(
            checkpoint_path, manifest_path, device, int(args.num_workers)
        )
        imu_archive = np.load(
            args.imu_dir.resolve() / f"fold_{fold}_logits.npz",
            allow_pickle=False,
        )
        imu_logits, device_counts, imu_present = align_imu(
            imu_archive, sample_ids, labels
        )
        joint_temperature, imu_temperature, base_weight = P12_PROTOCOLS[fold]
        weights = (
            base_weight * np.clip(device_counts.astype(np.float32) / 5.0, 0.0, 1.0)
        )[:, None]
        fused_logits = (
            (1.0 - weights) * joint_logits / joint_temperature
            + weights * imu_logits / imu_temperature
        )
        modality_tag = "_".join(model_info["modalities"])
        methods = {
            f"{modality_tag}_joint": joint_logits,
            f"{modality_tag}_joint_rfimu_fixed": fused_logits,
        }
        for method, logits in methods.items():
            predictions = logits.argmax(axis=1)
            rows.append(
                {
                    "inner_fold": fold,
                    "method": method,
                    **flatten_metrics(metric_bundle(labels, predictions)),
                }
            )
            for subject in sorted(set(subjects.tolist())):
                selected = subjects == subject
                subject_rows.append(
                    {
                        "inner_fold": fold,
                        "method": method,
                        "subject": subject,
                        **metric_bundle(
                            labels[selected], predictions[selected]
                        )["overall"],
                    }
                )
        np.savez_compressed(
            output / f"fold_{fold}_logits.npz",
            protocol=np.asarray("p27-strong-visual-skeleton-rfimu-inner-v2"),
            sample_ids=sample_ids,
            labels=labels,
            subjects=subjects,
            joint_logits=joint_logits.astype(np.float32),
            imu_logits=imu_logits,
            fused_logits=fused_logits.astype(np.float32),
            imu_present=imu_present,
            imu_device_counts=device_counts,
            outer_held_predictions_generated=np.asarray(False),
        )
        fold_info[str(fold)] = {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256(manifest_path),
            "model": model_info,
            "p12_fixed_protocol": {
                "joint_temperature": joint_temperature,
                "imu_temperature": imu_temperature,
                "base_weight": base_weight,
            },
            "samples": int(len(labels)),
            "imu_present_samples": int(imu_present.sum()),
            "outer_held_predictions_generated": False,
        }
        print(
            f"fold={fold} joint={np.mean(joint_logits.argmax(1) == labels):.4f} "
            f"joint_rfimu={np.mean(fused_logits.argmax(1) == labels):.4f}",
            flush=True,
        )

    write_csv(output / "fold_metrics.csv", rows)
    write_csv(output / "per_subject.csv", subject_rows)
    summary = {
        "protocol": "p27-strong-visual-skeleton-rfimu-inner-v2",
        "outer_fold": 0,
        "outer_train_only": True,
        "outer_held_predictions_generated": False,
        "evaluated_inner_folds": folds,
        "folds": fold_info,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
