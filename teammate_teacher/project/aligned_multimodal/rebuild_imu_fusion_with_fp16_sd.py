from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild existing S+D+IMU OOF and test logits after storage-FP16 "
            "Skeleton/Depth checkpoints are evaluated."
        )
    )
    parser.add_argument("--oof-base", type=Path, required=True)
    parser.add_argument("--oof-sd-root", type=Path, required=True)
    parser.add_argument("--imu-oof-summary", type=Path, required=True)
    parser.add_argument("--oof-output", type=Path, required=True)
    parser.add_argument("--test-base", type=Path, required=True)
    parser.add_argument("--test-sd", type=Path, required=True)
    parser.add_argument("--imu-final-summary", type=Path, required=True)
    parser.add_argument("--test-output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    return parser.parse_args()


def load_sd_oof(root: Path) -> dict[str, np.ndarray]:
    parts: dict[str, list[np.ndarray]] = {
        "sample_ids": [],
        "labels": [],
        "skeleton_logits": [],
        "depth_logits": [],
    }
    for fold in range(3):
        with np.load(
            root / f"fold_{fold}" / "logits_fp16.npz", allow_pickle=False
        ) as data:
            for key in parts:
                parts[key].append(data[key])
    result = {key: np.concatenate(value) for key, value in parts.items()}
    result["sample_ids"] = result["sample_ids"].astype(str)
    result["labels"] = result["labels"].astype(np.int64)
    return result


def main() -> None:
    args = parse_args()
    with np.load(args.oof_base.resolve(), allow_pickle=False) as data:
        oof_ids = data["sample_ids"].astype(str)
        labels = data["labels"].astype(np.int64)
        folds = data["folds"].astype(np.int64)
        old_sd_oof = data["sd_logits"].astype(np.float64)
        imu_oof = data["imu_logits"].astype(np.float64)
        old_fused_oof = data["fused_logits"].astype(np.float64)
    sd_oof = load_sd_oof(args.oof_sd_root.resolve())
    lookup = {
        sample_id: index
        for index, sample_id in enumerate(sd_oof["sample_ids"].tolist())
    }
    indices = np.asarray([lookup[sample_id] for sample_id in oof_ids])
    if not np.array_equal(sd_oof["labels"][indices], labels):
        raise ValueError("OOF labels differ after S+D alignment.")
    new_sd_oof = (
        0.6 * sd_oof["skeleton_logits"][indices].astype(np.float64)
        + 0.4 * sd_oof["depth_logits"][indices].astype(np.float64)
    )
    imu_oof_summary = json.loads(
        args.imu_oof_summary.resolve().read_text(encoding="utf-8")
    )
    protocols = imu_oof_summary["sources"]["stat_random_forest_device_dropout"][
        "cross_fitted_protocols"
    ]
    new_fused_oof = np.zeros_like(new_sd_oof)
    for protocol in protocols:
        fold = int(protocol["target_fold"])
        mask = folds == fold
        weight = float(protocol["selected_imu_weight"])
        new_fused_oof[mask] = (
            (1.0 - weight)
            * new_sd_oof[mask]
            / float(protocol["sd_temperature"])
            + weight
            * imu_oof[mask]
            / float(protocol["imu_temperature"])
        )
    oof_output = args.oof_output.resolve()
    oof_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        oof_output,
        sample_ids=oof_ids,
        labels=labels,
        folds=folds,
        sd_logits=new_sd_oof.astype(np.float32),
        imu_logits=imu_oof.astype(np.float32),
        fused_logits=new_fused_oof.astype(np.float32),
    )

    with np.load(args.test_base.resolve(), allow_pickle=False) as data:
        test_ids = data["sample_ids"].astype(str)
        old_sd_test = data["sd_logits"].astype(np.float64)
        imu_test = data["imu_logits"].astype(np.float64)
        old_fused_test = data["fused_logits"].astype(np.float64)
        device_counts = data["imu_device_counts"].astype(np.int64)
        weights = data["imu_weights"].astype(np.float64)
    with np.load(args.test_sd.resolve(), allow_pickle=False) as data:
        if not np.array_equal(data["sample_ids"].astype(str), test_ids):
            raise ValueError("Test S+D sample order differs.")
        new_sd_test = (
            0.6 * data["skeleton_logits"].astype(np.float64)
            + 0.4 * data["depth_logits"].astype(np.float64)
        )
    final_summary = json.loads(
        args.imu_final_summary.resolve().read_text(encoding="utf-8")
    )
    sd_temperature = float(final_summary["sd_temperature_oof_median"])
    imu_temperature = float(final_summary["imu_temperature_oof_median"])
    new_fused_test = (
        (1.0 - weights[:, None]) * new_sd_test / sd_temperature
        + weights[:, None] * imu_test / imu_temperature
    )
    test_output = args.test_output.resolve()
    test_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        test_output,
        sample_ids=test_ids,
        sd_logits=new_sd_test.astype(np.float32),
        imu_logits=imu_test.astype(np.float32),
        fused_logits=new_fused_test.astype(np.float32),
        imu_device_counts=device_counts,
        imu_weights=weights.astype(np.float32),
    )
    summary = {
        "method": (
            "Only S+D logits are regenerated from storage-FP16 checkpoints. "
            "The original IMU logits, cross-fitted protocols and final per-sample "
            "device-coverage weights are preserved."
        ),
        "oof": {
            "samples": len(oof_ids),
            "mean_abs_sd_logit_difference": float(
                np.abs(new_sd_oof - old_sd_oof).mean()
            ),
            "max_abs_sd_logit_difference": float(
                np.abs(new_sd_oof - old_sd_oof).max()
            ),
            "changed_sd_predictions": int(
                np.sum(new_sd_oof.argmax(1) != old_sd_oof.argmax(1))
            ),
            "changed_fused_predictions": int(
                np.sum(new_fused_oof.argmax(1) != old_fused_oof.argmax(1))
            ),
            "old_fused_accuracy": float(
                np.mean(old_fused_oof.argmax(1) == labels)
            ),
            "new_fused_accuracy": float(
                np.mean(new_fused_oof.argmax(1) == labels)
            ),
            "output": str(oof_output),
        },
        "test": {
            "samples": len(test_ids),
            "mean_abs_sd_logit_difference": float(
                np.abs(new_sd_test - old_sd_test).mean()
            ),
            "max_abs_sd_logit_difference": float(
                np.abs(new_sd_test - old_sd_test).max()
            ),
            "changed_sd_predictions": int(
                np.sum(new_sd_test.argmax(1) != old_sd_test.argmax(1))
            ),
            "changed_fused_predictions": int(
                np.sum(new_fused_test.argmax(1) != old_fused_test.argmax(1))
            ),
            "output": str(test_output),
        },
    }
    summary_path = args.summary.resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
